"""
Shared GPU utilities for E06B-E09B (real-model Phase B).

These replace the synthetic proxies in retention_sim.py with actual model
forward passes and actual KV-cache manipulation. Requires a CUDA GPU,
`torch`, and `transformers` -- written for RunPod, not this repo's authoring
environment (which has neither a GPU nor internet access, so this file was
written carefully but NOT executed before being committed -- see the note
at the bottom of the code README about verifying it on your pod and
reporting back anything that breaks).

VERSION SENSITIVITY: the HF `DynamicCache` internal layout
(`cache.key_cache[layer_idx]`, `cache.value_cache[layer_idx]`, both shaped
(batch, num_kv_heads, seq_len, head_dim)) is what this code targets. That
layout is accurate for transformers >=4.40,<4.47 roughly. If you're on a
newer transformers where `Cache` was refactored (e.g. a `.layers` attribute
replacing `.key_cache`/`.value_cache`), the `_get_layer_cache` helper below
is the one place to patch -- everything else calls through it.

Model: Qwen/Qwen2.5-0.5B-Instruct by default -- small enough to iterate
quickly and cheaply on a single RTX 3090, GQA architecture (consistent with
the rest of Part 2's model choices), open weights, no gating.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _require_torch_and_transformers():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "This experiment needs `torch` and `transformers` with a CUDA GPU. "
            "Run it on your RunPod pod after `pip install -r requirements-runpod.txt "
            "torch transformers`."
        ) from e


def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL, device: str = "cuda"):
    _require_torch_and_transformers()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tok


def _get_layer_cache(cache, layer_idx: int):
    """Returns (key_tensor, value_tensor) for one layer. One patch point if your
    transformers version renamed DynamicCache's internals."""
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def _set_layer_cache(cache, layer_idx: int, new_key, new_value):
    cache.key_cache[layer_idx] = new_key
    cache.value_cache[layer_idx] = new_value


def num_layers_in_cache(cache) -> int:
    return len(cache.key_cache)


def current_cache_len(cache) -> int:
    if num_layers_in_cache(cache) == 0 or cache.key_cache[0] is None:
        return 0
    return cache.key_cache[0].shape[-2]   # (batch, kv_heads, seq_len, head_dim)


# -- Cache policies: mutate a DynamicCache in place, after each decode step ----

def make_sliding_window_policy(window_size: int) -> Callable:
    def policy(cache, step: int, last_step_attentions=None, importance_state=None):
        seq_len = current_cache_len(cache)
        if seq_len <= window_size:
            return
        for layer_idx in range(num_layers_in_cache(cache)):
            k, v = _get_layer_cache(cache, layer_idx)
            _set_layer_cache(cache, layer_idx, k[..., -window_size:, :], v[..., -window_size:, :])
    return policy


def make_attention_sink_policy(num_sinks: int, window_size: int) -> Callable:
    """Keeps the first `num_sinks` positions plus a trailing window of
    (window_size - num_sinks) most recent positions."""
    trailing = max(window_size - num_sinks, 1)

    def policy(cache, step: int, last_step_attentions=None, importance_state=None):
        import torch
        seq_len = current_cache_len(cache)
        if seq_len <= window_size:
            return
        for layer_idx in range(num_layers_in_cache(cache)):
            k, v = _get_layer_cache(cache, layer_idx)
            k_sink, v_sink = k[..., :num_sinks, :], v[..., :num_sinks, :]
            k_tail, v_tail = k[..., -trailing:, :], v[..., -trailing:, :]
            new_k = torch.cat([k_sink, k_tail], dim=-2)
            new_v = torch.cat([v_sink, v_tail], dim=-2)
            _set_layer_cache(cache, layer_idx, new_k, new_v)
    return policy


def make_heavy_hitter_policy(cache_budget: int, recent_protect: int = 32) -> Callable:
    """
    H2O-style: track cumulative attention received by each cached position
    (averaged over heads and layers each step), and once the cache exceeds
    `cache_budget`, keep the most recent `recent_protect` positions plus the
    highest-cumulative-score remaining positions, dropping the rest.

    `importance_state` is a dict threaded through by the caller (see
    perplexity_under_policy) holding a running score tensor aligned to the
    CURRENT cache positions -- it must be re-indexed identically to however
    the cache itself gets pruned, or the scores and positions will drift
    out of alignment. That re-indexing happens inside this function.
    """
    def policy(cache, step: int, last_step_attentions=None, importance_state=None):
        import torch

        seq_len = current_cache_len(cache)
        if importance_state is None:
            raise ValueError("heavy_hitter_policy requires importance_state to be threaded through")

        # Update running scores with this step's attention (if provided).
        if last_step_attentions is not None:
            # last_step_attentions: tuple of (batch, num_heads, q_len=1, kv_len) per layer
            # average over layers and heads -> (kv_len,)
            stacked = torch.stack([a.mean(dim=(0, 1, 2)) for a in last_step_attentions], dim=0)
            step_scores = stacked.mean(dim=0)   # (kv_len,)
            if importance_state.get("scores") is None or importance_state["scores"].shape[0] != step_scores.shape[0] - 1:
                # first call, or lengths don't line up (e.g. right after prefill) -- (re)initialize
                importance_state["scores"] = torch.zeros(step_scores.shape[0], device=step_scores.device)
            else:
                # scores tracked len should be kv_len - 1 before this new token was appended
                importance_state["scores"] = torch.cat([
                    importance_state["scores"], torch.zeros(1, device=step_scores.device),
                ])
            importance_state["scores"] += step_scores

        if seq_len <= cache_budget:
            return

        scores = importance_state.get("scores")
        if scores is None or scores.shape[0] != seq_len:
            # No score signal available yet -- fall back to pure recency this step.
            keep_idx = torch.arange(seq_len - cache_budget, seq_len)
        else:
            recent_start = seq_len - recent_protect
            recent_idx = torch.arange(recent_start, seq_len)
            candidate_idx = torch.arange(0, recent_start)
            candidate_scores = scores[:recent_start]
            num_to_keep = max(cache_budget - recent_protect, 0)
            if num_to_keep > 0 and candidate_idx.numel() > 0:
                topk = torch.topk(candidate_scores, k=min(num_to_keep, candidate_idx.numel())).indices
                kept_candidates = candidate_idx[topk]
                keep_idx = torch.cat([kept_candidates, recent_idx]).sort().values
            else:
                keep_idx = recent_idx

        for layer_idx in range(num_layers_in_cache(cache)):
            k, v = _get_layer_cache(cache, layer_idx)
            _set_layer_cache(cache, layer_idx, k[..., keep_idx, :], v[..., keep_idx, :])
        if scores is not None and scores.shape[0] == seq_len:
            importance_state["scores"] = scores[keep_idx]

    return policy


def make_quantization_policy(bits: int, every_n_steps: int = 1) -> Callable:
    """Fake-quantizes (round-trips through reduced precision) the cache's
    real K/V tensors every `every_n_steps` decode steps -- simulates the
    numerical effect of storing KV cache at reduced bit-width."""
    def policy(cache, step: int, last_step_attentions=None, importance_state=None):
        import torch
        if step % every_n_steps != 0:
            return
        levels = 2 ** bits
        for layer_idx in range(num_layers_in_cache(cache)):
            k, v = _get_layer_cache(cache, layer_idx)
            for tensor, setter in ((k, "k"), (v, "v")):
                t_min, t_max = tensor.min(), tensor.max()
                scale = (t_max - t_min) / max(levels - 1, 1)
                if scale == 0:
                    continue
                q = torch.round((tensor - t_min) / scale)
                dq = (q * scale + t_min).to(tensor.dtype)
                if setter == "k":
                    k = dq
                else:
                    v = dq
            _set_layer_cache(cache, layer_idx, k, v)
    return policy


# -- Perplexity-under-policy harness --------------------------------------------

@dataclass
class PolicyPerplexityResult:
    policy_name: str
    warmup_tokens: int
    measured_tokens: int
    avg_nll: float
    perplexity: float
    final_cache_len: int


def perplexity_under_policy(
    model, tokenizer, text: str,
    policy_fn: Optional[Callable] = None,
    warmup_tokens: int = 256,
    max_tokens: int = 1024,
    needs_attentions: bool = False,
    device: str = "cuda",
) -> PolicyPerplexityResult:
    """
    Teacher-forced pass over `text`: the first `warmup_tokens` build up the
    cache with NO policy applied (establishing context). From there through
    `max_tokens`, at each step we predict the next real token from the
    model's logits (measuring NLL), then apply `policy_fn` to the cache
    (simulating that eviction/quantization policy running continuously
    during a long generation), and feed the REAL next token in for the next
    step (teacher forcing, not sampling -- we want to measure the policy's
    effect on predicting the actual continuation, not compounding sampling
    errors).
    """
    import torch
    from transformers import DynamicCache

    ids = tokenizer(text, return_tensors="pt", truncation=True,
                     max_length=warmup_tokens + max_tokens + 1).input_ids.to(device)
    total_len = ids.shape[1]
    if total_len < warmup_tokens + 2:
        raise ValueError(f"Text too short ({total_len} tokens) for warmup_tokens={warmup_tokens}")

    measured_len = min(max_tokens, total_len - warmup_tokens - 1)

    cache = DynamicCache()
    importance_state = {}

    with torch.no_grad():
        # Warmup: build context, no policy applied yet.
        warmup_ids = ids[:, :warmup_tokens]
        out = model(warmup_ids, past_key_values=cache, use_cache=True,
                     output_attentions=needs_attentions)
        cache = out.past_key_values

        total_nll = 0.0
        count = 0
        cur_input = ids[:, warmup_tokens:warmup_tokens + 1]

        for step in range(measured_len):
            out = model(cur_input, past_key_values=cache, use_cache=True,
                         output_attentions=needs_attentions)
            logits = out.logits[:, -1, :]   # (batch, vocab)
            target = ids[:, warmup_tokens + step + 1]
            nll = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
            total_nll += nll.item()
            count += 1

            cache = out.past_key_values
            if policy_fn is not None:
                attns = out.attentions if needs_attentions else None
                policy_fn(cache, step, last_step_attentions=attns, importance_state=importance_state)

            cur_input = ids[:, warmup_tokens + step + 1: warmup_tokens + step + 2]

    avg_nll = total_nll / count
    return PolicyPerplexityResult(
        policy_name=getattr(policy_fn, "__name__", "none") if policy_fn else "full_cache",
        warmup_tokens=warmup_tokens,
        measured_tokens=count,
        avg_nll=avg_nll,
        perplexity=math.exp(avg_nll),
        final_cache_len=current_cache_len(cache),
    )


def load_long_text(tokenizer, min_tokens: int = 2000) -> str:
    """
    Small built-in synthetic-but-coherent-enough long text for perplexity
    measurement, so these scripts don't require downloading a dataset.
    Repeats a short passage with light variation -- good enough to exercise
    cache policies; swap in a real long document (wikitext, a long article)
    for more meaningful absolute perplexity numbers.
    """
    passage = (
        "The transformer architecture processes sequences by computing attention "
        "between every pair of tokens, which lets the model relate distant parts "
        "of the input directly rather than through a chain of recurrent steps. "
        "Each layer refines the representation of every token using this "
        "attention mechanism combined with a position-wise feed-forward network. "
    )
    text = passage
    while len(tokenizer(text).input_ids) < min_tokens:
        text += passage
    return text
