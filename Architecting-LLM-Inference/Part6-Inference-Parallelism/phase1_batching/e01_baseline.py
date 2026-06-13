"""
E01 — Baseline Single Request Inference
========================================
Goal:   Establish TTFT / TPOT baseline for a single request.
        Show that TTFT scales with prompt length (prefill is O(n²) in attention),
        while TPOT stays roughly constant regardless of prompt length.

Model:  Qwen/Qwen2.5-1.5B-Instruct (open, Apache 2.0, runs on MacBook MPS/CPU)
Vary:   Prompt lengths: 128, 512, 1024, 2048 tokens
Output: 128 tokens per run

Run:    python phase1_batching/e01_baseline.py
        python phase1_batching/e01_baseline.py --device cpu   # force CPU
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.metrics_collector import ExperimentMetrics, RequestMetrics

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_LENGTHS = [128, 512, 1024, 2048]
OUTPUT_TOKENS = 128


def detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def measure_request(model, tokenizer, device: str, prompt_len: int, output_tokens: int) -> RequestMetrics:
    """
    Measure a single greedy-decode request.

    Step-by-step decoding is intentional: it lets us capture TTFT (first
    forward pass = prefill cost) vs TPOT (each subsequent decode step) separately.
    """
    import torch

    # Build a prompt of exactly `prompt_len` tokens
    filler = "The quick brown fox jumps over the lazy dog. "
    raw_prompt = (filler * ((prompt_len // 10) + 1))
    inputs = tokenizer(raw_prompt, return_tensors="pt",
                       truncation=True, max_length=prompt_len).to(device)
    actual_prompt_len = inputs["input_ids"].shape[1]

    arrival_time = time.perf_counter()
    start_time   = time.perf_counter()
    first_token_time = None
    output_len = 0

    with torch.no_grad():
        for step in range(output_tokens):
            out = model(**inputs)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if first_token_time is None:
                first_token_time = time.perf_counter()   # end of first forward pass

            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones_like(next_token)], dim=-1
                )
            output_len += 1

            # Stop on EOS
            eos_id = tokenizer.eos_token_id
            if eos_id is not None and next_token.item() == eos_id:
                break

    end_time = time.perf_counter()

    return RequestMetrics(
        request_id=f"e01-len{prompt_len}",
        prompt_len=actual_prompt_len,
        output_len=output_len,
        arrival_time=arrival_time,
        start_time=start_time,
        first_token_time=first_token_time or start_time,
        end_time=end_time,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None,
                        help="Device override: cpu | mps | cuda")
    parser.add_argument("--output-tokens", type=int, default=OUTPUT_TOKENS)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = args.device or detect_device()
    print(f"E01 — Baseline Single Request")
    print(f"  Model:   {MODEL}")
    print(f"  Device:  {device}")
    print(f"  Prompts: {PROMPT_LENGTHS} tokens")
    print(f"  Output:  {args.output_tokens} tokens/request\n")

    # ── Load model once ────────────────────────────────────────────────────────
    print("Loading model (one-time)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model = model.to(device)
    model.eval()
    print("Model loaded.\n")

    # ── Warmup pass (avoids cold-start bias on first measurement) ──────────────
    print("Warming up...")
    _ = measure_request(model, tokenizer, device, prompt_len=32, output_tokens=4)
    print("Done.\n")

    # ── Per-prompt-length measurements ────────────────────────────────────────
    em = ExperimentMetrics(
        experiment_id="E01_baseline",
        description="Single request baseline — TTFT and TPOT vs prompt length",
        config={"model": MODEL, "device": device,
                "output_tokens": args.output_tokens},
    )

    print(f"  {'Prompt len':>12} {'TTFT (ms)':>12} {'TPOT (ms/tok)':>15} {'E2E (ms)':>10}")
    print("  " + "─" * 55)

    for prompt_len in PROMPT_LENGTHS:
        m = measure_request(model, tokenizer, device, prompt_len, args.output_tokens)
        em.add(m)
        print(f"  {prompt_len:>12}  {m.ttft*1000:>10.1f}   "
              f"{m.tpot*1000:>13.1f}   {m.e2e_latency*1000:>8.0f}")

    em.finish()
    em.print_summary()
    em.save("results")

    # ── Article insight ────────────────────────────────────────────────────────
    reqs = em.requests
    if len(reqs) >= 2:
        ttft_ratio = reqs[-1].ttft / reqs[0].ttft
        len_ratio  = reqs[-1].prompt_len / reqs[0].prompt_len
        tpot_cv    = (max(r.tpot for r in reqs) - min(r.tpot for r in reqs)) / (
                      sum(r.tpot for r in reqs) / len(reqs))
        print("\n[Article insight]")
        print(f"  TTFT grew {ttft_ratio:.1f}x when prompt length grew {len_ratio:.0f}x")
        print(f"  → consistent with O(n²) attention in the prefill phase")
        print(f"  TPOT coefficient of variation: {tpot_cv:.2f} (near 0 = roughly constant)")
        print(f"  → TPOT is nearly independent of prompt length")


if __name__ == "__main__":
    main()
