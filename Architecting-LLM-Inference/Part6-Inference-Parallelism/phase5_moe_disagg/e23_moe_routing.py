"""
E23 — MoE Routing Analysis
============================
Goal:   Visualize expert selection patterns in Mixtral.
        Which experts fire most often? Does routing collapse occur?
        What is the routing entropy across tokens?
Run on: RunPod (single GPU, Mixtral 8x7B GPTQ)
"""

import torch
import json
import math
import collections
from pathlib import Path

MODEL = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
RESULTS_DIR = Path("results/e23_moe_routing")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "Write a Python function to sort a list.",
    "What is the capital of France?",
    "Solve the equation 2x + 5 = 13.",
    "Translate 'hello' to Spanish.",
    "Explain quantum entanglement.",
    "Write a haiku about autumn.",
    "What is the derivative of x^2?",
    "Summarize the French Revolution.",
] * 4   # 32 prompts


def routing_entropy(counts: dict, total: int) -> float:
    """Shannon entropy of expert selection distribution."""
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def analyze_routing():
    """
    Hook into Mixtral's MoE router to log which experts are selected per token.
    Uses transformers library directly (not vLLM) for easier hook access.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Load with auto device map for single GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # ── Hook to capture router logits ────────────────────────────────────────
    expert_selections = collections.defaultdict(list)   # layer_idx → [expert_id, ...]

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output[1] is the routing weights / expert indices in MoE layers
            if hasattr(output, "router_logits") and output.router_logits is not None:
                logits = output.router_logits   # (batch*seq, num_experts)
                _, selected = torch.topk(logits, k=2, dim=-1)
                for expert_id in selected.reshape(-1).tolist():
                    expert_selections[layer_idx].append(expert_id)
        return hook

    # Register hooks on MoE layers
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "block_sparse_moe"):
            h = layer.block_sparse_moe.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

    print(f"Registered hooks on {len(hooks)} MoE layers.")

    # ── Run inference ──────────────────────────────────────────────────────────
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            _ = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            if (i + 1) % 8 == 0:
                print(f"  Processed {i+1}/{len(PROMPTS)} prompts...")

    for h in hooks:
        h.remove()

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n── Expert Utilization per Layer ────────────────────────────────────")
    results = {}

    for layer_idx, selections in sorted(expert_selections.items()):
        counts = collections.Counter(selections)
        total = sum(counts.values())
        entropy = routing_entropy(counts, total)
        max_expert_pct = max(counts.values()) / total * 100
        min_expert_pct = min(counts.values()) / total * 100

        results[layer_idx] = {
            "total_selections": total,
            "expert_counts": dict(counts),
            "entropy_bits": entropy,
            "max_expert_pct": max_expert_pct,
            "min_expert_pct": min_expert_pct,
            "is_collapsed": max_expert_pct > 50,   # >50% → routing collapse risk
        }

        print(f"  Layer {layer_idx:2d}: entropy={entropy:.2f} bits  "
              f"max_expert={max_expert_pct:.1f}%  "
              f"{'⚠ COLLAPSE RISK' if max_expert_pct > 50 else 'balanced'}")

    # Save
    with open(RESULTS_DIR / "routing_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Global stats
    all_selections = []
    for sels in expert_selections.values():
        all_selections.extend(sels)
    global_counts = collections.Counter(all_selections)
    global_entropy = routing_entropy(global_counts, len(all_selections))

    print(f"\n  Global entropy: {global_entropy:.2f} bits")
    print(f"  Max entropy (uniform): {math.log2(8):.2f} bits (8 experts)")
    print(f"  Efficiency: {global_entropy / math.log2(8) * 100:.1f}%")
    print(f"\n  Results → {RESULTS_DIR}/routing_analysis.json")

    print("\n[Article insight]")
    print("  High entropy → experts used uniformly (load balanced).")
    print("  Low entropy → routing collapse — some experts overloaded, others idle.")
    print("  Routing collapse causes load imbalance in expert parallelism,")
    print("  reducing GPU utilization on under-used expert devices.")


if __name__ == "__main__":
    analyze_routing()
