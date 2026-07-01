"""
E12 (step 1 of 2) -- Feature/Version Compatibility Matrix
=============================================================
Run this BEFORE e12_combined_config_runpod.sh. Per the article outline's
own correction: not all optimizations are runtime flags, and not all flags
compose cleanly on every vLLM version/backend. This script queries your
installed vLLM for the flags the combined experiment wants to stack, and
tells you which combination is actually safe to run -- rather than assuming
everything can be enabled at once.

Run: python3 phase4_reuse_serving/e12_compat_matrix.py
"""

import subprocess
import sys
import json
from pathlib import Path


REQUIRED_FLAGS = {
    "--enable-prefix-caching": "prefix caching (reuse KV across requests with shared prefixes)",
    "--block-size": "paged allocation block size",
    "--kv-cache-dtype": "KV cache quantization (fp8/int8, version-dependent)",
    "--swap-space": "CPU offload capacity",
}

# Sliding-window eviction is a MODEL property, not a universal runtime flag --
# it is only safe to apply to models that natively use local/sliding-window
# attention (e.g. Mistral). Applying it to a model that doesn't is lossy
# compression that must be measured, not assumed safe. See article Part 1 SS11.
SLIDING_WINDOW_NATIVE_MODELS = {"mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-v0.1"}


def check_vllm_flags() -> dict:
    try:
        help_text = subprocess.run(
            ["vllm", "serve", "--help"], capture_output=True, text=True, timeout=30
        ).stdout
    except FileNotFoundError:
        return {"vllm_installed": False}
    except Exception as e:
        return {"vllm_installed": False, "error": str(e)}

    support = {"vllm_installed": True}
    for flag, desc in REQUIRED_FLAGS.items():
        support[flag] = {"supported": flag in help_text, "description": desc}
    return support


def check_model_compatibility(model_name: str) -> dict:
    return {
        "model": model_name,
        "sliding_window_native": model_name in SLIDING_WINDOW_NATIVE_MODELS,
        "recommendation": (
            "Safe to enable sliding-window eviction as a native feature."
            if model_name in SLIDING_WINDOW_NATIVE_MODELS else
            "This model does NOT natively use sliding-window attention. "
            "Applying window truncation here is lossy compression -- measure "
            "degradation explicitly (see E06/E07) rather than assuming it's safe."
        ),
    }


def main():
    print("E12 (step 1) -- Feature/Version Compatibility Matrix\n")

    flag_support = check_vllm_flags()
    if not flag_support.get("vllm_installed"):
        print("  vLLM not found on this machine. Run this on your RunPod GPU instance")
        print("  after `pip install -r requirements-runpod.txt`.")
        return

    print("-- vLLM flag support (this installed version) ----------------------")
    all_ok = True
    for flag, info in flag_support.items():
        if flag == "vllm_installed":
            continue
        status = "OK" if info["supported"] else "MISSING"
        if not info["supported"]:
            all_ok = False
        print(f"  [{status:>7}] {flag:24s} -- {info['description']}")

    model = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\n-- Model compatibility: {model} ----------------------")
    model_check = check_model_compatibility(model)
    print(f"  {model_check['recommendation']}")

    print("\n[Verdict]")
    if all_ok:
        print("  All required flags are supported. Safe to proceed to")
        print("  e12_combined_config_runpod.sh with block_size + prefix caching +")
        print("  kv-cache-dtype quantization stacked. Leave sliding-window eviction")
        print("  OUT of the combined run unless your model natively supports it,")
        print("  or you're deliberately measuring it as lossy compression.")
    else:
        print("  Some flags are missing on this vLLM version -- upgrade vLLM or drop")
        print("  the unsupported optimization from the combined run rather than")
        print("  assuming it silently no-ops.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E12_compat_matrix.json", "w") as f:
        json.dump({"flags": flag_support, "model": model_check}, f, indent=2)
    print("\n  Results saved -> results/E12_compat_matrix.json")


if __name__ == "__main__":
    main()
