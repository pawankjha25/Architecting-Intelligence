# Part 2: KV Cache -- Companion Code

Companion code for the *Architecting LLM Inference* Substack series -- Part 2
(Memory Architecture, Paging, and Cache-Footprint Optimization / Reuse,
Memory Hierarchy, Runtime Scheduling, and Distributed KV State).

Maps directly onto the 12 experiments + 1 simulation in
`../kv_cache_article_outline.md`.

---

## Structure

```
code/
├── benchmarks/                     # Shared utilities
│   ├── metrics_collector.py        # Latency/throughput/memory helpers
│   ├── workload_generator.py       # Synthetic request + RAG/chat workloads
│   └── memory_model.py             # M_kv formula + real model configs (MHA/GQA/MQA)
│
├── phase1_foundations/             # MacBook -- math + optional real-model measurement
│   ├── e01_cached_vs_uncached.py
│   ├── e02_memory_formula.py
│   └── e03_mha_gqa_mqa.py
│
├── phase2_allocation/              # MacBook -- pure Python, no GPU
│   ├── e04_allocator_simulation.py
│   ├── e05_block_size_sweep.py
│   ├── e05_client.py               # async client, used by e05 Phase B
│   └── e05_block_size_sweep_runpod.sh
│
├── phase3_footprint_opt/           # MacBook -- synthetic proxy simulations
│   ├── retention_sim.py            # shared retention-policy harness (see honesty note inside)
│   ├── e06_sliding_window.py
│   ├── e07_attention_sinks.py
│   ├── e08_eviction_policies.py
│   └── e09_kv_quantization.py
│
├── phase4_reuse_serving/           # RunPod -- real vLLM (+ one CPU-only prototype)
│   ├── e10_prefix_caching.sh
│   ├── e10_client.py
│   ├── e11_offload_prototype.py    # runs anywhere; needs CUDA to produce real numbers
│   ├── e11_cpu_offload_runpod.sh
│   ├── e12_compat_matrix.py        # RUN THIS FIRST, before the combined experiment
│   └── e12_combined_config_runpod.sh
│
└── phase5_simulation/              # MacBook -- pure Python
    └── sim01_cache_aware_routing.py
```

---

## Experiments

### Phase 1 -- Foundations (MacBook, pure math + optional real model)

| ID  | File | What it measures |
|-----|------|-------------------|
| E01 | e01_cached_vs_uncached.py | Two separate curves: compute/latency blowup with no cache vs. memory blowup with a naive unbounded persistent cache under concurrency |
| E02 | e02_memory_formula.py | Validates `M_kv = 2 x L x B x S x H_kv x D_h x P` linearity across seq_len, batch size, precision |
| E03 | e03_mha_gqa_mqa.py | Bytes/token, max concurrency, and a controlled group-size ablation across MHA/GQA/MQA |

### Phase 2 -- Allocation (MacBook, pure Python)

| ID  | File | What it measures |
|-----|------|-------------------|
| E04 | e04_allocator_simulation.py | Contiguous (first-fit, worst-case reservation) vs. paged allocator, same memory budget, same request stream -- rejections and fragmentation |
| E05 | e05_block_size_sweep.py | Block-size sweep (8/16/32/64/128): occupancy, internal fragmentation, block-table size tradeoff |
| E05 (Phase B) | e05_block_size_sweep_runpod.sh | Same sweep against a live vLLM server -- real throughput/TPOT/P99 |

### Phase 3 -- Cache-Footprint Optimization (MacBook, synthetic proxy simulations)

| ID  | File | What it measures |
|-----|------|-------------------|
| E06 | e06_sliding_window.py | Full cache vs. sliding window: memory reduction vs. needle-survival proxy |
| E07 | e07_attention_sinks.py | Sliding window vs. attention sinks -- correctly separates the stability claim from the (near-nonexistent) middle-content recall claim |
| E08 | e08_eviction_policies.py | Recency-only policies vs. H2O-style heavy-hitter retention -- shows where heavy-hitter actually wins (important-but-not-recent tokens) |
| E09 | e09_kv_quantization.py | BF16 down to INT2: reconstruction MSE/SNR, and why outlier channels motivate per-channel/per-token schemes (KIVI, KVQuant) |

**Honesty note:** E06-E09 use synthetic proxies (a "needle survival rate," a synthetic "collapse score," synthetic KV value distributions) because this code was authored without GPU/model/internet access. Every file's docstring says exactly what's simulated vs. what a real-model Phase B would need to measure instead. See `retention_sim.py`'s module docstring for the full explanation.

**Phase B for E06-E09 (real model, GPU required):**

| File | Replaces the proxy with |
|------|--------------------------|
| `gpu_cache_utils.py` | Shared model loading, `DynamicCache` manipulation primitives, and a perplexity-under-eviction-policy harness (teacher-forced, real next-token loss) |
| `e06_sliding_window_gpu.py` | Real perplexity vs. window size, in place of needle-survival rate |
| `e07_attention_sinks_gpu.py` | Real perplexity with/without retained sink tokens, in place of the synthetic collapse score |
| `e08_eviction_policies_gpu.py` | Real attention scores (`output_attentions=True`), accumulated across steps, in place of the synthetic importance signal |
| `e09_kv_quantization_gpu.py` | Real captured K/V tensors, fake-quantized, in place of synthetic Gaussian-plus-outlier values |

**IMPORTANT -- these four were NOT executed before being committed.** This
code was authored in an environment with no GPU, no `torch`/`transformers`
installed, and no internet access to reach Hugging Face -- there was no way
to test it end-to-end here. The `DynamicCache` manipulation logic in
`gpu_cache_utils.py` was written carefully against the API version pinned
in `requirements-runpod.txt` (`transformers>=4.40.0,<4.46.0`), and each
script's import/argument-parsing path was smoke-tested (it correctly
detects "no GPU" and exits cleanly), but the actual model-forward-pass
logic is unverified. Run these on your pod, and if anything throws --
especially an `AttributeError` around `cache.key_cache` -- that's the first
place to look; it usually means your installed transformers version
changed the `Cache` internals.

```bash
python3 phase3_footprint_opt/e06_sliding_window_gpu.py
python3 phase3_footprint_opt/e07_attention_sinks_gpu.py
python3 phase3_footprint_opt/e08_eviction_policies_gpu.py   # slower -- output_attentions=True adds overhead
python3 phase3_footprint_opt/e09_kv_quantization_gpu.py
```

Default model is `Qwen/Qwen2.5-0.5B-Instruct` (small, fast, GQA, no gating) --
override with `--model` on any of the four scripts.

### Phase 4 -- Reuse & Serving (RunPod RTX 3090, real vLLM; one CPU-only prototype)

| ID  | File | What it measures |
|-----|------|-------------------|
| E10 | e10_prefix_caching.sh / e10_client.py | Prefix caching cold vs. warm: TTFT with a shared prefix, caching disabled vs. enabled |
| E11 | e11_offload_prototype.py | Manual GPU<->CPU round-trip cost (labeled explicitly as a controlled prototype, not a production offload engine) |
| E11 (Phase B) | e11_cpu_offload_runpod.sh | Real vLLM `--swap-space` sweep -- throughput penalty vs. capacity gained |
| E12 | e12_compat_matrix.py | **Run first.** Checks which flags your vLLM version actually supports before stacking them |
| E12 | e12_combined_config_runpod.sh | Baseline vs. paged + prefix caching + fp8 KV cache stacked |

### Phase 5 -- Simulation (MacBook, pure Python)

| ID  | File | What it measures |
|-----|------|-------------------|
| Sim01 | sim01_cache_aware_routing.py | Round robin / least-queue / power-of-two / session affinity / prefix affinity / cache-value-aware routing -- latency, cache hit rate, load balance |

---

## Setup

### MacBook (Phase 1, 2, 3, 5 -- and E11's prototype)
```bash
cd Part2-KV-Cache/code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt

# Pure-Python experiments (no dependencies beyond stdlib):
python3 phase1_foundations/e01_cached_vs_uncached.py
python3 phase1_foundations/e02_memory_formula.py
python3 phase1_foundations/e03_mha_gqa_mqa.py
python3 phase2_allocation/e04_allocator_simulation.py
python3 phase2_allocation/e05_block_size_sweep.py
python3 phase3_footprint_opt/e06_sliding_window.py
python3 phase3_footprint_opt/e07_attention_sinks.py
python3 phase3_footprint_opt/e08_eviction_policies.py
python3 phase3_footprint_opt/e09_kv_quantization.py
python3 phase4_reuse_serving/e11_offload_prototype.py   # will report and skip gracefully without a GPU
python3 phase5_simulation/sim01_cache_aware_routing.py
```

### RunPod (Phase 4 real-serving experiments)
Recommended pod: **RTX 3090, 24GB VRAM**
Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

```bash
export HF_HOME=/workspace/hf_cache
git clone https://github.com/pawankjha25/Architecting-Intelligence.git
cd Architecting-Intelligence/Architecting-LLM-Inference/Part2-KV-Cache/code
pip install -r requirements-runpod.txt

bash phase2_allocation/e05_block_size_sweep_runpod.sh
bash phase4_reuse_serving/e10_prefix_caching.sh
python3 phase4_reuse_serving/e12_compat_matrix.py     # run before the combined experiment
bash phase4_reuse_serving/e11_cpu_offload_runpod.sh
bash phase4_reuse_serving/e12_combined_config_runpod.sh

# E06-E09 Phase B (real model, needs torch+transformers, see pinned versions above)
python3 phase3_footprint_opt/e06_sliding_window_gpu.py
python3 phase3_footprint_opt/e07_attention_sinks_gpu.py
python3 phase3_footprint_opt/e08_eviction_policies_gpu.py
python3 phase3_footprint_opt/e09_kv_quantization_gpu.py
```

### Important: model weights location
Always set `HF_HOME=/workspace/hf_cache` before starting vLLM -- `/workspace` is
network storage; the pod's root disk is too small for repeated model downloads.

---

## Model

Phase 4 RunPod experiments use **Qwen/Qwen2.5-7B-Instruct** (Apache 2.0, no
gating required), consistent with Part 6's choice so results are comparable
across the series.

---

## Results

All experiments save JSON to `results/` (created automatically, relative to
wherever you run the script from -- run from `code/` to keep them all in one
place):

```
results/
├── E01_cached_vs_uncached.json
├── E02_memory_formula.json
├── E03_mha_gqa_mqa.json
├── E04_allocator_simulation.json
├── E05_block_size_sweep.json
├── e05_block_size_sweep/            # Phase B, RunPod
├── E06_sliding_window.json
├── E07_attention_sinks.json
├── E08_eviction_policies.json
├── E09_kv_quantization.json
├── e10_prefix_caching/              # RunPod
├── E11_offload_prototype.json
├── e11_cpu_offload/                 # RunPod
├── E12_compat_matrix.json
├── e12_combined/                    # RunPod
└── Sim01_cache_aware_routing.json
```

---

## Status

All 12 experiments + Sim01 implemented. Phase 1, 2, 3, 5 verified to run
end-to-end in a plain Python 3.10 environment (no GPU, no torch, no
internet) as part of authoring this code. Phase 4 scripts are written
against vLLM's documented flags but not yet run against a live pod --
run `e12_compat_matrix.py` first on your actual RunPod instance before
trusting the combined-config flags blindly.
