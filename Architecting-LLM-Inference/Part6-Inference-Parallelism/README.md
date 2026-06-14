# Part 6: Parallelism for Large-Scale LLM Inference

Companion code for the *Architecting LLM Inference* Substack series — Part 6.

Covers the full stack of techniques that make production LLM serving fast and memory-efficient: from KV cache memory management on a single GPU to tensor, pipeline, and expert parallelism across multiple GPUs.

---

## Structure

```
Part6-Inference-Parallelism/
├── benchmarks/                    # Shared utilities
│   ├── workload_generator.py      # Synthetic request generation
│   └── metrics_collector.py       # Latency/throughput helpers
│
├── phase1_batching/               # MacBook — Batching & Scheduling
│   ├── e01_static_batching.py
│   ├── e02_static_batching.py
│   ├── e03_dynamic_batching.py
│   ├── e04_continuous_batching.py
│   └── e05_priority_scheduling.py
│
├── phase2_memory/                 # MacBook — Memory Management
│   ├── e06_block_allocator.py     # PagedAttention memory model
│   ├── e07_prefix_cache.py        # RadixAttention prefix cache
│   └── e08_kv_offload_sim.py      # GPU→CPU→NVMe KV tiering
│
├── phase3_single_gpu/             # RunPod RTX 3090 — Real Serving
│   ├── e09_baseline.sh            # vLLM baseline — throughput vs latency
│   ├── e09_benchmark_client.py    # Async benchmark client (used by all experiments)
│   ├── e10_paged_attention.py     # Block size sweep + concurrency scaling
│   ├── e11_continuous_batching.sh # Rate sweep showing continuous batching gains
│   ├── e12_prefix_caching.sh      # TTFT reduction from prefix cache
│   ├── e12_prefix_caching_client.py
│   ├── e13_chunked_prefill.sh     # Chunked prefill — TTFT vs TPOT tradeoff
│   ├── e14_kv_offloading.sh       # CPU offload — capacity vs throughput
│   ├── e15_speculative_decoding.sh # Draft model speedup vs K tokens
│   ├── e16_multi_lora.sh          # Multi-LoRA serving overhead
│   ├── e26_gqa_vs_mha.py          # GQA vs MHA memory comparison
│   ├── e27_cuda_graphs.sh         # CUDA Graphs on/off — TPOT impact
│   └── e27_cuda_graphs_compare.py
│
├── phase4_parallelism/            # RunPod Multi-GPU — Parallelism
└── phase5_moe_disagg/             # RunPod Multi-GPU — MoE & Disaggregation
```

---

## Experiments

### Phase 1 — Batching & Scheduling (MacBook, pure Python)

| ID  | File | What it measures |
|-----|------|-----------------|
| E01 | e01_static_batching.py | Static batching baseline — padding waste at different batch sizes |
| E02 | e02_static_batching.py | Static batching throughput vs latency tradeoff |
| E03 | e03_dynamic_batching.py | Dynamic batching — time-window collection vs count-based |
| E04 | e04_continuous_batching.py | Continuous batching — iteration-level scheduling simulation |
| E05 | e05_priority_scheduling.py | Priority queue scheduling — SLO-aware request ordering |

### Phase 2 — Memory Management (MacBook, pure Python)

| ID  | File | What it measures |
|-----|------|-----------------|
| E06 | e06_block_allocator.py | PagedAttention memory model — block table, free list, CoW, fragmentation |
| E07 | e07_prefix_cache.py | Prefix cache hit rates across workload types (system prompt, chat, RAG, random) |
| E08 | e08_kv_offload_sim.py | KV cache offloading simulation — GPU→CPU→NVMe latency tradeoffs |

### Phase 3 — Single GPU Serving (RunPod RTX 3090)

| ID  | File | What it measures |
|-----|------|-----------------|
| E09 | e09_baseline.sh | vLLM baseline — TTFT, TPOT, throughput at 1/5/10/20/50 req/s |
| E10 | e10_paged_attention.py | Block size sweep (16/32/64) + concurrency scaling (1→128) |
| E11 | e11_continuous_batching.sh | Continuous batching rate sweep — throughput vs latency curve |
| E12 | e12_prefix_caching.sh | Prefix caching — TTFT reduction % with shared system prompt |
| E13 | e13_chunked_prefill.sh | Chunked prefill — chunk sizes 256/512/1024/2048 |
| E14 | e14_kv_offloading.sh | KV CPU offload — throughput penalty at 0/4/8/16 GB offloaded |
| E15 | e15_speculative_decoding.sh | Speculative decoding — TPOT speedup at K=3/5/7 draft tokens |
| E16 | e16_multi_lora.sh | Multi-LoRA serving — memory overhead and latency with 2 adapters |
| E26 | e26_gqa_vs_mha.py | GQA vs MHA — KV cache memory savings |
| E27 | e27_cuda_graphs.sh | CUDA Graphs on/off — TPOT speedup at low vs high batch sizes |

---

## Key Results (so far)

### E09 — Baseline (RTX 3090, Qwen2.5-7B-Instruct)

| Rate (req/s) | Tok/s | TTFT_mean | TTFT_p99 | TPOT_mean |
|-------------|-------|-----------|----------|-----------|
| 1           | 2251  | 88ms      | 930ms    | 20ms      |
| 5           | 2185  | 61ms      | 73ms     | 21ms      |
| 10          | 2092  | 64ms      | 82ms     | 21ms      |
| 20          | 1838  | 79ms      | 122ms    | 26ms      |
| 50          | 1681  | 92ms      | 154ms    | 32ms      |

Throughput peaks around 5 req/s then degrades as the GPU saturates. TTFT_p99 climbs from 73ms → 154ms as requests queue up.

### E10 — Concurrency Scaling (block_size=16)

| Concurrency | Tok/s | Req/s | P99 Latency |
|-------------|-------|-------|-------------|
| 1           | 50    | 0.4   | 2559ms      |
| 8           | 390   | 3.0   | 2643ms      |
| 32          | 1290  | 10.1  | 2931ms      |
| 64          | 2038  | 15.9  | 3349ms      |
| 128         | 2586  | 20.2  | 5331ms      |

PagedAttention enables ~40x throughput gain (50→2038 tok/s) from concurrency=1 to 64 without OOM. At 128 the GPU saturates and P99 jumps to 5.3s.

---

## Setup

### MacBook (Phase 1 & 2)
```bash
cd Part6-Inference-Parallelism
pip install -r requirements-mac.txt

# Run any Phase 1/2 experiment
python phase1_batching/e01_static_batching.py
python phase2_memory/e06_block_allocator.py
python phase2_memory/e07_prefix_cache.py
python phase2_memory/e08_kv_offload_sim.py
```

### RunPod (Phase 3+)
Recommended pod: **RTX 3090, 24GB VRAM**
Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

```bash
# On RunPod pod
export HF_HOME=/workspace/hf_cache
git clone https://github.com/pawankjha25/Architecting-Intelligence.git
cd Architecting-Intelligence/Architecting-LLM-Inference/Part6-Inference-Parallelism
pip install -r requirements-runpod.txt
pip install pynvml --break-system-packages

# Run experiments
bash phase3_single_gpu/e09_baseline.sh
python3 phase3_single_gpu/e10_paged_attention.py
bash phase3_single_gpu/e11_continuous_batching.sh
bash phase3_single_gpu/e12_prefix_caching.sh
bash phase3_single_gpu/e13_chunked_prefill.sh
bash phase3_single_gpu/e14_kv_offloading.sh
bash phase3_single_gpu/e15_speculative_decoding.sh
bash phase3_single_gpu/e16_multi_lora.sh
bash phase3_single_gpu/e27_cuda_graphs.sh
```

### Important: model weights location
Always set `HF_HOME=/workspace/hf_cache` before starting vLLM. The `/workspace` mount is network storage (~850TB) — the root disk is only 20GB and will fill up with model weights otherwise.

---

## Concepts Covered

**Batching**
- Static batching — fixed batch, padded to max length, simple but wasteful
- Dynamic batching — time-window collection, reduces padding but still not optimal
- Continuous batching (iteration-level scheduling) — insert new requests mid-generation; no padding waste; the foundation of modern LLM serving

**Memory Management**
- PagedAttention — non-contiguous KV cache blocks (like virtual memory); eliminates pre-allocation waste; enables high concurrency
- Prefix caching — reuse KV blocks for shared prefixes (system prompts, RAG documents); reduces TTFT for repeated prefixes
- KV cache offloading — spill overflow KV blocks to CPU RAM (PCIe, 16 GB/s) or NVMe (7 GB/s); trades throughput for capacity
- Copy-on-Write (CoW) — shared prefix blocks only copied when a sequence diverges; enables safe prefix sharing

**Single GPU Serving**
- Chunked prefill — split large prefill into chunks, interleave with decode steps; stabilizes decode latency at cost of slight TTFT increase
- Speculative decoding — small draft model proposes K tokens; target model verifies in one forward pass; speeds up decode at low concurrency
- Multi-LoRA — serve multiple LoRA adapters from one base model; memory overhead per adapter is small vs loading separate models
- CUDA Graphs — capture decode forward pass as graph; replay with single API call; eliminates kernel launch overhead; biggest gain at low batch sizes
- GQA (Grouped Query Attention) — share K/V heads across Q heads; reduces KV cache size; no accuracy loss for inference

**Interconnects**
- PCIe — CPU↔GPU bus (16 GB/s); bottleneck for KV offloading
- NVLink — direct GPU↔GPU (600–900 GB/s); enables tensor parallelism across GPUs
- InfiniBand — inter-node GPU networking (~50 GB/s); RDMA bypasses CPU; enables pipeline parallelism across servers

---

## Model

All Phase 3 experiments use **Qwen/Qwen2.5-7B-Instruct** (Apache 2.0, no HuggingFace gating required).

---

## Results

All experiment results are saved as JSON under `results/`:
```
results/
├── e09_baseline/E09_baseline.json
├── e10_paged_attention/results.json
├── e11_continuous_batching/E11_continuous_batching.json
├── e12_prefix_caching/E12_prefix_caching.json
├── e13_chunked_prefill/E13_chunked_prefill.json
├── e14_kv_offloading/E14_kv_offloading.json
├── e15_speculative_decoding/E15_speculative_decoding.json
├── e16_multi_lora/E16_multi_lora.json
└── e27_cuda_graphs/E27_cuda_graphs.json
```
