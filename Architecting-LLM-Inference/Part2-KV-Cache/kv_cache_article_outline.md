# Architecting KV Cache for LLM Inference — Final Two-Part Outline

Depth target: intermediate → advanced. Basics get a paragraph, not a section. Published as two long-form technical articles (book-chapter depth as currently scoped — see trimming notes at the end if shorter, blog-length pieces are wanted instead).

---

# Part 1 — Memory Architecture, Paging, and Cache-Footprint Optimization

## 1. Introduction
1.1 KV cache as a compute–memory trade-off
1.2 Why KV cache becomes a production bottleneck
1.3 Questions investigated in this article
1.4 Experimental environment and methodology
1.5 Scope of Parts 1 and 2

## 2. Compressed Preliminaries (~half a page)
2.1 Autoregressive decoding and repeated computation
2.2 Why K and V are retained but Q is not
2.3 Prefill versus decode cache behavior
2.4 From tensor reuse to persistent model state

## 3. Quantitative Foundation
3.1 Deriving the KV-cache memory equation
3.2 Bytes stored per token
3.3 Memory growth across layers and active sequences
3.4 Prompt tokens versus generated tokens
3.5 Cache capacity versus concurrency
3.6 MHA, GQA, and MQA
3.7 KV-head count and cache-bandwidth implications
3.8 Weight memory versus KV-cache memory
3.9 Theoretical versus framework-observed memory

`M_kv = 2 × L × B × S × H_kv × D_h × P` (bytes)

> **Experiment 1 — Cached vs. uncached decoding.**
> Report two *separate* curves, not one — this was the key correction from the earlier draft:
> - **No cache:** compute/latency blowup. Each step recomputes attention over the full sequence (O(n²) total), so nothing persists in memory between steps — the transient activation memory is freed after each forward pass. The story here is tokens/sec collapsing and per-token latency growing, not standing memory growth.
> - **Naive persistent cache, unbounded, many concurrent long sequences:** this is where GPU memory actually grows monotonically and eventually OOMs, because nothing is evicted.
> Measure: per-token latency by decode position, end-to-end latency, tokens/sec, peak/allocated memory, incremental KV memory with caching, repeated-computation cost without caching.

> **Experiment 2 — Formula vs. measured KV memory.**
> Vary: prompt length, generated length, batch/concurrency, cache precision, allocator overhead.

> **Experiment 3 — MHA vs. GQA vs. MQA.**
> Measure: bytes per cached token, TPOT, bandwidth, maximum concurrency, quality implications.

## 4. Logical and Physical KV-Cache Architecture
4.1 Logical cache visible to the attention layer
4.2 Physical cache managed by the runtime
4.3 Tensor layouts
4.4 Layer-, head-, token-, and block-major layouts
4.5 Logical positions versus physical slots
4.6 RoPE positions versus cache addresses
4.7 KV-cache manager responsibilities
4.8 Request and sequence metadata
4.9 Block metadata
4.10 Block tables and slot mappings
4.11 Free-block pools
4.12 Reference counts
4.13 Copy-on-write
4.14 Allocation, append, release, and reclamation
4.15 Cancellation and failure cleanup

## 5. Contiguous Allocation and Fragmentation
5.1 Static maximum-length allocation
5.2 Dynamic contiguous growth
5.3 Unknown output length
5.4 Reservation waste
5.5 Internal fragmentation
5.6 External fragmentation
5.7 Reallocation and memory copies
5.8 Effects on continuous batching and concurrency

> **Experiment 4 — Contiguous vs. paged allocator simulation.**
> Pure Python, event-driven, no GPU needed — isolates allocator behavior from kernel effects. Model: variable arrival times, variable prompt/output lengths, request completion and reclamation, internal/external fragmentation, allocation failures, cache utilization.

## 6. Paged KV Cache and PagedAttention
6.1 Motivation for block-based allocation
6.2 Logical blocks
6.3 Physical blocks
6.4 Request block tables
6.5 Incremental block allocation
6.6 Partially filled tail blocks
6.7 Block reclamation
6.8 Shared prefix blocks
6.9 Reference counting
6.10 Copy-on-write during branch divergence
6.11 Page-aware attention
6.12 Metadata and block-table access overhead
6.13 Interaction with prefix caching
6.14 Interaction with quantized cache
6.15 Where PagedAttention helps — and where it does not

## 7. PagedAttention versus Virtual-Memory-Backed Allocation (full treatment)
7.1 Software-managed paging
7.2 Non-contiguous physical KV blocks
7.3 Kernel-visible block tables
7.4 Contiguous virtual address reservation
7.5 Lazy physical-page mapping
7.6 CUDA virtual-memory management
7.7 Attention-kernel compatibility
7.8 Mapping and unmapping overhead
7.9 Page size and allocation granularity
7.10 Portability and implementation complexity
7.11 PagedAttention vs. vAttention trade-off matrix

> vAttention is not just another paging variant: it keeps the KV cache contiguous in *virtual* address space and uses CUDA virtual-memory APIs to map physical pages on demand, whereas PagedAttention makes block mapping explicit to the serving runtime and the attention kernel itself.

## 8. Block Size and Physical Layout Optimization
8.1 Internal fragmentation versus metadata overhead
8.2 Block-table size
8.3 Allocation frequency
8.4 Kernel locality
8.5 Prefix-sharing granularity
8.6 Eviction granularity
8.7 Offload-transfer granularity
8.8 Quantization-group alignment
8.9 Workload-dependent block size

> **Experiment 5 — Block-size sweep.** Test 8/16/32/64/128 tokens per block. Measure occupancy, memory waste, allocation overhead, throughput, TPOT, P99 latency, prefix-cache effectiveness.
> Caution: supported block sizes are runtime/kernel-version dependent — don't assume arbitrary values work on every vLLM build.

## 9. Hybrid Cache Managers
9.1 Full-attention layers
9.2 Sliding-window layers
9.3 Local/global hybrid architectures
9.4 Cross-attention cache
9.5 State-space and recurrent layers
9.6 Per-layer cache specifications
9.7 Separate block pools versus shared pools
9.8 Prefix reuse across heterogeneous attention types
9.9 Memory accounting for mixed architectures

## 10. Cache-Footprint Optimization: What Should Be Retained?

| Method family | Retention signal | What is removed | Main failure mode |
|---|---|---|---|
| Sliding window | Recency | Old tokens | Distant context loss |
| Attention sinks | Initial anchors + recency | Middle history | Missed distant evidence |
| H2O | Cumulative attention + recency | Low-attention tokens | Historical score bias |
| SnapKV | Observation-window attention | Low-ranked prompt positions | Query/generation drift |
| Pyramid methods | Layer sensitivity | Layer-specific tokens | Calibration dependence |
| Semantic selection | Similarity/clusters | Redundant states | Selection overhead |
| Quantization | Numerical compressibility | Precision | Accumulated error |
| Low rank | Subspace structure | Full dimensionality | Reconstruction overhead |

## 11. Sliding-Window and Streaming Caches
11.1 Fixed local window · 11.2 Memory saturation at W · 11.3 Ring-buffer implementation · 11.4 Block-level eviction · 11.5 Position handling · 11.6 Layer-specific windows · 11.7 Hybrid local/global layers · 11.8 Quality loss beyond the window

> **Experiment 6 — Full cache vs. sliding window.** Measure memory, TPOT, max concurrency, perplexity, long-context retrieval, distance-sensitive accuracy.

## 12. Attention Sinks
12.1 Why pure window eviction destabilizes streaming · 12.2 Initial sink tokens · 12.3 Recent-token window · 12.4 Fixed-memory streaming · 12.5 Sink count vs. usable recent context · 12.6 Relationship to massive activations · 12.7 Limitations across models

> **Experiment 7 — Sliding window vs. attention sinks.** Test whether generation quality collapses without retained initial tokens, whether the effect varies by model, whether sinks improve perplexity but not necessarily retrieval.

## 13. Heavy Hitters and Attention-Based Eviction
13.1 Heavy-hitter observation · 13.2 Cumulative attention importance · 13.3 Recent-token protection · 13.4 H2O retention policy · 13.5 Head and layer aggregation · 13.6 Online score-update overhead · 13.7 Block-aligned implementation · 13.8 Failure modes and stale importance

> H2O's core idea: retain both recent tokens and tokens that historically receive substantial attention, rather than treating age as the sole signal.

> **Experiment 8 — Recency vs. attention-based eviction.** Compare FIFO, sliding window, sink+window, H2O-style heavy-hitter+recent.

## 14. Query-Aware and Observation-Based Selection
14.1 Why historical attention alone may be insufficient · 14.2 Observation windows · 14.3 Head-specific prompt features · 14.4 SnapKV selection procedure · 14.5 Query-aware token selection · 14.6 Block-level vs. token-level retrieval · 14.7 Selection/indexing overhead · 14.8 Generation-time drift

## 15. Layer- and Head-Adaptive Budgets
15.1 Limitations of uniform cache budgets · 15.2 Layer sensitivity · 15.3 PyramidInfer · 15.4 PyramidKV · 15.5 Retrieval-sensitive heads · 15.6 Layer-specific budgets · 15.7 Head-specific budgets · 15.8 Calibration and workload sensitivity · 15.9 Dynamic budget allocation

> Discuss PyramidInfer and PyramidKV together but not as identical — both exploit layer-dependent information concentration, but mechanisms and allocation formulations differ.

## 16. Semantic Compression and Token Consolidation
16.1 Similarity-based token selection · 16.2 KV clustering · 16.3 Token merging · 16.4 Attention-weighted centroids · 16.5 Landmark tokens · 16.6 Summary tokens · 16.7 Recurrent memory · 16.8 External retrieval as cache replacement · 16.9 Information preservation vs. compute overhead

## 17. Low-Rank and Latent Cache Representations
17.1 Low-rank structure in K and V · 17.2 Per-layer rank selection · 17.3 Shared bases · 17.4 Palu · 17.5 LoRC · 17.6 SVDq · 17.7 CSKV · 17.8 ReCalKV · 17.9 Reconstruction overhead · 17.10 Multi-head latent attention · 17.11 MLA vs. post-hoc KV compression · 17.12 Training-time vs. inference-only techniques

> Present named systems as representative approaches, not a leaderboard — some require architecture changes/training, others are post-training/plug-in.

## 18. KV-Cache Quantization
18.1 Capacity and bandwidth benefits · 18.2 FP8 · 18.3 INT8 · 18.4 INT4 · 18.5 INT2 · 18.6 KIVI · 18.7 KVQuant · 18.8 TurboQuant · 18.9 Key/value distribution differences · 18.10 Per-channel vs. per-token scaling · 18.11 Per-head and per-block scaling · 18.12 Static vs. dynamic quantization · 18.13 Outlier handling · 18.14 Quantization metadata · 18.15 Fused dequantization and attention

> **Experiment 9 — KV precision frontier.** Compare BF16/FP16, FP8 (where supported), INT8, INT4, optional INT2. Measure beyond perplexity: long-context retrieval, instruction following, exact match, output divergence, memory, TPOT, conversion overhead.

## 19. Mixed-Precision, Residual, and Joint Compression
19.1 High-precision recent cache · 19.2 Quantized historical cache · 19.3 Age-adaptive precision · 19.4 Outlier residuals · 19.5 Layer-adaptive precision · 19.6 Head-adaptive precision · 19.7 Joint eviction and quantization · 19.8 Joint low-rank and quantized cache · 19.9 Compounding gains vs. compounding errors · 19.10 Optimization compatibility matrix

## 20. Part 1 Synthesis
20.1 Allocation optimization vs. information compression · 20.2 Which methods require model modification · 20.3 Which methods are runtime-only · 20.4 Which methods affect quality · 20.5 Which methods reduce capacity, bandwidth, or both · 20.6 Findings from implemented experiments · 20.7 Open questions · 20.8 Transition to Part 2

---

# Part 2 — Reuse, Memory Hierarchy, Runtime Scheduling, and Distributed KV State

## 1. Introduction
1.1 KV cache as reusable and movable distributed state · 1.2 Recompute, retain, restore, or transfer · 1.3 Local optimization vs. cluster-wide optimization · 1.4 Experimental and simulation boundaries

## 2. Prefix Reuse
2.1 Repeated-prefix opportunities · 2.2 Exact-prefix reuse · 2.3 Block-level prefix hashing · 2.4 Longest reusable prefix · 2.5 Partial-block constraints · 2.6 Cold and warm requests · 2.7 Prefix-cache memory cost · 2.8 Reuse-value calculation

> **Experiment 10 — Prefix caching cold vs. warm.** Measure TTFT, prefill tokens avoided, throughput, cache-hit length, memory retained, lookup overhead.

## 3. Radix-Tree Caching
3.1 RadixAttention · 3.2 Token-prefix tree · 3.3 Shared path ownership · 3.4 Partial-prefix matching · 3.5 Multi-turn conversations · 3.6 Agent and tool-schema reuse · 3.7 Cache-aware scheduling · 3.8 Radix-tree eviction · 3.9 Hash blocks vs. radix trees

## 4. Session and Branch Sharing
4.1 Session cache persistence · 4.2 Session-affinity routing · 4.3 Beam-search sharing · 4.4 Parallel sampling · 4.5 Speculative decoding · 4.6 Tree-structured reasoning · 4.7 Reference counting · 4.8 Copy-on-write · 4.9 Accepted and rejected speculative branches

## 5. Cache Identity, Correctness, and Security (full section)
5.1 Token IDs and exact prefix identity · 5.2 Model version · 5.3 Tokenizer version · 5.4 Position and RoPE configuration · 5.5 Adapter/LoRA identity · 5.6 Multimodal preprocessing state · 5.7 Cache precision and layout · 5.8 Hash collisions · 5.9 Namespace separation · 5.10 Cross-tenant timing leakage · 5.11 Prefix-existence side channels · 5.12 Cache poisoning · 5.13 Safe sharing boundaries

> **Correction applied:** sampling parameters (temperature/top-p/top-k) act on logits *after* the forward pass and do not alter already-computed prompt K/V. They matter for branch/session reuse semantics, but should not be presented as part of prefix-KV identity. The real identity components are model weights, adapters, tokenization, positions, multimodal inputs, and attention configuration.

## 6. Replacement and Retention Policies
6.1 FIFO · 6.2 LRU · 6.3 LFU · 6.4 Size-aware retention · 6.5 Prefill-cost-aware retention · 6.6 Hit-probability-aware retention · 6.7 Tenant quotas · 6.8 Priority-aware retention · 6.9 Cost-benefit scoring · 6.10 Popular-prefix replication decisions

## 7. KV Cache Across the Memory Hierarchy
7.1 GPU HBM · 7.2 Peer GPU memory · 7.3 CPU DRAM · 7.4 Pinned host memory · 7.5 Local NVMe · 7.6 Remote DRAM · 7.7 Disaggregated cache services · 7.8 CXL-attached memory · 7.9 Hot and cold cache blocks · 7.10 Promotion and demotion policies

## 8. CPU Offload and Restore
8.1 Block-wise offload · 8.2 Layer-wise offload · 8.3 Asynchronous transfers · 8.4 Prefetch · 8.5 Double buffering · 8.6 PCIe bottlenecks · 8.7 Cache thrashing · 8.8 Quantization before transfer · 8.9 Restore vs. recompute (`choose restore when T_restore < T_recompute`)

> **Experiment 11 — GPU-only vs. CPU-offloaded cache.** A manual `.to("cpu")` prototype is useful pedagogically but won't represent a production offload engine — it likely introduces synchronization stalls and unoptimized transfers a real async/double-buffered engine avoids. Label it explicitly as a controlled prototype, and compare against a runtime-native offload path where available.

## 9. NVMe, Remote, and Hierarchical Caching
9.1 NVMe-resident cold state · 9.2 Batched I/O · 9.3 Compression before storage · 9.4 Remote cache transport · 9.5 Hierarchical lookup · 9.6 Prefetch prediction · 9.7 Read amplification · 9.8 Tail-latency risks · 9.9 Failure and stale-entry handling

## 10. Cache Access and Kernel Optimization
10.1 FlashAttention vs. KV-cache optimization · 10.2 Flash decoding · 10.3 PagedAttention kernels · 10.4 Split-K decode · 10.5 Sparse/block-sparse cache access · 10.6 Query-selected block access · 10.7 Fused cache append · 10.8 Fused quantization · 10.9 Fused dequantization and attention · 10.10 Cache-layout tuning

> Clarification to keep prominent: FlashAttention reduces attention *intermediate* memory and I/O (avoids materializing the full score matrix); it does not by itself reduce the persistent KV-cache footprint. Different memory pool, different problem.

## 11. Runtime Scheduling Under KV Constraints
11.1 Continuous batching · 11.2 Cache-aware batch formation · 11.3 Chunked prefill · 11.4 Incremental cache construction · 11.5 Admission control · 11.6 Output-length uncertainty · 11.7 Cache reservation · 11.8 Elastic reservation · 11.9 Preemption · 11.10 Swap vs. recompute · 11.11 Fairness and starvation · 11.12 Prefix-aware scheduling

## 12. Tensor and Pipeline Parallel KV Ownership
12.1 KV-head sharding · 12.2 GQA/MQA complications · 12.3 KV replication · 12.4 Pipeline-stage cache ownership · 12.5 Per-stage memory imbalance · 12.6 Cache lifecycle coordination · 12.7 Quantized distributed KV

## 13. Context and Sequence Parallelism
13.1 Sequence-dimension partitioning · 13.2 Context parallelism · 13.3 Distributed attention statistics · 13.4 Global softmax · 13.5 Communication overhead · 13.6 Long-context capacity scaling · 13.7 Topology-aware placement

## 14. Prefill–Decode Disaggregation
14.1 Why prefill and decode have different resource profiles · 14.2 Prefill pools · 14.3 Decode pools · 14.4 KV-cache handoff · 14.5 Decode-worker selection · 14.6 Queueing and backpressure · 14.7 When disaggregation helps · 14.8 When transfer overhead dominates

## 15. KV Transfer Engines (one of Part 2's deepest sections)
15.1 KV serialization and memory registration · 15.2 Layer-streamed transfer · 15.3 Block-streamed transfer · 15.4 Direct GPU-to-GPU transfer · 15.5 RDMA · 15.6 NVLink/NVSwitch · 15.7 InfiniBand and RoCE · 15.8 Transfer/computation overlap · 15.9 Quantized transfer · 15.10 Mooncake-style transfer engines · 15.11 Cache location metadata · 15.12 Transfer failure and retries

## 16. Cache-Aware Routing
16.1 Why least-loaded routing is insufficient · 16.2 Session affinity · 16.3 Prefix affinity · 16.4 Cached-prefix length · 16.5 Worker memory pressure · 16.6 Transfer and recomputation cost · 16.7 Locality vs. queue balance · 16.8 Routing objective formulation

> **Simulation 1 — Cache-aware routing.** Compare round robin, least queue, power-of-two choices, session affinity, prefix affinity, cache-value-aware routing. (Simulated, not hardware-measured.)

## 17. Migration, Replication, and Distributed Cache Services
17.1 Live KV migration · 17.2 Popular-prefix replication · 17.3 Replica memory cost · 17.4 Worker draining · 17.5 Cluster rebalancing · 17.6 External cache services · 17.7 Cache directories · 17.8 Consistency and ownership · 17.9 Recovery after worker failure · 17.10 Recompute from token history

## 18. Multi-Region and Failure Recovery
18.1 Session home region · 18.2 Cross-region cache transfer · 18.3 Recompute in destination region · 18.4 Compressed session checkpoints · 18.5 Regional failover · 18.6 Privacy and residency constraints · 18.7 Cost and latency trade-offs

## 19. Combined Optimization Experiment
> **Experiment 12 — Combined configuration.**
> Baseline: BF16, contiguous/full cache, no prefix reuse, GPU-only.
> Optimized: paged allocation + prefix caching + quantized historical cache + bounded/sliding cache (where architecture permits) + cache-aware admission.
>
> Do not combine sliding-window eviction with every model by default — safe only when the model natively uses sliding/local attention, or when deliberately treated as lossy compression with measured degradation. Not all vLLM flags/versions compose cleanly. **Start with a feature/version compatibility matrix before running the combined experiment**, rather than assuming everything can be enabled at once.

## 20. Experimental Results and Recommendations
20.1 Memory-capacity frontier · 20.2 Latency-throughput frontier · 20.3 Quality-memory frontier · 20.4 Bandwidth-memory frontier · 20.5 Prefix-reuse break-even point · 20.6 Offload break-even point · 20.7 Transfer break-even point · 20.8 Results by workload · 20.9 What was hardware-measured · 20.10 What was simulated · 20.11 What remained discussion-only

## 21. Workload-Specific Design Recommendations
21.1 Interactive chat · 21.2 Multi-turn assistants · 21.3 RAG with repeated documents · 21.4 Code generation · 21.5 Long-document QA · 21.6 Agentic workflows · 21.7 Parallel sampling · 21.8 Batch inference · 21.9 Multi-tenant serving · 21.10 Long-running sessions

*(Trim candidate: this can become one comparison table — workload × recommended technique stack — instead of ten subsections, if targeting shorter/blog length.)*

## 22. Conclusion and Research Directions
22.1 KV state as an information hierarchy · 22.2 Joint retention, precision, and placement · 22.3 Learned cache controllers · 22.4 Cache-aware model architectures · 22.5 Distributed KV as a first-class serving primitive · 22.6 Standardized KV-cache benchmarks

---

## Corrections applied in this final pass
- Experiment 1 reframed: no-cache = compute/latency blowup (transient memory only); naive unbounded persistent cache = the actual memory-growth/OOM story.
- Sampling parameters removed from strict prefix-KV correctness key (they act post-logits, don't affect prompt K/V).
- vAttention treated as a full architectural alternative to software paging, not a minor variant (Part 1 §7).
- H2O, SnapKV, PyramidKV/PyramidInfer, quantization methods, low-rank methods, and MLA presented as representative families, not leaderboards.
- Explicit note that not all optimizations are runtime flags or mutually compatible — compatibility matrix required before Experiment 12.
- CPU offload, cache-aware routing, and KV transfer experiments labeled honestly as prototypes/simulations, not production measurements.

## If trimming toward blog-post length instead of book-chapter depth
Candidates to compress first: Part 1 §16 (semantic compression — currently a grab-bag, could be 3–4 sentences), Part 2 §18 (multi-region failure recovery — discussion-only, can shrink to a page), Part 2 §21 (collapse to one table).
