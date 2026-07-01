# Part 2: KV Cache

Understand key-value caching during autoregressive decoding—memory layout, reuse, and impact on latency and throughput.

## Contents

| Folder / file | Purpose |
|---------------|---------|
| [notebooks/](./notebooks/) | Exploratory Jupyter notebooks |
| [code/](./code/) | Runnable examples and benchmarks |
| [diagrams/](./diagrams/) | Memory and attention diagrams |
| [youtube-slides/](./youtube-slides/) | Slide decks and visuals for video content |
| [references.md](./references.md) | Papers, blogs, and external links |

## Status

Article outline: [kv_cache_article_outline.md](./kv_cache_article_outline.md) --
two-part outline (Memory Architecture/Paging/Footprint Optimization, and
Reuse/Memory Hierarchy/Runtime Scheduling/Distributed KV State).

Code: [code/README.md](./code/README.md) -- all 12 experiments + 1 routing
simulation implemented (E01-E12, Sim01). Phase 1/2/3/5 (foundations,
allocation, footprint optimization, routing simulation) are pure-Python and
verified to run without a GPU. Phase 4 (prefix caching, CPU offload, combined
config) targets a RunPod GPU instance running vLLM -- see `code/README.md`
for setup.

<!-- TODO: Add notebook index and diagrams -->
