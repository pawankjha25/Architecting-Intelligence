# Architecting LLM Inference

A multi-part series on how large language models are served in production: latency, throughput, memory, and the systems that make inference fast and reliable.

## Parts

| Part | Topic | Status |
|------|-------|--------|
| [Part1-EndToEnd-Mental-Model](./Part1-EndToEnd-Mental-Model/) | End-to-end mental model of LLM inference | — |
| [Part2-KV-Cache](./Part2-KV-Cache/) | KV cache design and memory tradeoffs | — |
| [Part3-Continuous-Batching](./Part3-Continuous-Batching/) | Continuous batching and scheduling | — |
| [Part4-vLLM-Internals](./Part4-vLLM-Internals/) | vLLM internals (PagedAttention, etc.) | — |
| [Part5-Speculative-Decoding](./Part5-Speculative-Decoding/) | Speculative decoding | — |
| [Part6-Inference-Parallelism](./Part6-Inference-Parallelism/) | Tensor/pipeline/expert parallelism for inference | — |
| [Part7-Quantization](./Part7-Quantization/) | Quantization for inference | — |
| [Part8-Production-Serving](./Part8-Production-Serving/) | Production serving patterns | — |
| [Part9-Future-of-Inference](./Part9-Future-of-Inference/) | Emerging directions in inference | — |

## Per-part assets

Each part may include:

- `article-assets/` — Figures and media for written articles
- `diagrams/` — Architecture and flow diagrams
- `notebooks/` — Exploratory Jupyter notebooks
- `code/` — Runnable examples and benchmarks
- `youtube-slides/` — Slide decks and visuals for video content
- `references.md` — Papers, blogs, and links for that part
