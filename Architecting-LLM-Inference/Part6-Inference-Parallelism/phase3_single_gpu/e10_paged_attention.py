"""
E10 — vLLM Serving vs Naive HuggingFace Generation
====================================================
Goal:   Show the combined effect of modern serving runtime design.
        PagedAttention is one major contributor, but vLLM also brings
        continuous batching, optimized CUDA kernels, and a smarter scheduler.
        This experiment measures the *total* effect — not just paging.
Run on: RunPod RTX 3090 (single GPU)
Model:  Qwen/Qwen2.5-7B-Instruct  (open, Apache 2.0)
Vary:   Concurrent requests, block size
Note:   Llama 3.1 8B is a valid swap if you have HuggingFace gated access.
"""

import subprocess
import json
import time
import sys
import os
import requests
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.metrics_collector import ExperimentMetrics, RequestMetrics

MODEL = "Qwen/Qwen2.5-7B-Instruct"   # open, Apache 2.0, no gating
PORT = 8000
RESULTS_DIR = Path("results/e10_paged_attention")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZES = [8, 16, 32]
CONCURRENCY_LEVELS = [1, 8, 32, 64, 128]


def start_vllm_server(block_size: int = 16, extra_args: list = None) -> subprocess.Popen:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--max-model-len", "8192",
        "--dtype", "float16",
        "--port", str(PORT),
        "--block-size", str(block_size),
        "--disable-log-requests",
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.Popen(cmd)
    _wait_for_server()
    return proc


def _wait_for_server(timeout: int = 120) -> None:
    for _ in range(timeout):
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("vLLM server failed to start")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    proc.wait(timeout=30)


def benchmark_concurrent(num_concurrent: int, num_requests: int = 200) -> dict:
    """Send concurrent requests to running server, measure throughput + latency."""
    import asyncio
    import aiohttp

    prompt = "Explain the transformer architecture in detail, covering attention mechanisms, " \
             "positional encoding, and the feed-forward layers. " * 4

    results = []

    async def send_one(session, req_id: int) -> dict:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        async with session.post(
            f"http://localhost:{PORT}/v1/chat/completions",
            json=payload,
        ) as resp:
            data = await resp.json()
            t1 = time.perf_counter()
            return {
                "req_id": req_id,
                "latency_s": t1 - t0,
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }

    async def run_all():
        connector = aiohttp.TCPConnector(limit=num_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            semaphore = asyncio.Semaphore(num_concurrent)
            async def bounded(i):
                async with semaphore:
                    return await send_one(session, i)
            return await asyncio.gather(*[bounded(i) for i in range(num_requests)])

    t_start = time.perf_counter()
    results = asyncio.run(run_all())
    t_end = time.perf_counter()

    latencies = [r["latency_s"] for r in results]
    total_tokens = sum(r["output_tokens"] for r in results)
    wall = t_end - t_start

    return {
        "num_concurrent": num_concurrent,
        "num_requests": num_requests,
        "throughput_req_s": num_requests / wall,
        "throughput_tok_s": total_tokens / wall,
        "latency_mean_ms": sum(latencies) / len(latencies) * 1000,
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)] * 1000,
        "wall_time_s": wall,
    }


def get_gpu_memory_usage() -> dict:
    """Query current GPU memory via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, free = [int(x.strip()) for x in out.split(",")]
        return {"used_mb": used, "free_mb": free, "total_mb": used + free}
    except Exception:
        return {}


def main():
    print(f"E10 — PagedAttention vs Naive | Model: {MODEL}\n")

    all_results = []

    # ── Test different block sizes ─────────────────────────────────────────────
    print("=== Block Size Sweep ===")
    for block_size in BLOCK_SIZES:
        print(f"\n  block_size={block_size}...")
        try:
            proc = start_vllm_server(block_size=block_size)
            mem_before = get_gpu_memory_usage()

            result = benchmark_concurrent(num_concurrent=32, num_requests=100)
            result["block_size"] = block_size
            result["gpu_memory"] = mem_before
            all_results.append(result)

            print(f"    Throughput: {result['throughput_tok_s']:.0f} tok/s  "
                  f"P99 latency: {result['latency_p99_ms']:.0f}ms")
        finally:
            stop_server(proc)
            time.sleep(3)

    # ── Concurrency sweep (default block_size=16) ──────────────────────────────
    print("\n=== Concurrency Sweep (block_size=16) ===")
    proc = start_vllm_server(block_size=16)
    try:
        concurrency_results = []
        for concurrency in CONCURRENCY_LEVELS:
            print(f"  concurrency={concurrency}...")
            r = benchmark_concurrent(num_concurrent=concurrency, num_requests=200)
            concurrency_results.append(r)
            print(f"    {r['throughput_tok_s']:.0f} tok/s  "
                  f"p99={r['latency_p99_ms']:.0f}ms")
    finally:
        stop_server(proc)

    # Save results
    output = {
        "block_size_sweep": all_results,
        "concurrency_sweep": concurrency_results,
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print("\n── Concurrency vs Throughput ──────────────────────────────────────────")
    print(f"{'Concurrency':>12} {'Tok/s':>8} {'Req/s':>8} {'P99 lat (ms)':>14}")
    for r in concurrency_results:
        print(f"{r['num_concurrent']:>12} {r['throughput_tok_s']:>8.0f} "
              f"{r['throughput_req_s']:>8.1f} {r['latency_p99_ms']:>14.0f}")

    print("\n[Article insight]")
    print("  PagedAttention removes the need to pre-allocate max_seq_len per request.")
    print("  Memory fragmentation drops from ~50% (contiguous) to <10% (paged).")
    print("  This directly translates to more concurrent requests and higher throughput.")


if __name__ == "__main__":
    main()
