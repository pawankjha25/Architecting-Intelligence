"""
E12 �� Prefix Caching Client
Sends requests with a long shared system prompt to measure TTFT reduction.
Usage: python e12_prefix_caching_client.py --port 8000 --model Qwen/... --label no_prefix_cache
"""

import argparse
import asyncio
import aiohttp
import time
import json
import statistics
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a world-class expert assistant with deep knowledge in science, "
    "technology, history, mathematics, and literature. "
    "Always provide thorough, accurate, and well-structured answers. "
) * 32   # ~512 tokens — long enough to make prefix caching worthwhile

QUERIES = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "What is 17 times 23?",
    "Name three programming languages.",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "Describe the water cycle briefly.",
    "What year did World War II end?",
    "What is machine learning?",
    "Name the planets in our solar system.",
] * 10   # repeat to get 100 requests


async def send_request(session, port: int, model: str, query: str,
                        sem: asyncio.Semaphore) -> dict:
    async with sem:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "max_tokens": 64,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        async with session.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            t1 = time.perf_counter()
            return {
                "latency_s": t1 - t0,
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }


async def run(port: int, model: str, label: str, output_dir: str):
    results = []
    sem = asyncio.Semaphore(16)

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, port, model, q, sem) for q in QUERIES]
        # Warmup: first request populates the prefix cache
        print(f"  Warmup request (populates cache)...")
        warmup = await send_request(session, port, model, QUERIES[0], sem)
        print(f"  Warmup TTFT: {warmup['latency_s']*1000:.0f}ms")

        print(f"  Running {len(QUERIES)} requests...")
        results = await asyncio.gather(*tasks)

    latencies = [r["latency_s"] * 1000 for r in results]
    print(f"  [{label}] TTFT mean={statistics.mean(latencies):.0f}ms  "
          f"p50={statistics.median(latencies):.0f}ms  "
          f"p99={sorted(latencies)[int(len(latencies)*0.99)]:.0f}ms")

    out = {
        "label": label,
        "warmup_latency_ms": warmup["latency_s"] * 1000,
        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "system_prompt_tokens": len(SYSTEM_PROMPT.split()),
        "num_requests": len(QUERIES),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/{label}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--label", default="test")
    parser.add_argument("--output-dir", default="results/e12_prefix_caching")
    args = parser.parse_args()
    asyncio.run(run(args.port, args.model, args.label, args.output_dir))
