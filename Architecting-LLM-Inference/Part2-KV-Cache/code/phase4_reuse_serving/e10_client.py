"""
E10 client -- measures TTFT for requests sharing a common prefix, split into
"cold" (first request that establishes the prefix in cache) and "warm"
(subsequent requests that should hit the cached prefix, if prefix caching
is enabled server-side). Requires `aiohttp` (RunPod only).
"""

import sys
import os
import argparse
import asyncio
import time
import json

import aiohttp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.metrics_collector import ExperimentMetrics, RequestMetrics


async def send_one(session, url, prompt, max_tokens):
    start = time.perf_counter()
    first_token_time = None
    payload = {"model": "default", "prompt": prompt, "max_tokens": max_tokens, "stream": True}
    async with session.post(url, json=payload) as resp:
        async for line in resp.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
    end = time.perf_counter()
    return {
        "ttft_ms": (first_token_time - start) * 1000 if first_token_time else None,
        "e2e_ms": (end - start) * 1000,
    }


async def run(args):
    shared_prefix = "DOCUMENT_CONTEXT " * args.shared_prefix_len
    url = f"http://localhost:{args.port}/v1/completions"

    cold_results = []
    warm_results = []

    async with aiohttp.ClientSession() as session:
        # First request establishes the prefix -- this is "cold" even with
        # prefix caching enabled, since nothing is cached yet.
        first_prompt = shared_prefix + "question 0"
        cold_results.append(await send_one(session, url, first_prompt, 32))

        # Subsequent requests reuse the same prefix -- "warm" if caching works.
        for i in range(1, args.num_requests):
            prompt = shared_prefix + f"question {i}"
            warm_results.append(await send_one(session, url, prompt, 32))

    def summarize(results, label):
        ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
        return {
            "label": label,
            "count": len(results),
            "ttft_ms_mean": sum(ttfts) / len(ttfts) if ttfts else None,
            "ttft_ms_min": min(ttfts) if ttfts else None,
            "ttft_ms_max": max(ttfts) if ttfts else None,
        }

    summary = {
        "mode": args.mode,
        "shared_prefix_len": args.shared_prefix_len,
        "cold": summarize(cold_results, "cold (first request)"),
        "warm": summarize(warm_results, f"warm ({args.num_requests - 1} subsequent requests)"),
    }
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", type=str, default="enabled")
    parser.add_argument("--shared-prefix-len", type=int, default=2000)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/e10_prefix_caching/result.json")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
