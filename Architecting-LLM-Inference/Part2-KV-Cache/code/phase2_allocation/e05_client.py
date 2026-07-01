"""
E05 (Phase B) client -- async load generator against a live vLLM server.
Used by e05_block_size_sweep_runpod.sh. Requires `aiohttp` (RunPod only).
"""

import sys
import os
import argparse
import asyncio
import time

import aiohttp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.metrics_collector import ExperimentMetrics, RequestMetrics
from benchmarks.workload_generator import WorkloadConfig, generate_synthetic_workload


async def send_request(session, url, req, semaphore, metrics: ExperimentMetrics):
    async with semaphore:
        start = time.perf_counter()
        payload = {
            "model": "default",
            "prompt": req.prompt,
            "max_tokens": req.expected_output_len,
            "stream": True,
        }
        first_token_time = None
        try:
            async with session.post(url, json=payload) as resp:
                async for line in resp.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
        except Exception as e:
            print(f"  request {req.request_id} failed: {e}")
            return
        end = time.perf_counter()
        metrics.add(RequestMetrics(
            request_id=req.request_id,
            prompt_len=req.prompt_len,
            output_len=req.expected_output_len,
            arrival_time=start,
            start_time=start,
            first_token_time=first_token_time or end,
            end_time=end,
        ))


async def run(args):
    cfg = WorkloadConfig(num_requests=args.num_requests, arrival_rate=1000.0)  # fire quickly, gated by semaphore
    requests = generate_synthetic_workload(cfg)

    metrics = ExperimentMetrics(
        experiment_id=f"E05_block_{args.block_size}",
        description=f"Block-size sweep, block_size={args.block_size}",
        config={"block_size": args.block_size, "concurrency": args.concurrency},
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    url = f"http://localhost:{args.port}/v1/completions"

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, url, r, semaphore, metrics) for r in requests]
        await asyncio.gather(*tasks)

    metrics.finish()
    metrics.print_summary()
    metrics.save(output_dir=os.path.dirname(args.output) or ".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/e05_block_size_sweep/result.json")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
