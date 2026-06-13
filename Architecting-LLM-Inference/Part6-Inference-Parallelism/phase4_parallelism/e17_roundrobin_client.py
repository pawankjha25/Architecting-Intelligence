"""
E17 — Round-Robin Client for Data Parallel Experiment
Distributes requests evenly across N vLLM replicas.
"""

import argparse
import asyncio
import aiohttp
import time
import json
import statistics
import itertools
from pathlib import Path


async def send_request(session, base_url: str, model: str,
                        req_id: int, sem: asyncio.Semaphore) -> dict:
    prompt = "Explain transformer attention in detail. " * 4
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    async with sem:
        t0 = time.perf_counter()
        async with session.post(f"{base_url}/v1/chat/completions",
                                json=payload,
                                timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()
            t1 = time.perf_counter()
            return {
                "req_id": req_id,
                "url": base_url,
                "latency_s": t1 - t0,
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }


async def run(ports: list[int], model: str, num_requests: int,
              request_rate: float, output: str):
    urls = [f"http://localhost:{p}" for p in ports]
    url_cycle = itertools.cycle(urls)
    sem = asyncio.Semaphore(num_requests)

    async with aiohttp.ClientSession() as session:
        tasks = []
        interval = 1.0 / request_rate
        for i in range(num_requests):
            url = next(url_cycle)
            tasks.append(send_request(session, url, model, i, sem))
            await asyncio.sleep(interval)
        results = await asyncio.gather(*tasks)

    latencies = [r["latency_s"] * 1000 for r in results]
    total_tokens = sum(r["output_tokens"] for r in results)
    wall = results[-1]["latency_s"] + latencies[0] / 1000

    output_data = {
        "num_replicas": len(ports),
        "num_requests": num_requests,
        "throughput_req_s": num_requests / (max(latencies) / 1000),
        "output_tokens_total": total_tokens,
        "latency_mean_ms": statistics.mean(latencies),
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
    }

    print(f"  2-replica: {output_data['throughput_req_s']:.1f} req/s  "
          f"p99={output_data['latency_p99_ms']:.0f}ms")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", type=int, nargs="+", default=[8000, 8001])
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-requests", type=int, default=400)
    parser.add_argument("--request-rate", type=float, default=40.0)
    parser.add_argument("--output", default="results/e17_data_parallel/two_replicas.json")
    args = parser.parse_args()
    asyncio.run(run(args.ports, args.model, args.num_requests,
                    args.request_rate, args.output))
