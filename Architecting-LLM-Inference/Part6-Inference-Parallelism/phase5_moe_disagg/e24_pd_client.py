"""
E24 — P/D Disaggregation Client
Sends long-prompt requests to prefill instance, short-continuation requests to decode.
Measures TTFT difference to show prefill/decode specialization effect.
"""

import argparse
import asyncio
import aiohttp
import time
import json
import statistics
from pathlib import Path

LONG_PROMPT = "word " * 1024    # ~1K token prompt (prefill-heavy)
SHORT_PROMPT = "Hello, " * 8    # short prompt (decode-heavy)


async def send(session, port: int, model: str, prompt: str,
               sem: asyncio.Semaphore) -> dict:
    async with sem:
        t0 = time.perf_counter()
        async with session.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 128, "temperature": 0.0},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            await resp.json()
            return {"latency_ms": (time.perf_counter() - t0) * 1000, "port": port}


async def run(prefill_port, decode_port, model, output_dir):
    sem = asyncio.Semaphore(16)

    async with aiohttp.ClientSession() as session:
        # Long prompts → prefill instance
        prefill_tasks = [send(session, prefill_port, model, LONG_PROMPT, sem)
                         for _ in range(50)]
        # Short prompts → decode instance
        decode_tasks = [send(session, decode_port, model, SHORT_PROMPT, sem)
                        for _ in range(50)]

        prefill_results = await asyncio.gather(*prefill_tasks)
        decode_results = await asyncio.gather(*decode_tasks)

    def stats(results, label):
        lats = [r["latency_ms"] for r in results]
        print(f"  {label}: mean={statistics.mean(lats):.0f}ms  "
              f"p99={sorted(lats)[int(len(lats)*0.99)]:.0f}ms")
        return {"label": label, "mean_ms": statistics.mean(lats),
                "p99_ms": sorted(lats)[int(len(lats) * 0.99)]}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "prefill_optimized": stats(prefill_results, f"Prefill (port {prefill_port})"),
        "decode_optimized":  stats(decode_results,  f"Decode  (port {decode_port})"),
    }
    with open(f"{output_dir}/pd_comparison.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefill-port", type=int, default=8000)
    p.add_argument("--decode-port", type=int, default=8001)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output-dir", default="results/e24_pd_disaggregation")
    args = p.parse_args()
    asyncio.run(run(args.prefill_port, args.decode_port, args.model, args.output_dir))
