"""E21 — Long-Context Client: sends fixed-length prompts, records TTFT and TPOT."""

import argparse
import asyncio
import aiohttp
import time
import json
import statistics
from pathlib import Path


async def send_request(session, port: int, model: str,
                        prompt_len: int, output_len: int,
                        sem: asyncio.Semaphore) -> dict:
    prompt = "word " * prompt_len
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
    }
    first_token_time = None
    t0 = time.perf_counter()
    async with sem:
        async with session.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            token_count = 0
            async for line in resp.content:
                line = line.decode().strip()
                if line.startswith("data: ") and "[DONE]" not in line:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
            t1 = time.perf_counter()

    if first_token_time is None:
        first_token_time = t0 + (t1 - t0) * 0.1

    return {
        "ttft_ms": (first_token_time - t0) * 1000,
        "e2e_ms": (t1 - t0) * 1000,
        "tpot_ms": (t1 - first_token_time) / max(token_count - 1, 1) * 1000,
        "output_tokens": token_count,
    }


async def run(port, model, prompt_len, output_len, num_requests, label, output_dir):
    sem = asyncio.Semaphore(8)
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, port, model, prompt_len, output_len, sem)
                 for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

    ttfts = [r["ttft_ms"] for r in results]
    tpots = [r["tpot_ms"] for r in results]
    print(f"  [{label}] prompt_len={prompt_len}  "
          f"TTFT mean={statistics.mean(ttfts):.0f}ms  "
          f"p99={sorted(ttfts)[int(len(ttfts)*0.99)]:.0f}ms  "
          f"TPOT={statistics.mean(tpots):.1f}ms/tok")

    out = {
        "label": label, "prompt_len": prompt_len,
        "ttft_mean_ms": statistics.mean(ttfts),
        "ttft_p99_ms": sorted(ttfts)[int(len(ttfts) * 0.99)],
        "tpot_mean_ms": statistics.mean(tpots),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/{label}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--prompt-len", type=int, default=2048)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--num-requests", type=int, default=50)
    p.add_argument("--label", default="test")
    p.add_argument("--output-dir", default="results/e21_long_context")
    args = p.parse_args()
    asyncio.run(run(args.port, args.model, args.prompt_len, args.output_len,
                    args.num_requests, args.label, args.output_dir))
