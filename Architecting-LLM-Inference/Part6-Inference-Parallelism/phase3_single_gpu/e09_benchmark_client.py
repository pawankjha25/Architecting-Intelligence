"""
E09 — Benchmark Client
=======================
Sends concurrent requests to a running vLLM server, measures
TTFT, TPOT, E2E latency, and throughput at varying request rates.
Also captures live GPU utilization and memory usage via pynvml.
"""

import asyncio
import aiohttp
import time
import json
import os
import argparse
import random
import statistics
import threading

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

# ── Synthetic workload (ShareGPT-style length distribution) ───────────────────
PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in detail, covering its history, key principles, and modern applications.",
    "Write a comprehensive guide on {topic} including best practices and common pitfalls.",
    "Compare and contrast {topic} with related approaches. What are the trade-offs?",
    "You are an expert in {topic}. A student asks: what are the most important things to understand?",
    "Summarize the key developments in {topic} over the last decade.",
]

TOPICS = [
    "machine learning", "distributed systems", "database indexing", "neural networks",
    "operating systems", "computer vision", "natural language processing", "cryptography",
    "reinforcement learning", "graph algorithms", "cloud computing", "microservices",
    "transformer architecture", "attention mechanisms", "GPU computing", "memory management",
]

def make_prompt(prompt_len_tokens: int) -> str:
    topic = random.choice(TOPICS)
    template = random.choice(PROMPT_TEMPLATES)
    base = template.format(topic=topic)
    # Pad to approximate token length (1 token ≈ 4 chars)
    padding = " ".join(["detail"] * max(0, (prompt_len_tokens * 4 - len(base)) // 6))
    return base + " " + padding

def generate_requests(n: int, prompt_len_mean=256, prompt_len_std=128,
                      output_len_mean=128, output_len_std=64) -> list[dict]:
    random.seed(42)
    reqs = []
    for i in range(n):
        prompt_len = max(32, int(random.gauss(prompt_len_mean, prompt_len_std)))
        output_len = max(16, int(random.gauss(output_len_mean, output_len_std)))
        reqs.append({
            "id": f"req-{i:04d}",
            "prompt": make_prompt(prompt_len),
            "max_tokens": output_len,
        })
    return reqs


# ── GPU Monitor ───────────────────────────────────────────────────────────────

class GPUMonitor:
    """
    Background thread that polls GPU utilization + memory every second.
    Summarizes into mean/peak stats to attach to benchmark results.
    Falls back gracefully if pynvml is not available.
    """

    def __init__(self, gpu_index: int = 0, interval: float = 1.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._available = _PYNVML_AVAILABLE
        if self._available:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                name_bytes = pynvml.nvmlDeviceGetName(self._handle)
                self.gpu_name = name_bytes.decode() if isinstance(name_bytes, bytes) else name_bytes
            except Exception:
                self._available = False
                self.gpu_name = "unknown"
        else:
            self.gpu_name = "unknown (pynvml not installed)"

    def start(self):
        if self._available:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

    def _poll(self):
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.samples.append({
                    "gpu_util_pct": util.gpu,
                    "mem_used_gb":  round(mem.used / 1e9, 2),
                    "mem_total_gb": round(mem.total / 1e9, 2),
                })
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        if not self._available or not self.samples:
            return {"gpu_monitor": "unavailable"}
        gpu_utils = [s["gpu_util_pct"] for s in self.samples]
        mem_used  = [s["mem_used_gb"]  for s in self.samples]
        return {
            "gpu_name":           self.gpu_name,
            "gpu_util_mean_pct":  round(statistics.mean(gpu_utils), 1),
            "gpu_util_peak_pct":  max(gpu_utils),
            "mem_used_mean_gb":   round(statistics.mean(mem_used), 2),
            "mem_used_peak_gb":   max(mem_used),
            "mem_total_gb":       self.samples[0]["mem_total_gb"],
            "num_samples":        len(self.samples),
        }


# ── Single streaming request ───────────────────────────────────────────────────

async def send_request(session: aiohttp.ClientSession, base_url: str,
                       model: str, req: dict) -> dict:
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": req["prompt"],
        "max_tokens": req["max_tokens"],
        "temperature": 0.0,
        "stream": True,
    }

    arrival = time.perf_counter()
    first_token_time = None
    token_count = 0

    try:
        async with session.post(url, json=payload) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
                except Exception:
                    continue

        end_time = time.perf_counter()
        ttft = (first_token_time - arrival) * 1000 if first_token_time else None
        e2e = (end_time - arrival) * 1000
        tpot = ((end_time - first_token_time) / max(token_count - 1, 1) * 1000
                if first_token_time and token_count > 1 else None)

        return {
            "request_id": req["id"],
            "arrival": arrival,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "e2e_ms": e2e,
            "output_tokens": token_count,
            "success": True,
        }
    except Exception as e:
        return {"request_id": req["id"], "success": False, "error": str(e)}


# ── Rate-limited sender ────────────────────────────────────────────────────────

async def run_benchmark(base_url: str, model: str, requests: list[dict],
                        request_rate: float) -> tuple[list[dict], dict]:
    """Send requests at given rate (req/s), collect metrics + GPU stats."""
    connector = aiohttp.TCPConnector(limit=512)
    timeout = aiohttp.ClientTimeout(total=300)

    tasks = []
    monitor = GPUMonitor()
    monitor.start()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i, req in enumerate(requests):
            target_time = i / request_rate
            task = asyncio.create_task(send_request(session, base_url, model, req))
            tasks.append(task)

            if i < len(requests) - 1:
                next_target = (i + 1) / request_rate
                sleep_time = next_target - target_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        results = await asyncio.gather(*tasks)

    gpu_stats = monitor.stop()
    return list(results), gpu_stats


# ── Summarize ─────────────────────────────────────────────────────────────────

def summarize(results: list[dict], request_rate: float) -> dict:
    successful = [r for r in results if r.get("success")]
    ttfts = [r["ttft_ms"] for r in successful if r.get("ttft_ms") is not None]
    tpots = [r["tpot_ms"] for r in successful if r.get("tpot_ms") is not None]
    e2es  = [r["e2e_ms"]  for r in successful if r.get("e2e_ms")  is not None]
    total_tokens = sum(r.get("output_tokens", 0) for r in successful)

    def pct(data, p):
        if not data: return 0
        idx = int(len(data) * p / 100)
        return sorted(data)[min(idx, len(data)-1)]

    total_time_s = (max(r["e2e_ms"] for r in successful) / 1000) if successful else 1
    throughput = total_tokens / total_time_s if total_time_s > 0 else 0

    return {
        "request_rate": request_rate,
        "num_requests": len(results),
        "successful": len(successful),
        "ttft_ms_mean":   round(statistics.mean(ttfts), 1) if ttfts else 0,
        "ttft_ms_median": round(statistics.median(ttfts), 1) if ttfts else 0,
        "ttft_ms_p95":    round(pct(ttfts, 95), 1) if ttfts else 0,
        "ttft_ms_p99":    round(pct(ttfts, 99), 1) if ttfts else 0,
        "tpot_ms_mean":   round(statistics.mean(tpots), 1) if tpots else 0,
        "e2e_ms_mean":    round(statistics.mean(e2es), 1) if e2es else 0,
        "e2e_ms_p99":     round(pct(e2es, 99), 1) if e2es else 0,
        "output_tokens_per_sec": round(throughput, 1),
        "total_output_tokens": total_tokens,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--request-rates", nargs="+", type=float,
                        default=[1, 5, 10, 20, 50])
    parser.add_argument("--result-dir", default="results/e09_baseline")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    requests = generate_requests(args.num_prompts)

    all_results = []
    print(f"\n{'Rate (req/s)':>12} {'Tok/s':>8} {'TTFT_mean':>10} "
          f"{'TTFT_p99':>10} {'TPOT_mean':>10} {'E2E_p99':>10} {'GPU%':>8} {'Mem(GB)':>9}")
    print("-" * 85)

    for rate in args.request_rates:
        print(f"  Running rate={rate} req/s ...", flush=True)
        results, gpu_stats = await run_benchmark(args.base_url, args.model, requests, rate)
        summary = summarize(results, rate)
        summary["gpu"] = gpu_stats
        all_results.append(summary)

        gpu_util = gpu_stats.get("gpu_util_mean_pct", "n/a")
        gpu_mem  = gpu_stats.get("mem_used_peak_gb", "n/a")
        print(f"{rate:>12} {summary['output_tokens_per_sec']:>8.0f} "
              f"{summary['ttft_ms_mean']:>10.0f} {summary['ttft_ms_p99']:>10.0f} "
              f"{summary['tpot_ms_mean']:>10.0f} {summary['e2e_ms_p99']:>10.0f} "
              f"  GPU={gpu_util}% mem={gpu_mem}GB")

        # Save per-rate result
        with open(f"{args.result_dir}/baseline_rps{int(rate)}.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Save combined
    save_data = {
        "experiment_id": "E09_baseline",
        "model": args.model,
        "description": "vLLM baseline — throughput vs latency at varying request rates",
        "rate_sweep": all_results,
    }
    with open(f"{args.result_dir}/E09_baseline.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved → {args.result_dir}/E09_baseline.json")


if __name__ == "__main__":
    asyncio.run(main())
