"""
Sim01 -- Cache-Aware Routing
==============================
Goal: Compare load-balancing policies for routing requests across multiple
      GPU replicas, where each replica has its own local prefix cache.
      Naive least-loaded routing ignores WHICH worker already has a
      request's prefix cached -- cache-aware routing trades a bit of
      load-balance precision for a much higher prefix cache hit rate.

Pure Python discrete-event simulation, no GPU/model needed. This is
explicitly a SIMULATION, not a hardware-measured result (per the article's
own labeling convention) -- real routing behavior depends on network RTT,
real cache eviction, and real prefill/decode cost curves this sim only
approximates.

Policies compared:
  - round_robin        -- cycles through workers regardless of load or cache
  - least_queue         -- always picks the worker with the shortest queue
  - power_of_two        -- samples 2 random workers, picks the shorter queue
  - session_affinity    -- routes by session_id hash (sticky per session)
  - prefix_affinity     -- routes by prefix_id hash (sticky per shared prefix)
  - cache_value_aware   -- scores workers by (queue_length - cache_hit_bonus)

Run: python3 phase5_simulation/sim01_cache_aware_routing.py
"""

import sys
import os
import json
import random
import heapq
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.workload_generator import WorkloadConfig, generate_rag_workload

NUM_WORKERS = 4
CACHE_SLOTS_PER_WORKER = 3   # how many distinct prefixes each worker can keep cached
FULL_PREFILL_TIME = 1.0      # synthetic seconds if prefix is NOT cached on the chosen worker
CACHED_PREFILL_TIME = 0.2    # synthetic seconds if prefix IS cached (skips most of prefill)
DECODE_TIME_PER_TOKEN = 0.01


@dataclass
class Worker:
    worker_id: int
    free_at: float = 0.0
    cached_prefixes: deque = field(default_factory=lambda: deque(maxlen=CACHE_SLOTS_PER_WORKER))
    total_requests: int = 0
    cache_hits: int = 0

    def has_cached(self, prefix_id) -> bool:
        return prefix_id in self.cached_prefixes

    def serve(self, arrival_time: float, prefix_id, output_len: int) -> dict:
        hit = self.has_cached(prefix_id)
        prefill_time = CACHED_PREFILL_TIME if hit else FULL_PREFILL_TIME
        service_time = prefill_time + output_len * DECODE_TIME_PER_TOKEN

        start = max(arrival_time, self.free_at)
        finish = start + service_time
        wait = start - arrival_time
        latency = finish - arrival_time

        self.free_at = finish
        self.total_requests += 1
        if hit:
            self.cache_hits += 1
        else:
            self.cached_prefixes.append(prefix_id)

        return {"wait": wait, "latency": latency, "cache_hit": hit}


def make_workers() -> list[Worker]:
    return [Worker(worker_id=i) for i in range(NUM_WORKERS)]


# -- Routing policies -----------------------------------------------------------

def route_round_robin(workers, req, counter):
    idx = counter[0] % len(workers)
    counter[0] += 1
    return workers[idx]


def route_least_queue(workers, req, counter):
    return min(workers, key=lambda w: w.free_at)


def route_power_of_two(workers, req, counter, rng):
    a, b = rng.sample(workers, 2)
    return a if a.free_at <= b.free_at else b


def route_session_affinity(workers, req, counter):
    idx = hash(req["session_id"]) % len(workers)
    return workers[idx]


def route_prefix_affinity(workers, req, counter):
    idx = hash(req["prefix_id"]) % len(workers)
    return workers[idx]


def route_cache_value_aware(workers, req, counter):
    """Score = expected wait time if routed here, minus a bonus for a cache hit."""
    def score(w):
        wait = max(0.0, w.free_at - req["arrival_time"])
        bonus = 0.5 if w.has_cached(req["prefix_id"]) else 0.0
        return wait - bonus
    return min(workers, key=score)


POLICIES = {
    "round_robin": lambda workers, req, counter, rng: route_round_robin(workers, req, counter),
    "least_queue": lambda workers, req, counter, rng: route_least_queue(workers, req, counter),
    "power_of_two": lambda workers, req, counter, rng: route_power_of_two(workers, req, counter, rng),
    "session_affinity": lambda workers, req, counter, rng: route_session_affinity(workers, req, counter),
    "prefix_affinity": lambda workers, req, counter, rng: route_prefix_affinity(workers, req, counter),
    "cache_value_aware": lambda workers, req, counter, rng: route_cache_value_aware(workers, req, counter),
}


def build_requests(num_requests=400, num_prefixes=6, seed=42):
    rng = random.Random(seed)
    cfg = WorkloadConfig(num_requests=num_requests, arrival_rate=20.0, seed=seed)
    raw = generate_rag_workload(cfg, num_docs=num_prefixes, doc_len=400)
    requests = []
    for i, r in enumerate(raw):
        requests.append({
            "request_id": r.request_id,
            "arrival_time": r.arrival_time,
            "prefix_id": hash(r.prompt[:50]) % num_prefixes,   # which shared doc this maps to
            "session_id": f"session-{i % 40}",                  # 40 distinct sessions
            "output_len": r.expected_output_len,
        })
    return requests


def run_policy(policy_name: str, requests: list[dict]) -> dict:
    rng = random.Random(7)
    workers = make_workers()
    counter = [0]
    waits, latencies, hits = [], [], 0

    for req in requests:
        worker = POLICIES[policy_name](workers, req, counter, rng)
        result = worker.serve(req["arrival_time"], req["prefix_id"], req["output_len"])
        waits.append(result["wait"])
        latencies.append(result["latency"])
        hits += int(result["cache_hit"])

    load_counts = [w.total_requests for w in workers]
    load_mean = sum(load_counts) / len(load_counts)
    load_std = (sum((c - load_mean) ** 2 for c in load_counts) / len(load_counts)) ** 0.5

    return {
        "policy": policy_name,
        "avg_wait_s": sum(waits) / len(waits),
        "avg_latency_s": sum(latencies) / len(latencies),
        "p99_latency_s": sorted(latencies)[int(len(latencies) * 0.99)],
        "cache_hit_rate_pct": hits / len(requests) * 100,
        "load_balance_stddev": load_std,
        "per_worker_load": load_counts,
    }


def main():
    print("Sim01 -- Cache-Aware Routing\n")

    requests = build_requests(num_requests=400, num_prefixes=6)
    print(f"Requests: {len(requests)}, workers: {NUM_WORKERS}, "
          f"cache slots/worker: {CACHE_SLOTS_PER_WORKER}, distinct prefixes: 6\n")

    print(f"{'policy':>18} | {'avg wait (s)':>12} | {'avg latency (s)':>16} | "
          f"{'cache hit %':>11} | {'load stddev':>12}")
    print("-" * 80)

    results = []
    for policy_name in POLICIES:
        r = run_policy(policy_name, requests)
        results.append(r)
        print(f"{policy_name:>18} | {r['avg_wait_s']:>12.3f} | {r['avg_latency_s']:>16.3f} | "
              f"{r['cache_hit_rate_pct']:>10.1f}% | {r['load_balance_stddev']:>12.2f}")

    print("\n[Article insight]")
    print("  Least-queue and power-of-two balance load well but ignore cache locality --")
    print("  they get low cache-hit rates because they don't consider WHICH worker")
    print("  already has a request's prefix. Prefix affinity gets the highest hit rate")
    print("  by always routing the same prefix to the same worker, but at the cost of")
    print("  worse load balance if prefixes aren't uniformly popular. Cache-value-aware")
    print("  routing is the middle ground: it scores workers by wait time MINUS a cache")
    print("  bonus, capturing most of the hit-rate gain without fully sacrificing load")
    print("  balance -- this is the real design tension production routers navigate.")

    Path("results").mkdir(exist_ok=True)
    with open("results/Sim01_cache_aware_routing.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved -> results/Sim01_cache_aware_routing.json")


if __name__ == "__main__":
    main()
