"""
Unified metrics collector for all LLM inference experiments.
Captures latency, throughput, memory, and efficiency metrics.
"""

import time
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RequestMetrics:
    request_id: str
    prompt_len: int
    output_len: int
    arrival_time: float
    start_time: float         # when inference began (after queue)
    first_token_time: float   # when first token was emitted
    end_time: float           # when last token was emitted

    @property
    def queue_time(self) -> float:
        return self.start_time - self.arrival_time

    @property
    def ttft(self) -> float:
        """Time to first token (from arrival)."""
        return self.first_token_time - self.arrival_time

    @property
    def tpot(self) -> float:
        """Time per output token (decode phase only)."""
        decode_time = self.end_time - self.first_token_time
        return decode_time / max(self.output_len - 1, 1)

    @property
    def e2e_latency(self) -> float:
        return self.end_time - self.arrival_time

    @property
    def throughput_tokens(self) -> float:
        return self.output_len / (self.end_time - self.start_time)


@dataclass
class ExperimentMetrics:
    experiment_id: str
    description: str
    config: dict = field(default_factory=dict)
    requests: list[RequestMetrics] = field(default_factory=list)
    start_wall_time: float = field(default_factory=time.time)
    end_wall_time: Optional[float] = None

    def add(self, req: RequestMetrics) -> None:
        self.requests.append(req)

    def finish(self) -> None:
        self.end_wall_time = time.time()

    # ── Latency ──────────────────────────────────────────────────────────────

    def percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def ttft_stats(self) -> dict:
        vals = [r.ttft for r in self.requests]
        return self._stats(vals, "ttft_ms", scale=1000)

    def tpot_stats(self) -> dict:
        vals = [r.tpot for r in self.requests]
        return self._stats(vals, "tpot_ms", scale=1000)

    def e2e_stats(self) -> dict:
        vals = [r.e2e_latency for r in self.requests]
        return self._stats(vals, "e2e_ms", scale=1000)

    def _stats(self, vals: list[float], name: str, scale: float = 1.0) -> dict:
        if not vals:
            return {}
        scaled = [v * scale for v in vals]
        return {
            f"{name}_mean":   statistics.mean(scaled),
            f"{name}_median": statistics.median(scaled),
            f"{name}_p95":    self.percentile(scaled, 95),
            f"{name}_p99":    self.percentile(scaled, 99),
            f"{name}_min":    min(scaled),
            f"{name}_max":    max(scaled),
        }

    # ── Throughput ────────────────────────────────────────────────────────────

    def throughput_stats(self) -> dict:
        if not self.requests or self.end_wall_time is None:
            return {}
        total_output_tokens = sum(r.output_len for r in self.requests)
        total_input_tokens = sum(r.prompt_len for r in self.requests)
        wall_time = self.end_wall_time - self.start_wall_time
        return {
            "total_requests":       len(self.requests),
            "total_output_tokens":  total_output_tokens,
            "total_input_tokens":   total_input_tokens,
            "wall_time_s":          wall_time,
            "requests_per_sec":     len(self.requests) / wall_time,
            "output_tokens_per_sec": total_output_tokens / wall_time,
            "input_tokens_per_sec": total_input_tokens / wall_time,
        }

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "description":   self.description,
            "config":        self.config,
            **self.ttft_stats(),
            **self.tpot_stats(),
            **self.e2e_stats(),
            **self.throughput_stats(),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"  {self.experiment_id}: {self.description}")
        print(f"{'='*60}")
        print(f"  Requests:        {s.get('total_requests', 0)}")
        print(f"  Throughput:      {s.get('requests_per_sec', 0):.2f} req/s  "
              f"| {s.get('output_tokens_per_sec', 0):.0f} tok/s")
        print(f"  TTFT:            mean={s.get('ttft_ms_mean', 0):.1f}ms  "
              f"p95={s.get('ttft_ms_p95', 0):.1f}ms  "
              f"p99={s.get('ttft_ms_p99', 0):.1f}ms")
        print(f"  TPOT:            mean={s.get('tpot_ms_mean', 0):.1f}ms  "
              f"p95={s.get('tpot_ms_p95', 0):.1f}ms")
        print(f"  E2E latency:     mean={s.get('e2e_ms_mean', 0):.1f}ms  "
              f"p99={s.get('e2e_ms_p99', 0):.1f}ms")

    def save(self, output_dir: str = "results") -> Path:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"{self.experiment_id}.json"
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"  Results saved → {path}")
        return path


# ── Convenience timer context manager ─────────────────────────────────────────

class RequestTimer:
    """Use as context manager to time a single request."""

    def __init__(self, request_id: str, prompt_len: int, arrival_time: float):
        self.request_id = request_id
        self.prompt_len = prompt_len
        self.arrival_time = arrival_time
        self.start_time: float = 0.0
        self.first_token_time: float = 0.0
        self.end_time: float = 0.0
        self.output_len: int = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def mark_first_token(self):
        self.first_token_time = time.perf_counter()

    def __exit__(self, *_):
        self.end_time = time.perf_counter()

    def to_metrics(self) -> RequestMetrics:
        return RequestMetrics(
            request_id=self.request_id,
            prompt_len=self.prompt_len,
            output_len=self.output_len,
            arrival_time=self.arrival_time,
            start_time=self.start_time,
            first_token_time=self.first_token_time if self.first_token_time else self.start_time,
            end_time=self.end_time,
        )


if __name__ == "__main__":
    # Smoke test
    em = ExperimentMetrics("test", "smoke test", config={"batch_size": 8})
    em.start_wall_time = time.time()

    for i in range(10):
        base = time.perf_counter()
        rm = RequestMetrics(
            request_id=f"req-{i}",
            prompt_len=512,
            output_len=128,
            arrival_time=base,
            start_time=base + 0.01,
            first_token_time=base + 0.1,
            end_time=base + 0.5,
        )
        em.add(rm)

    em.finish()
    em.print_summary()
