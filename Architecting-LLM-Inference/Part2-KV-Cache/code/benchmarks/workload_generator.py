"""
Workload generator for KV-cache experiments (Part 2).
Produces synthetic request traces with configurable shared-prefix length,
used across the allocation, footprint, and reuse experiments.
"""

import random
import json
from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    prompt: str
    prompt_len: int
    expected_output_len: int
    arrival_time: float
    priority: int = 0  # 0 = normal, 1 = high


@dataclass
class WorkloadConfig:
    num_requests: int = 100
    arrival_rate: float = 10.0        # requests per second (Poisson)
    prompt_len_mean: int = 512
    prompt_len_std: int = 256
    output_len_mean: int = 128
    output_len_std: int = 64
    shared_prefix_len: int = 0        # for prefix-caching experiments
    seed: int = 42


def generate_synthetic_workload(config: WorkloadConfig) -> list[Request]:
    """
    Generate a synthetic request workload with Poisson arrivals
    and Gaussian prompt/output length distributions.
    """
    random.seed(config.seed)
    requests = []
    current_time = 0.0

    shared_prefix = "A " * config.shared_prefix_len if config.shared_prefix_len > 0 else ""

    for i in range(config.num_requests):
        inter_arrival = random.expovariate(config.arrival_rate)
        current_time += inter_arrival

        prompt_len = max(16, int(random.gauss(config.prompt_len_mean, config.prompt_len_std)))
        output_len = max(1, int(random.gauss(config.output_len_mean, config.output_len_std)))

        unique_part = "word " * max(0, prompt_len - config.shared_prefix_len)
        prompt = shared_prefix + unique_part

        requests.append(Request(
            request_id=f"req-{i:04d}",
            prompt=prompt.strip(),
            prompt_len=prompt_len,
            expected_output_len=output_len,
            arrival_time=current_time,
            priority=random.choices([0, 1], weights=[0.8, 0.2])[0],
        ))

    return requests


def generate_chat_workload(config: WorkloadConfig, turns: int = 5) -> list[Request]:
    """Multi-turn chat workload -- simulates growing context windows per session."""
    random.seed(config.seed)
    requests = []
    current_time = 0.0
    num_sessions = max(1, config.num_requests // turns)

    for session_id in range(num_sessions):
        context = ""
        for turn in range(turns):
            inter_arrival = random.expovariate(config.arrival_rate)
            current_time += inter_arrival

            user_msg = "word " * random.randint(20, 80)
            context += f"User: {user_msg}\nAssistant: response\n"

            requests.append(Request(
                request_id=f"session-{session_id:03d}-turn-{turn}",
                prompt=context,
                prompt_len=len(context.split()),
                expected_output_len=random.randint(50, 200),
                arrival_time=current_time,
            ))

    return requests


def generate_rag_workload(config: WorkloadConfig, num_docs: int = 10, doc_len: int = 800) -> list[Request]:
    """
    RAG-style workload -- a small pool of documents gets reused as prefixes
    across many queries, with a short unique question appended each time.
    Useful for prefix-caching and radix-tree experiments.
    """
    random.seed(config.seed)
    docs = [f"DOC{d} " * doc_len for d in range(num_docs)]
    requests = []
    current_time = 0.0

    for i in range(config.num_requests):
        current_time += random.expovariate(config.arrival_rate)
        doc = random.choice(docs)
        question = "question " * random.randint(10, 30)
        prompt = doc + question
        requests.append(Request(
            request_id=f"rag-{i:04d}",
            prompt=prompt,
            prompt_len=len(prompt.split()),
            expected_output_len=max(1, int(random.gauss(config.output_len_mean, config.output_len_std))),
            arrival_time=current_time,
        ))

    return requests


def load_sharegpt(path: str, num_requests: int = 100, seed: int = 42) -> list[Request]:
    """Load ShareGPT dataset for realistic prompt length distribution."""
    random.seed(seed)
    with open(path) as f:
        data = json.load(f)

    requests = []
    sampled = random.sample(data, min(num_requests, len(data)))
    current_time = 0.0

    for i, conv in enumerate(sampled):
        current_time += random.expovariate(10.0)
        turns = conv.get("conversations", [])
        if not turns:
            continue
        prompt = turns[0].get("value", "")
        output_len = len(turns[1].get("value", "").split()) if len(turns) > 1 else 128

        requests.append(Request(
            request_id=f"sharegpt-{i:04d}",
            prompt=prompt,
            prompt_len=len(prompt.split()),
            expected_output_len=output_len,
            arrival_time=current_time,
        ))

    return requests


def print_workload_stats(requests: list[Request]) -> None:
    prompt_lens = [r.prompt_len for r in requests]
    output_lens = [r.expected_output_len for r in requests]
    arrivals = [r.arrival_time for r in requests]

    print(f"Requests:      {len(requests)}")
    print(f"Prompt len:    mean={sum(prompt_lens)/len(prompt_lens):.0f}  "
          f"min={min(prompt_lens)}  max={max(prompt_lens)}")
    print(f"Output len:    mean={sum(output_lens)/len(output_lens):.0f}  "
          f"min={min(output_lens)}  max={max(output_lens)}")
    duration = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 1.0
    print(f"Duration:      {duration:.1f}s  "
          f"(avg arrival rate: {len(requests)/max(duration, 1e-6):.1f} req/s)")


if __name__ == "__main__":
    cfg = WorkloadConfig(num_requests=200, arrival_rate=10.0, shared_prefix_len=64)
    reqs = generate_synthetic_workload(cfg)
    print_workload_stats(reqs)
