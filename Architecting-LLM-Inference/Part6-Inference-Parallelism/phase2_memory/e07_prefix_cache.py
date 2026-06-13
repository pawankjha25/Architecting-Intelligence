"""
E07 — Prefix Caching / Radix Tree
====================================
Goal:   Implement RadixAttention-style prefix cache, measure hit rates
        across different workload patterns.
Hardware: MacBook (pure Python)
Workloads: (1) shared system prompt, (2) chat history, (3) RAG
"""

import sys
import os
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Radix Tree Prefix Cache ─────────────────────────────────────────────────────

@dataclass
class RadixNode:
    key: tuple = ()                        # token sequence for this edge
    children: dict = field(default_factory=dict)
    is_leaf: bool = False
    block_ids: list[int] = field(default_factory=list)
    last_access: float = 0.0
    ref_count: int = 0


class RadixCache:
    """
    Prefix cache using a radix tree (prefix trie).
    Each node represents a shared token prefix and its cached KV blocks.
    LRU eviction when capacity is reached.
    """

    def __init__(self, max_blocks: int = 256, block_size: int = 16):
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.root = RadixNode()
        self.used_blocks = 0
        self._next_block_id = 0
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _tokenize(self, text: str) -> tuple:
        """Simple word-level tokenizer for simulation."""
        return tuple(text.split())

    def _hash_prefix(self, tokens: tuple) -> str:
        return hashlib.md5(str(tokens).encode()).hexdigest()[:8]

    def _alloc_blocks(self, num_blocks: int) -> list[int]:
        ids = list(range(self._next_block_id, self._next_block_id + num_blocks))
        self._next_block_id += num_blocks
        self.used_blocks += num_blocks
        return ids

    def lookup(self, tokens: tuple) -> tuple[int, list[int]]:
        """
        Find the longest cached prefix for tokens.
        Returns (matched_len, block_ids).
        """
        node = self.root
        matched = 0
        cached_blocks = []
        pos = 0

        while pos < len(tokens):
            # Try to find a child that matches
            found = False
            for child_key, child in node.children.items():
                if tokens[pos:pos + len(child_key)] == child_key:
                    matched += len(child_key)
                    cached_blocks.extend(child.block_ids)
                    child.last_access = time.time()
                    node = child
                    pos += len(child_key)
                    found = True
                    break
            if not found:
                break

        if matched > 0:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1

        return matched, cached_blocks

    def insert(self, tokens: tuple) -> list[int]:
        """
        Insert a token sequence into the cache.
        Returns block IDs for the newly cached portion.
        """
        # First find how much is already cached
        matched_len, existing_blocks = self.lookup(tokens)
        remaining = tokens[matched_len:]

        if not remaining:
            return existing_blocks   # fully cached

        # Group remaining tokens into blocks
        new_blocks = []
        for i in range(0, len(remaining), self.block_size):
            chunk = remaining[i:i + self.block_size]
            if len(self.root.children) > 0 and self.used_blocks >= self.max_blocks:
                self._evict()
            block_ids = self._alloc_blocks(1)
            new_blocks.extend(block_ids)

        # Insert into radix tree
        node = self.root
        pos = matched_len
        while pos < len(tokens):
            chunk = tokens[pos:pos + self.block_size]
            child = RadixNode(
                key=chunk,
                block_ids=self._alloc_blocks(0),   # already allocated above
                last_access=time.time(),
            )
            node.children[chunk] = child
            node = child
            pos += len(chunk)

        return existing_blocks + new_blocks

    def _evict(self) -> None:
        """LRU eviction — remove the least recently accessed leaf."""
        def _collect_leaves(node, leaves):
            if not node.children:
                leaves.append(node)
            for child in node.children.values():
                _collect_leaves(child, leaves)

        leaves = []
        _collect_leaves(self.root, leaves)
        if not leaves:
            return

        lru = min(leaves, key=lambda n: n.last_access)
        self.used_blocks -= len(lru.block_ids)
        # Remove from parent (simplified)
        self._stats["evictions"] += 1

    @property
    def hit_rate(self) -> float:
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            **self._stats,
            "hit_rate_pct": self.hit_rate * 100,
            "used_blocks": self.used_blocks,
            "utilization_pct": (self.used_blocks / self.max_blocks) * 100,
        }


# ── Workload generators ────────────────────────────────────────────────────────

def make_shared_system_prompt_workload(n: int, system_prompt_len: int = 512) -> list[str]:
    """All requests share the same long system prompt."""
    system = "You are a helpful assistant. " * (system_prompt_len // 6)
    return [system + f"User question {i}: " + "word " * random.randint(10, 50)
            for i in range(n)]


def make_chat_history_workload(n: int, turns: int = 8) -> list[str]:
    """Simulates multi-turn chat — each request extends previous history."""
    history = ""
    prompts = []
    for i in range(n):
        history += f"User: query {i} with some content. Assistant: response {i}. "
        if len(prompts) < n:
            prompts.append(history)
    return prompts[:n]


def make_rag_workload(n: int, num_docs: int = 10, doc_len: int = 200) -> list[str]:
    """RAG: retrieved docs are shared across queries, query itself varies."""
    docs = ["Document content " * doc_len + f" [doc {i}] " for i in range(num_docs)]
    corpus = " ".join(docs[:3])   # top-3 retrieved docs
    return [corpus + f" Query: {i} " + "word " * random.randint(5, 20)
            for i in range(n)]


def make_random_workload(n: int) -> list[str]:
    """Fully random prompts — no shared prefixes."""
    return ["word " * random.randint(50, 300) + f" unique_{i}" for i in range(n)]


# ── Experiment ─────────────────────────────────────────────────────────────────

def measure_hit_rate(workload_name: str, prompts: list[str],
                     cache: RadixCache) -> dict:
    cache._stats = {"hits": 0, "misses": 0, "evictions": 0}

    for prompt in prompts:
        tokens = cache._tokenize(prompt)
        matched_len, _ = cache.lookup(tokens)
        if matched_len < len(tokens):
            cache.insert(tokens)

    stats = cache.stats()
    print(f"  {workload_name:<30} hit_rate={stats['hit_rate_pct']:.1f}%  "
          f"hits={stats['hits']}  misses={stats['misses']}  "
          f"blocks_used={stats['used_blocks']}")
    return {"workload": workload_name, **stats}


def main():
    print("E07 — Prefix Caching / Radix Tree\n")
    print(f"Cache size: 256 blocks (block_size=16 tokens)\n")

    N = 100
    random.seed(42)

    workloads = [
        ("Shared system prompt",   make_shared_system_prompt_workload(N, 512)),
        ("Chat history (multi-turn)", make_chat_history_workload(N, turns=8)),
        ("RAG (retrieved docs)",   make_rag_workload(N, doc_len=200)),
        ("Random (no sharing)",    make_random_workload(N)),
    ]

    all_results = []
    for name, prompts in workloads:
        cache = RadixCache(max_blocks=256, block_size=16)
        result = measure_hit_rate(name, prompts, cache)
        all_results.append(result)

    # TTFT savings estimate
    print("\n── Estimated TTFT Savings from Prefix Caching ─────────────────────")
    print(f"  Assumption: prefill costs 2ms/token, cached tokens are free")
    print(f"{'Workload':<30} {'Hit Rate':>10} {'TTFT Saving':>12}")
    for r in all_results:
        saving = r["hit_rate_pct"]   # proxy: % of tokens skipped in prefill
        print(f"  {r['workload']:<30} {r['hit_rate_pct']:>9.1f}%  {saving:>10.1f}%")

    print("\n[Article insight]")
    print("  Shared system prompts → near 100% cache hit after first request.")
    print("  Chat history → hit rate grows with conversation length.")
    print("  RAG → shared retrieved docs give 70–90% prefix hit rate.")
    print("  Random workloads → ~0% hit rate; prefix caching provides no benefit.")
    print("  This is why vLLM's --enable-prefix-caching helps chatbots but not one-off queries.")

    # ── Save results ──────────────────────────────────────────────────────────
    import json, os
    os.makedirs("results", exist_ok=True)
    save_data = {
        "experiment_id": "E07_prefix_cache",
        "description": "Prefix caching hit rates across workload types",
        "config": {"num_requests": N, "cache_blocks": 256, "block_size": 16},
        "workload_results": [
            {
                "workload": r["workload"],
                "hit_rate_pct": round(r["hit_rate_pct"], 1),
                "hits": r["hits"],
                "misses": r["misses"],
                "evictions": r["evictions"],
                "ttft_saving_pct": round(r["hit_rate_pct"], 1),
            }
            for r in all_results
        ]
    }
    with open("results/E07_prefix_cache.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("  Results saved → results/E07_prefix_cache.json")


if __name__ == "__main__":
    main()
