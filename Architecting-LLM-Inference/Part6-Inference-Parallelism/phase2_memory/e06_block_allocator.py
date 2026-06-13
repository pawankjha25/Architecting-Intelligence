"""
E06 — KV Cache Block Allocator (Conceptual)
=============================================
Goal:   Implement the core PagedAttention memory model from scratch.
        Understand the data structures before reading vLLM source code.

Components implemented:
  1. Block — physical memory unit (block_size tokens)
  2. FreeList — O(1) allocate / free
  3. BlockTable — per-sequence mapping of logical → physical blocks
  4. Copy-on-Write (CoW) — enables safe prefix sharing between sequences
  5. Fragmentation measurement

Hardware: MacBook (pure Python — no GPU needed)

Key insight: PagedAttention's memory model is essentially virtual memory
for KV caches. Each sequence has a "page table" (block_table) mapping
logical token positions to physical memory blocks. Shared prefixes map
to the same physical blocks via ref counting — CoW handles divergence.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────────

BLOCK_SIZE = 16          # tokens per block (vLLM default)
TOTAL_BLOCKS = 128       # simulated GPU KV memory


# ── 1. Block ───────────────────────────────────────────────────────────────────

@dataclass
class Block:
    """
    One physical KV cache block = BLOCK_SIZE token slots.
    ref_count > 1 means this block is shared (prefix cache).
    """
    block_id: int
    ref_count: int = 0
    token_ids: list = field(default_factory=list)   # conceptual: tokens stored here

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0

    @property
    def is_shared(self) -> bool:
        return self.ref_count > 1

    @property
    def num_filled(self) -> int:
        return len(self.token_ids)

    @property
    def is_full(self) -> bool:
        return len(self.token_ids) >= BLOCK_SIZE


# ── 2. FreeList ────────────────────────────────────────────────────────────────

class FreeList:
    """
    O(1) allocate and free using a pre-populated list of block IDs.
    In real vLLM this is managed by the BlockAllocator.
    """

    def __init__(self, total_blocks: int):
        self._blocks = {i: Block(block_id=i) for i in range(total_blocks)}
        self._free_ids: list[int] = list(range(total_blocks))

    def allocate(self) -> Optional[Block]:
        """Grab one free block. Returns None if OOM."""
        if not self._free_ids:
            return None
        bid = self._free_ids.pop()
        block = self._blocks[bid]
        block.ref_count = 1
        block.token_ids = []
        return block

    def free(self, block: Block) -> None:
        """Return block to free pool (only when ref_count hits 0)."""
        block.ref_count -= 1
        if block.ref_count == 0:
            block.token_ids = []
            self._free_ids.append(block.block_id)

    def incr_ref(self, block: Block) -> None:
        """Increment ref count — used when sharing a block (prefix cache)."""
        block.ref_count += 1

    @property
    def num_free(self) -> int:
        return len(self._free_ids)

    @property
    def num_used(self) -> int:
        return len(self._blocks) - len(self._free_ids)


# ── 3. BlockTable ──────────────────────────────────────────────────────────────

class BlockTable:
    """
    Per-sequence mapping: logical block index → physical Block.
    This is the "page table" for one sequence's KV cache.
    """

    def __init__(self, seq_id: str, free_list: FreeList):
        self.seq_id = seq_id
        self.free_list = free_list
        self._table: list[Block] = []   # index = logical block number
        self.num_tokens: int = 0

    # ── Append tokens ──────────────────────────────────────────────────────────

    def append_token(self, token_id: int) -> bool:
        """
        Add one token to the sequence. Allocates a new block if needed.
        Returns False if OOM.
        """
        # Need a new block?
        if not self._table or self._table[-1].is_full:
            block = self.free_list.allocate()
            if block is None:
                return False   # GPU OOM
            self._table.append(block)

        self._table[-1].token_ids.append(token_id)
        self.num_tokens += 1
        return True

    # ── Copy-on-Write ──────────────────────────────────────────────────────────

    def cow_last_block(self) -> bool:
        """
        If the last block is shared (ref_count > 1), copy it to a new
        private block before writing. This is Copy-on-Write.

        Why: shared prefix blocks must not be mutated. When a new sequence
        diverges from a shared prefix, it gets its own physical copy of
        the last (partial) block so it can write new tokens independently.
        """
        if not self._table:
            return True

        last = self._table[-1]
        if not last.is_shared:
            return True   # already private, nothing to do

        # Allocate a new block and copy content
        new_block = self.free_list.allocate()
        if new_block is None:
            return False   # OOM during CoW

        new_block.token_ids = last.token_ids.copy()

        # Release our reference to the old shared block
        self.free_list.free(last)

        # Point to new private block
        self._table[-1] = new_block
        return True

    def fork_from(self, parent: "BlockTable") -> None:
        """
        Create a child sequence sharing the parent's prefix blocks.
        All parent blocks get their ref_count incremented (no copy yet).
        CoW happens lazily when the child writes its first new token.
        """
        for block in parent._table:
            self.free_list.incr_ref(block)
            self._table.append(block)
        self.num_tokens = parent.num_tokens

    # ── Free ───────────────────────────────────────────────────────────────────

    def free_all(self) -> None:
        for block in self._table:
            self.free_list.free(block)
        self._table.clear()
        self.num_tokens = 0

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def num_blocks(self) -> int:
        return len(self._table)

    @property
    def internal_fragmentation_tokens(self) -> int:
        """Unused token slots in the last (partial) block."""
        if not self._table:
            return 0
        last = self._table[-1]
        return BLOCK_SIZE - last.num_filled

    def physical_block_ids(self) -> list[int]:
        return [b.block_id for b in self._table]


# ── 4. Fragmentation Measurement ──────────────────────────────────────────────

def measure_fragmentation(free_list: FreeList,
                           sequences: dict[str, BlockTable]) -> dict:
    """
    Internal fragmentation: wasted slots in partially-filled last blocks.
    External fragmentation: N/A with paged allocation (by design, it's zero).
    """
    wasted_slots = sum(s.internal_fragmentation_tokens for s in sequences.values())
    used_slots = sum(s.num_tokens for s in sequences.values())
    total_allocated_slots = sum(s.num_blocks * BLOCK_SIZE for s in sequences.values())

    shared_blocks = sum(
        1 for bid, b in free_list._blocks.items()
        if b.ref_count > 1
    )

    return {
        "used_blocks": free_list.num_used,
        "free_blocks": free_list.num_free,
        "utilization_pct": free_list.num_used / len(free_list._blocks) * 100,
        "internal_frag_tokens": wasted_slots,
        "internal_frag_pct": (wasted_slots / max(total_allocated_slots, 1)) * 100,
        "shared_blocks": shared_blocks,
        "active_sequences": len(sequences),
    }


# ── Demo ───────────────────────────────────────────────────────────────────────

def demo_basic_allocation():
    print("── 1. Basic Allocation ───────────────────────────────────────────────")
    free_list = FreeList(TOTAL_BLOCKS)
    sequences: dict[str, BlockTable] = {}

    # Allocate 5 sequences of varying lengths
    for i, length in enumerate([20, 45, 70, 12, 33]):
        seq = BlockTable(f"seq-{i}", free_list)
        for tok in range(length):
            assert seq.append_token(tok), "OOM"
        sequences[f"seq-{i}"] = seq
        print(f"  seq-{i}: {length} tokens → {seq.num_blocks} blocks  "
              f"(internal frag: {seq.internal_fragmentation_tokens} slots)")

    stats = measure_fragmentation(free_list, sequences)
    print(f"\n  Fragmentation: {stats['internal_frag_pct']:.1f}%  "
          f"({stats['internal_frag_tokens']} wasted slots)")
    print(f"  Utilization:   {stats['utilization_pct']:.1f}%  "
          f"({stats['used_blocks']}/{TOTAL_BLOCKS} blocks used)")

    for seq in sequences.values():
        seq.free_all()


def demo_copy_on_write():
    print("\n── 2. Copy-on-Write (Prefix Sharing) ────────────────────────────────")
    free_list = FreeList(TOTAL_BLOCKS)

    # Parent sequence: simulate a system prompt (32 tokens = 2 full blocks)
    parent = BlockTable("parent", free_list)
    for tok in range(32):
        parent.append_token(tok)

    blocks_before = free_list.num_used
    print(f"  Parent: 32 tokens, {parent.num_blocks} blocks, "
          f"{free_list.num_used} blocks used")

    # Fork two children — they share parent's blocks (no copy yet)
    child_a = BlockTable("child_a", free_list)
    child_b = BlockTable("child_b", free_list)
    child_a.fork_from(parent)
    child_b.fork_from(parent)

    print(f"  After fork (2 children): {free_list.num_used} blocks used "
          f"(still {blocks_before} — sharing!)")
    print(f"  Parent last block ref_count: {parent._table[-1].ref_count}")

    # Child A writes a new token — triggers CoW on its last (shared) block
    child_a.cow_last_block()
    child_a.append_token(999)
    print(f"\n  After child_a writes (CoW triggered): {free_list.num_used} blocks used")
    print(f"  child_a blocks: {child_a.physical_block_ids()}")
    print(f"  child_b blocks: {child_b.physical_block_ids()}")
    print(f"  parent blocks:  {parent.physical_block_ids()}")
    print(f"  → child_a now has its own last block; child_b and parent still share")

    parent.free_all()
    child_a.free_all()
    child_b.free_all()
    print(f"\n  After freeing all: {free_list.num_free} blocks free "
          f"(back to {TOTAL_BLOCKS})")


def demo_fragmentation_sweep():
    print("\n── 3. Fragmentation vs Sequence Length Variance ─────────────────────")
    random.seed(42)

    for std in [0, 8, 32, 64]:
        free_list = FreeList(TOTAL_BLOCKS)
        seqs = {}
        for i in range(20):
            length = max(1, int(random.gauss(32, std)))
            seq = BlockTable(f"s{i}", free_list)
            for t in range(length):
                if not seq.append_token(t):
                    break
            seqs[f"s{i}"] = seq

        stats = measure_fragmentation(free_list, seqs)
        print(f"  std={std:>3}  frag={stats['internal_frag_pct']:>5.1f}%  "
              f"util={stats['utilization_pct']:>5.1f}%  "
              f"blocks={stats['used_blocks']}")

        for s in seqs.values():
            s.free_all()


def main():
    print("E06 — KV Cache Block Allocator (Conceptual Implementation)")
    print(f"BLOCK_SIZE={BLOCK_SIZE}  TOTAL_BLOCKS={TOTAL_BLOCKS}\n")

    demo_basic_allocation()
    demo_copy_on_write()
    demo_fragmentation_sweep()

    print("\n[Article insight]")
    print("  Block table = page table for KV cache.")
    print("  Free list = O(1) allocator; no compaction needed.")
    print("  Ref counting enables prefix sharing with zero copy overhead.")
    print("  CoW is lazy: shared blocks only copied when a sequence diverges.")
    print("  Internal fragmentation bounded by BLOCK_SIZE (at most 15 wasted slots/seq).")
    print("  External fragmentation = 0 by construction (non-contiguous is fine).")


if __name__ == "__main__":
    main()
