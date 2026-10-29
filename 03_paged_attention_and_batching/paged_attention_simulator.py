#!/usr/bin/env python3
"""
PagedAttention Memory Manager Simulator.
Demonstrates physical block allocation, logical-to-physical block table mapping,
and memory fragmentation comparison between Naive Pre-allocation vs. PagedAttention.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import random


BLOCK_SIZE = 16  # Tokens per block/page (standard in vLLM)


@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 0
    is_free: bool = True


class PhysicalMemoryPool:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.blocks = [PhysicalBlock(block_id=i) for i in range(num_blocks)]
        self.free_block_ids = list(range(num_blocks))

    def allocate_block(self) -> Optional[int]:
        if not self.free_block_ids:
            return None
        block_id = self.free_block_ids.pop()
        block = self.blocks[block_id]
        block.is_free = False
        block.ref_count = 1
        return block_id

    def free_block(self, block_id: int):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.ref_count = 0
            block.is_free = True
            self.free_block_ids.append(block_id)

    @property
    def free_blocks_count(self) -> int:
        return len(self.free_block_ids)

    @property
    def total_capacity_tokens(self) -> int:
        return self.num_blocks * BLOCK_SIZE


@dataclass
class SequenceRequest:
    request_id: str
    prompt_len: int
    max_output_tokens: int
    actual_output_tokens: int
    current_tokens: int = 0
    block_table: List[int] = field(default_factory=list)


class PagedAttentionManager:
    def __init__(self, total_memory_blocks: int):
        self.pool = PhysicalMemoryPool(total_memory_blocks)
        self.active_requests: Dict[str, SequenceRequest] = {}

    def allocate_request(self, req: SequenceRequest) -> bool:
        """Allocates initial physical blocks for prompt tokens."""
        req.current_tokens = req.prompt_len
        needed_blocks = math.ceil(req.current_tokens / BLOCK_SIZE)
        
        if self.pool.free_blocks_count < needed_blocks:
            return False  # OOM / Insufficient memory

        for _ in range(needed_blocks):
            block_id = self.pool.allocate_block()
            req.block_table.append(block_id)

        self.active_requests[req.request_id] = req
        return True

    def append_token(self, request_id: str) -> bool:
        """Generates 1 token; allocates a new physical block only when current page fills."""
        req = self.active_requests[request_id]
        req.current_tokens += 1

        # Check if new token crosses into a new block
        if (req.current_tokens - 1) % BLOCK_SIZE == 0 and req.current_tokens > 1:
            new_block_id = self.pool.allocate_block()
            if new_block_id is None:
                return False  # Out of memory during generation
            req.block_table.append(new_block_id)

        return True

    def finish_request(self, request_id: str):
        """Frees all physical blocks mapped in the block table."""
        req = self.active_requests.pop(request_id)
        for block_id in req.block_table:
            self.pool.free_block(block_id)


def run_fragmentation_simulation():
    print("=" * 84)
    print("  PAGEDATTENTION vs. NAIVE CONTIGUOUS PRE-ALLOCATION SIMULATION")
    print("=" * 84)

    random.seed(42)
    max_model_len = 2048
    num_requests = 40
    requests: List[SequenceRequest] = []

    # Generate realistic workload: varying prompt lengths & output lengths
    for i in range(num_requests):
        prompt_len = random.randint(128, 512)
        # Most requests output 30 to 200 tokens (short to medium responses)
        actual_output = random.randint(20, 250)
        requests.append(SequenceRequest(
            request_id=f"req_{i:02d}",
            prompt_len=prompt_len,
            max_output_tokens=max_model_len,
            actual_output_tokens=actual_output
        ))

    # --- 1. Naive Contiguous Allocation (Pre-allocates max_model_len tokens) ---
    naive_total_allocated_tokens = num_requests * max_model_len
    naive_tokens_actually_used = sum(r.prompt_len + r.actual_output_tokens for r in requests)
    naive_wasted_tokens = naive_total_allocated_tokens - naive_tokens_actually_used
    naive_waste_pct = (naive_wasted_tokens / naive_total_allocated_tokens) * 100.0

    # --- 2. PagedAttention Allocation ---
    # Total memory pool sized to hold ~5000 blocks
    manager = PagedAttentionManager(total_memory_blocks=5000)
    for r in requests:
        manager.allocate_request(r)
        for _ in range(r.actual_output_tokens):
            manager.append_token(r.request_id)

    paged_blocks_used = sum(len(r.block_table) for r in requests)
    paged_capacity_allocated_tokens = paged_blocks_used * BLOCK_SIZE
    paged_wasted_tokens = paged_capacity_allocated_tokens - naive_tokens_actually_used
    paged_waste_pct = (paged_wasted_tokens / paged_capacity_allocated_tokens) * 100.0

    print(f"\nWorkload Summary: {num_requests} requests (Prompt: 128-512 tokens, Output: 20-250 tokens)")
    print(f"Total tokens actually used across all requests: {naive_tokens_actually_used:,} tokens\n")

    print(f"{'Metric':<38} | {'Naive Pre-allocation':<20} | {'PagedAttention':<18}")
    print("-" * 82)
    print(f"{'Memory Reserved (Tokens)':<38} | {naive_total_allocated_tokens:<20,d} | {paged_capacity_allocated_tokens:<18,d}")
    print(f"{'Unused Wasted Slots (Tokens)':<38} | {naive_wasted_tokens:<20,d} | {paged_wasted_tokens:<18,d}")
    print(f"{'Memory Waste / Fragmentation (%)':<38} | {naive_waste_pct:<20.2f}% | {paged_waste_pct:<18.2f}%")
    print(f"{'Effective Concurrency Multiplier':<38} | {'1.0x (Baseline)':<20} | {naive_total_allocated_tokens / paged_capacity_allocated_tokens:<18.2f}x")
    print("-" * 82)

    print("\nVisualizing Request 00 Block Table Mapping:")
    sample = requests[0]
    print(f"  • Request ID: {sample.request_id}")
    print(f"  • Prompt Length: {sample.prompt_len} | Output Length: {sample.actual_output_tokens} | Total: {sample.current_tokens} tokens")
    print(f"  • Logical Blocks needed: {len(sample.block_table)}")
    print(f"  • Non-contiguous Physical Block IDs: {sample.block_table}")
    print("=" * 84)


if __name__ == "__main__":
    run_fragmentation_simulation()
