# Module 03: PagedAttention, Continuous Batching, & Chunked Prefill

In high-throughput multi-user LLM serving, raw model speed is secondary to **memory management and scheduling efficiency**. This module breaks down the innovations behind **vLLM**, **Orca**, and **TensorRT-LLM**.

---

## 1. The Memory Fragmentation Crisis (Why Early Systems Blew Up)

Before **PagedAttention** (Kwon et al., SOSP 2023), inference systems allocated contiguous GPU memory for each request based on its maximum possible sequence length (`max_model_len`, e.g., 4096 tokens):

```
Pre-PagedAttention Allocation:
Request A (allocates 4096 tokens): [█████ 20 tokens generated ... 4076 RESERVED & WASTED MEMORY]
Request B (allocates 4096 tokens): [██████████ 150 tokens ... 3946 RESERVED & WASTED MEMORY]
```

This caused two catastrophic forms of waste:
1. **Internal Fragmentation**: Over 60–80% of allocated KV cache was reserved for future tokens that were never generated.
2. **External Fragmentation**: Free memory was scattered in non-contiguous slices, blocking new requests from launching even when total free memory was large.

---

## 2. PagedAttention: OS Virtual Memory Applied to LLMs

PagedAttention solves fragmentation by borrowing the classic OS paging concept:

```
Logical Blocks (Request Sequence)        Block Table (Mapping)        Physical Blocks (GPU Memory)
┌──────────────┐                         ┌─────────────┬────────────┐ ┌──────────────┐
│ Block 0: 0-15│ ──────────────────────> │ Logical 0   │ Physical 7 │ │ Block 1 (Req B)
├──────────────┤                         ├─────────────┼────────────┤ ├──────────────┤
│ Block 1:16-31│ ──────────────────────> │ Logical 1   │ Physical 2 │ │ Block 2 (Req A)
├──────────────┤                         ├─────────────┼────────────┤ ├──────────────┤
│ Block 2:32-47│ ──────────────────────> │ Logical 2   │ Physical 9 │ │ Block 7 (Req A)
└──────────────┘                         └─────────────┴────────────┘ ├──────────────┤
                                                                      │ Block 9 (Req A)
                                                                      └──────────────┘
```

### Key Properties:
- **Fixed-size Block Pages**: KV tensors are split into pages of $K$ tokens (typically 16 or 32 tokens).
- **Physical Memory Pool**: All KV cache is pre-allocated as a single global pool of physical blocks.
- **On-Demand Allocation**: A new physical block is allocated only when the current page is completely full.
- **Near-Zero Fragmentation**: Only the very last block of a request may have unused slots ($< 4\%$ average memory waste).
- **Copy-on-Write (CoW)**: When branching prompts (e.g., beam search, parallel sampling, multi-turn chat), multiple requests point to the exact same physical blocks until one mutates.

---

## 3. Continuous Batching (Iteration-Level Scheduling)

Traditional batching (Static Batching) schedules a batch of requests together:

```
Static Batching (Coarse-Grained):
Req 1 (30 tok):  [====================] (Done, waits idle for Req 3)
Req 2 (50 tok):  [================================] (Done, waits idle)
Req 3 (100 tok): [================================================================]
GPU Utilization: 43% (Heavy padding waste)

Continuous Batching (Iteration-Level):
Step 0-30:   [Req 1, Req 2, Req 3]
Step 31:     Req 1 finishes! -> Evict Req 1, Immediately insert Req 4!
Step 31-50:  [Req 4, Req 2, Req 3]
Step 51:     Req 2 finishes! -> Evict Req 2, Immediately insert Req 5!
GPU Utilization: >95%
```

In continuous batching, the scheduler iterates step-by-step:
1. Complete 1 decode step for all active requests in the running batch.
2. Check for end-of-sequence (`<EOS>`) or max token limit.
3. Immediately release finished requests' KV blocks to the pool.
4. If free blocks permit, admit waiting requests from the waiting queue.

---

## 4. Chunked Prefill: Solving the "Prefill Bubble"

When a user submits a long prompt (e.g., an 8,000-token PDF):
- Prefilling 8k tokens takes hundreds of milliseconds of 100% GPU compute.
- During this time, all concurrent decode requests **freeze**, causing severe spikes in Time-Per-Output-Token (TPOT / Jitter).

**Chunked Prefill** (e.g., Sarathi-Serve, vLLM Chunked Prefill):
- Breaks the 8,000-token prompt into chunks (e.g., 512 tokens each).
- Pairs a 512-token prefill chunk with ongoing decode tokens in the same forward pass!
- Flattens GPU utilization and eliminates TPOT latency spikes.
