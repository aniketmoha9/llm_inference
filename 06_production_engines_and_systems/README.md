# Module 06: Production Inference Engines & Distributed Systems

Building a production-grade LLM inference service requires synthesizing kernels, memory managers, schedulers, and distributed networks into a unified engine.

---

## 1. Production Engine Landscape & Trade-offs

| Engine | Primary Architecture | Strengths | Best Use Case |
| :--- | :--- | :--- | :--- |
| **vLLM** | PagedAttention + Continuous Batching | General-purpose, huge model support, high community adoption, robust OpenAI-compatible API | General API serving, High-throughput cloud inference |
| **SGLang** | RadixAttention (Prefix Caching) + Fast Runtime | Blazing fast prefix reuse, structured JSON decoding, fastest for agents/multi-turn | Agentic workflows, long multi-turn chats, structured output |
| **TensorRT-LLM** | Fused C++ CUDA kernels + In-Flight Batching | Absolute maximum throughput on NVIDIA GPUs, FP8/FP4 optimization | Enterprise NVIDIA Hopper/Blackwell clusters |
| **llama.cpp** | CPU/Metal C++ pure implementation, GGUF | Zero-dependency, CPU/Metal efficiency, tiny footprint | Local Mac, edge devices, local development |
| **MLX (Apple)** | Unified memory native framework | Direct Apple Silicon Metal utilization, zero-copy memory transfers | Apple M-series local development & fine-tuning |

---

## 2. RadixAttention & Automatic Prefix Caching (SGLang)

In agentic systems, few-shot prompting, and multi-turn chat, prompts share huge common prefixes:
- System prompt (e.g., 2,000 tokens of persona & instructions)
- Tool definitions & OpenAPI schemas (e.g., 3,000 tokens)
- Conversation history

### The Radix Tree Cache
Instead of discarding the KV cache when a request finishes:
- SGLang and vLLM retain KV pages in a **Radix Tree** (prefix tree).
- When a new request arrives, the engine matches the prompt tokens against the tree.
- Matched prefix blocks are **reused directly** with zero recomputation!

```
                    Root (Common System Prompt)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     User A: "Turn 1"                      User B: "Turn 1"
            │                                     │
            ▼                                     ▼
     Bot A: "Reply 1"                      Bot B: "Reply 1"
```

**Impact**: Cuts Time-To-First-Token (TTFT) from hundreds of milliseconds to under **5 ms** for cached prefixes!

---

## 3. The Future: Prefill-Decode Disaggregation (PD Disaggregation)

In monolithic serving, a single GPU runs both Prefill (compute-bound) and Decode (memory-bandwidth bound):

```
Monolithic Node (Prefill + Decode Colliding):
┌────────────────────────────────────────────────────────┐
│ GPU: 100% Compute Burst (Prefill 4k tok) ------------> │ (Decodes freeze! TPOT spikes)
│ GPU: Memory Bandwidth Bottleneck (Decode 32 users) ---> │ (Tensor Cores sit 90% idle)
└────────────────────────────────────────────────────────┘
```

### Disaggregated Architecture (e.g., Splitwise, DistServe, Mooncake):

```
  [ Incoming Requests ]
           │
           ▼
 ┌──────────────────┐           High-Speed RDMA          ┌──────────────────┐
 │  Prefill Nodes   │ ─────────────────────────────────> │   Decode Nodes   │
 │ • High TFLOPs    │          KV Cache Transfer         │ • High Bandwidth │
 │ • Chunked GEMM   │                                    │ • PagedAttention │
 │ • Large Batches  │                                    │ • Stable TPOT    │
 └──────────────────┘                                    └──────────────────┘
```

1. **Dedicated Prefill Pool**: Optimized for massive compute (e.g., 8-GPU tensor parallel, FP8 GEMM). Quickly chews through long context prompts.
2. **KV Cache Transfer**: Transmits generated KV cache across RDMA / InfiniBand network directly into the Decode pool's memory.
3. **Dedicated Decode Pool**: Optimized for steady, low-latency token generation with 0% prefill interference.

**Results**:
- Eliminates TTFT and TPOT variance (guaranteed SLA).
- Yields up to $2\times$ higher overall system throughput per dollar of hardware.
