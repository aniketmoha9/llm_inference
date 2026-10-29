# Module 02: KV Cache Mechanics & Attention Architectures

In transformer-based autoregressive generation, each new token depends on all previous tokens. Without caching, the attention computation becomes disastrously redundant.

---

## 1. Why the KV Cache is Mandatory

At decoding step $t$, the attention equation for token $t$ is:

$$\text{Attention}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}$$

Notice:
- $Q_t$ depends **only** on the current token at position $t$.
- $K_{1:t}$ and $V_{1:t}$ are the Key and Value representations for all tokens from position $1$ to $t$.
- For positions $1 \dots t-1$, the weights $W_K$ and $W_V$ have **already** been applied in previous iterations. Their representations do not change!

### Naive Generation vs. KV Caching

```
Naive Generation (No Cache):
Step 1: Compute Q1, K1, V1                   -> Attn(Q1, K1, V1)
Step 2: Re-compute K1, V1 + compute K2, V2   -> Attn(Q2, K1:2, V1:2)
Step 3: Re-compute K1:2, V1:2 + compute K3   -> Attn(Q3, K1:3, V1:3)
... Total attention FLOPs across S tokens: O(S^3) !

With KV Caching:
Step 1: Compute & STORE [K1], [V1]           -> Attn(Q1, K1, V1)
Step 2: Append [K2], [V2] to Cache           -> Attn(Q2, K1:2, V1:2)
Step 3: Append [K3], [V3] to Cache           -> Attn(Q3, K1:3, V1:3)
... Total attention FLOPs across S tokens: O(S^2) !
```

---

## 2. KV Cache Memory Footprint Formula

While the KV Cache saves enormous amounts of compute, it introduces a severe **memory capacity and memory bandwidth bottleneck**.

The memory required to store the KV cache for a single model across all layers is:

$$\text{KV Cache Size (Bytes)} = 2 \times L \times H_{KV} \times d_{\text{head}} \times S \times B \times P$$

Where:
- $2$: One tensor for Keys, one tensor for Values.
- $L$: Number of transformer layers (`num_layers`).
- $H_{KV}$: Number of Key-Value attention heads (`num_kv_heads`).
- $d_{\text{head}}$: Head dimension (`hidden_dim / num_heads`).
- $S$: Sequence length (prompt + generated tokens).
- $B$: Batch size (concurrent requests).
- $P$: Precision in bytes ($2\text{ bytes for FP16/BF16}$, $1\text{ byte for FP8/INT8}$).

### Concrete Example: LLaMA-2-70B (MHA: 64 KV heads) vs LLaMA-3-70B (GQA: 8 KV heads)

At sequence length $S = 4096$, batch size $B = 16$, FP16 ($P=2$):
- **LLaMA-2-70B (MHA, $H_{KV}=64$, $L=80$, $d=128$)**:
  $$\text{KV Size} = 2 \times 80 \times 64 \times 128 \times 4096 \times 16 \times 2 = 171.8 \text{ GB}$$
  *(Exceeds the memory of two 80GB A100 GPUs just for the KV cache of 16 users!)*
- **LLaMA-3-70B (GQA, $H_{KV}=8$, $L=80$, $d=128$)**:
  $$\text{KV Size} = 2 \times 80 \times 8 \times 128 \times 4096 \times 16 \times 2 = 21.47 \text{ GB}$$
  *($8\times$ memory reduction!)*

---

## 3. Evolution of Attention Architectures

```
 Multi-Head Attention (MHA)       Grouped-Query Attention (GQA)       Multi-Query Attention (MQA)
      (e.g., LLaMA-1)                 (e.g., LLaMA-3, Mistral)                (e.g., Falcon)
  
  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8          Q1 Q2 Q3 Q4  Q5 Q6 Q7 Q8           Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
  │  │  │  │  │  │  │  │            \  /  \  /   \  /  \  /             \  \  \  │  /  /  /  /
  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼             ▼      ▼     ▼      ▼                \  \  \ │ /  /  /  /
  K1 K2 K3 K4 K5 K6 K7 K8              K1    K2    K3    K4                         K1
  V1 V2 V3 V4 V5 V6 V7 V8              V1    V2    V3    V4                         V1
  
  1 Key/Value head per Query       1 Key/Value head per group         1 Key/Value head for ALL
  (Heaviest KV cache)              (Optimal balance: speed + quality) (Smallest KV cache, capacity drop)
```

### DeepSeek Multi-Head Latent Attention (MLA)

In DeepSeek-V2/V3/R1, a groundbreaking architecture called **Multi-Head Latent Attention (MLA)** was introduced:
- Instead of caching high-dimensional multi-head keys and values ($H_{KV} \times d$), MLA compresses them into a low-rank latent vector:
  $$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \quad (d_c \ll H \cdot d_{\text{head}})$$
- Only the compact latent vector $\mathbf{c}_t^{KV}$ and a decoupled RoPE key vector $\mathbf{k}_t^R$ are stored in the KV cache!
- During decoding, Keys and Values can be dynamically projected back, or attention can be computed directly in the latent space by absorbing weights into the Query projection:
  $$Q W^{UK} (\mathbf{c}^{KV})^T = (Q W^{UK}) (\mathbf{c}^{KV})^T$$
- Result: **MLA achieves better modeling performance than MHA while requiring even less KV cache memory than MQA!**
