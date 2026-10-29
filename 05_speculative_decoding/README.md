# Module 05: Speculative Decoding & Fast Verification

Speculative Decoding (Leviathan et al., 2023; Chen et al., 2023) breaks the sequential bottleneck of autoregressive generation **without any loss in output quality**.

---

## 1. Why Speculative Decoding Works (The Roofline Connection)

In Module 01, we established that single-token decode is severely memory-bandwidth bound:
- Loading a 70B model takes $\approx 20\text{ ms}$ on an H100 GPU.
- Whether you process **1 token** or **5 tokens** in that forward pass, the time to load the model weights across the bus is virtually identical!

**Core Idea**:
1. Use an ultra-fast, small **Draft Model** (e.g., 1B parameter model) to quickly predict $K$ candidate tokens.
2. Run all $K$ tokens through the large **Target Model** (e.g., 70B model) in **a single forward pass**.
3. Verify the candidates using rejection sampling.

```
Step 1: Draft Model proposes K tokens:
        [x1, x2, x3, x4]  (Very fast, ~2 ms)

Step 2: Target Model verifies ALL K tokens in 1 parallel forward pass:
        P_target(x1), P_target(x2), P_target(x3), P_target(x4)  (~20 ms)

Step 3: Verification (Rejection Sampling):
        x1: ACCEPT (p_target >= p_draft)
        x2: ACCEPT (p_target >= p_draft)
        x3: REJECT! -> Resample x3 from adjusted target distribution.
        Discard x4.

Result: Generated 3 target tokens in 1 target model forward pass!
Effective latency per token cut by 2-3x!
```

---

## 2. Mathematically Lossless Rejection Sampling

Speculative decoding is **mathematically proven** to preserve the exact probability distribution of the target model.

For each speculative token $\hat{x}$:

1. Calculate acceptance probability:
   $$\beta = \min\left(1, \frac{P_{\text{target}}(\hat{x})}{P_{\text{draft}}(\hat{x})}\right)$$
2. Draw $r \sim \text{Uniform}(0, 1)$.
   - If $r \le \beta$: **Accept** $\hat{x}$.
   - If $r > \beta$: **Reject** $\hat{x}$ and resample a replacement token from the positive residual distribution:
     $$P_{\text{resample}}(x) = \frac{\max(0, P_{\text{target}}(x) - P_{\text{draft}}(x))}{\sum_y \max(0, P_{\text{target}}(y) - P_{\text{draft}}(y))}$$
   - Any subsequent speculative tokens $\hat{x}_{k+1 \dots K}$ are discarded.

---

## 3. Advanced Speculative Variants

- **Self-Speculative Decoding (Medusa / EAGLE)**: Instead of maintaining a separate draft model, attach small prediction heads to the target model's upper layers or use tree-based attention masks to verify multiple speculation paths simultaneously.
- **Prompt Lookup / Lookahead Decoding**: For code editing or document summarization, tokens are often copied directly from the prompt. N-gram matching can draft candidates with zero draft model compute!
