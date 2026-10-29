# Module 04: LLM Quantization (INT8, INT4, FP8, AWQ, & KV Cache Quant)

Because LLM decoding is memory-bandwidth bound, shrinking the numerical precision of weights and KV tensors yields immediate, linear speedups.

---

## 1. Mathematical Foundations of Quantization

Quantization maps continuous 16-bit or 32-bit floating point numbers onto a discrete grid of integers or low-precision floats.

### Symmetric Quantization (Zero-Point $Z = 0$)
Used when the value distribution is roughly zero-centered (common for weights):

$$S = \frac{\max(|x|)}{2^{b-1} - 1}$$
$$X_q = \text{clip}\left(\text{round}\left(\frac{X}{S}\right), -2^{b-1}, 2^{b-1}-1\right)$$
$$\hat{X} = X_q \times S$$

For 8-bit signed integer ($\text{INT8}$, $b=8$): range is $[-127, 127]$.  
For 4-bit signed integer ($\text{INT4}$, $b=4$): range is $[-7, 7]$.

### Asymmetric (Affine) Quantization
Used when the distribution is skewed or strictly positive (common for ReLU/GELU activations or attention weights):

$$S = \frac{x_{\max} - x_{\min}}{2^b - 1}, \quad Z = \text{round}\left(-\frac{x_{\min}}{S}\right)$$
$$X_q = \text{clip}\left(\text{round}\left(\frac{X}{S}\right) + Z, 0, 2^b - 1\right)$$
$$\hat{X} = S \times (X_q - Z)$$

---

## 2. Quantization Granularity

Where do you share the scaling factor $S$?

```
Per-Tensor:           Per-Channel:                   Group-wise (Block-wise):
[  Matrix W  ]        [ Row 0 ] -> Scale S0          [ G0: 128 elements ] -> S0
One single Scale S    [ Row 1 ] -> Scale S1          [ G1: 128 elements ] -> S1
High quantization     [ Row 2 ] -> Scale S2          [ G2: 128 elements ] -> S2
error if outliers     Handles row-level variance     Isolates outliers locally (AWQ/GPTQ)
```

---

## 3. Weight-Only vs. Weight-and-Activation Quantization

| Paradigm | Quantized Tensors | Precision | Best For | Typical Algorithms |
| :--- | :--- | :--- | :--- | :--- |
| **Weight-Only** | Weights only | W4A16, W8A16 | Decode phase / Latency / Consumer GPUs | **AWQ, GPTQ, GGUF** |
| **Weight + Activation** | Weights & Activations | W8A8, FP8, W4A4 | Prefill phase / High batch throughput | **SmoothQuant, FP8 (Hopper/Ada)** |
| **KV Cache Quantization** | Past K & V tensors | KV-FP8, KV-INT4 | Long context (32k-1M) & Large batch sizes | **vLLM FP8 KV, KIVI (INT2/INT4)** |

### Why Weight-Only (W4A16) is so effective for Decode:
In single-batch decode, the latency bottleneck is loading weights from DRAM/HBM. With W4A16:
- 4-bit weights are loaded over the memory bus (4× less data).
- Inside the GPU/Apple Silicon SRAM registers, the weights are dequantized to FP16.
- Standard FP16 Tensor Cores execute the dot product.
- Result: **Near $4\times$ speedup without requiring specialized INT4 matrix multiplication hardware!**

### The Activation Outlier Problem & SmoothQuant
In models $\ge 6.7\text{B}$ parameters, systematic outlier channels emerge in activations with values $100\times$ larger than normal channels.
- Quantizing activations to INT8 naively destroys accuracy.
- **SmoothQuant** observed that weights are easy to quantize, while activations are hard.
- It applies a per-channel smoothing factor $s$ to migrate the quantization difficulty from activations into weights:
$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$
This smooths the activations so both $X$ and $W$ can be cleanly quantized to INT8!
