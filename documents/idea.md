# Select-and-Apply: Pure-FHE Sparse MoE Inference

## 1. Motivation

Sparse MoE 对每个 token 只激活少量 `k` 个 expert，而不是计算全部 `N` 个 expert。

在隐私推理中，expert routing 由用户输入决定，因此 selected expert ID 本身也是敏感信息。

现有思路主要有两个问题：

1. **Dense-FHE**：计算所有 `N` 个 experts，再秘密选择结果。
   - routing privacy 好；
   - 但失去 MoE sparsity，计算复杂度接近 `N × Expert`.

2. **Select-Then-Compute / Select-Weight**：
   - 先秘密得到 selected expert weight `Enc(W_e)`；
   - 再计算 `Enc(W_e) × Enc(x)`；
   - 避免计算所有 experts，但引入昂贵的 ciphertext-ciphertext matrix multiplication。

我们的目标是：

> 在不泄露 routing、不修改原始 MoE、不需要客户端在线交互的情况下，只计算被选中的 `k` 个 expert，同时避免显式生成 `Enc(W_e)`。

---

## 2. Core Idea: Select-and-Apply

定义一个新的 FHE primitive：

\[
\operatorname{SelectApply}
(
Enc(e), Enc(x), \{W_i\}_{i=1}^{N}
)
\rightarrow
Enc(W_e x)
\]

其中：

- `e`：selected expert ID；
- `Enc(e)`：加密的 expert ID；
- `Enc(x)`：加密 activation；
- `W_i`：服务器持有的 plaintext expert weights。

传统方案：

\[
Enc(e)
\rightarrow
Enc(W_e)
\rightarrow
Enc(W_e) \times Enc(x)
\]

Select-and-Apply：

\[
Enc(e), Enc(x), \{W_i\}
\rightarrow
Enc(W_e x)
\]

核心目标：

> 不显式 materialize 整个 encrypted selected weight，而是让 encrypted expert ID 直接控制 plaintext expert operator 对 encrypted activation 的应用。

---

## 3. Threat Model

- Client 持有私有输入。
- Server 持有 plaintext MoE model。
- Server 为 honest-but-curious。
- Server 不应知道：
  - 用户输入；
  - router score；
  - selected expert ID。
- Client 不应获得模型权重。
- 推理过程中 Client 不参与中间协议。

通信形式：

\[
Client \rightarrow Enc(input)
\rightarrow Server
\rightarrow Enc(output)
\rightarrow Client
\]

即 non-interactive FHE inference。

---

## 4. FHE Design

初步考虑 mixed-scheme FHE：

\[
\boxed{CKKS + TFHE/FHEW}
\]

### CKKS

负责大规模数值计算：

- Router linear layer；
- Expert linear layers；
- packed matrix-vector / matrix-matrix operations。

### TFHE/FHEW

负责 encrypted control：

- comparison；
- Top-k / Argmax；
- encrypted expert index；
- Select-and-Apply 中的秘密选择。
