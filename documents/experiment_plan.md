# 从 SecMoE、CryptoMoE 到 CipherMoE 的实验规划

本文总结 [SecMoE](SecMoE.pdf) 和 [CryptoMoE](CryptoMoE.pdf) 的实验组织方式，
并据此规划 CipherMoE 的下一阶段工作。`idea.md` 只提供最初问题，不作为已经成立的
技术结论。尤其需要重新验证：纯 FHE、非交互、隐藏路由、保留稀疏性和不修改模型，
是否能够同时实现。

## 1. 两篇工作的实验如何组织

### 1.1 SecMoE

SecMoE 是交互式两方安全计算方案，不是纯 FHE。它的实验围绕一个核心主张展开：
Select-Then-Compute 的成本随 expert 数增长得比 Dense 方案慢。
以下总结主要对应论文的 Experiments 部分（PDF 第 6–7 页）。

| 层次 | 实验内容 |
| --- | --- |
| 环境 | 两个节点；每节点 64 vCPU、128 GB RAM；LAN 和 WAN 两种网络 |
| 数值设置 | `Z_(2^64)` 定点数，18 bit 小数精度 |
| 模型 | T5-small 改造的 MoE-Small，以及 Switch-Base |
| 扩展变量 | expert 数量从 4/8 扩展到 128，固定 top-1 和 8 tokens |
| Baseline | Iron、BumbleBee；说明为何不比较依赖 trusted dealer 或大量客户端内存的方法 |
| 系统指标 | 端到端运行时间、通信量；同时报告 LAN/WAN |
| 模型指标 | CoLA、QNLI、RTE 上的准确率/MCC，与浮点明文结果比较 |
| 子协议分析 | 单独比较安全 GeLU，并分析通信量随 expert 数的变化 |

值得借鉴的地方：

1. 先声明威胁模型和不能直接比较的工作。
2. 同时测小模型、较大模型和 expert 数扩展趋势，而不是只给一个点。
3. 同时报系统效率和任务准确率。
4. 明确总成本是否包含离线阶段。SecMoE 选择报告离线加在线总成本，因为其离线成本占主导。

局限：它是交互式 2PC、固定 top-1、输入只有 8 tokens；MoE-Small 还是由 dense
模型人工改造而来。因此其绝对延迟不能直接作为 CipherMoE 的 baseline。

### 1.2 CryptoMoE

CryptoMoE 同样是混合 HE/MPC 的交互式 2PC。它的实验链条比 SecMoE 更完整：
以下总结主要对应第 5 节（PDF 第 7–10 页）及附录中的额外消融。

1. **单个 MoE 层成本**：先测 latency/token 和 communication/token。
2. **端到端模型**：再测完整模型准确率、LAN/WAN 延迟和通信。
3. **参数消融**：改变每个 expert 的 token 容量 `t` 和输入序列长度。
4. **组件消融**：依次加入 balanced routing、confidence-aware selection 和 batch MatMul。
5. **Latency breakdown**：拆分 router、dispatch、expert linear、SiLU、combine、attention 等阶段。
6. **扩展性和限制**：在更大模型上测明文准确率，并明确私有推理因内存不足无法完成。

具体设置包括：

| 项目 | 设置 |
| --- | --- |
| 框架 | SecretFlow-SPU，无 trusted third party |
| 硬件 | Intel Xeon Platinum 8468，48 cores |
| 网络 | LAN：3 Gbps/0.2 ms；WAN：400 Mbps/40 ms |
| 真实模型 | DeepSeekMoE、OLMoE、QWenMoE |
| 任务 | 8 个 zero-shot commonsense reasoning 数据集 |
| Baseline | 公开路由 Insecure、计算所有 experts 的 Dense、基于 CipherPrune 的 dispatch/combine |
| 主参数 | expert 容量 `t`、序列长度、batch size |
| 主指标 | accuracy、amortized latency/token、communication/token |

最值得借鉴的是它的证据顺序：**primitive/单层 → 完整模型 → 消融 → breakdown →
扩展性和失败边界**。它还展示了 kernel 加速不等于端到端加速：优化 expert MatMul 后，
WAN 下的 top-k 和通信可能成为新瓶颈。

但 CryptoMoE 通过固定每个 expert 的容量并丢弃超额 token 来隐藏负载，这会改变原模型；
其计算、通信和准确率权衡与非交互纯 FHE 不同，也不能直接作为同口径性能 baseline。

## 2. 对 CipherMoE 的直接启示

我们的实验不能只回答“`Enc(W_e x)` 能否算对”，而应分别回答：

1. **结构可行性**：真实 MoE 权重是否能被小 `r` 的 basis 准确表示？
2. **密码学收益**：在相同安全参数和打包下，Basis 是否比 Dense/Select-Weight 快？
3. **系统收益**：加入 router、top-k、非线性、多个 token 后，优势是否仍然存在？
4. **模型质量**：basis 近似是否影响 perplexity 和下游任务准确率？
5. **资源边界**：时间、密文内存、明文模型内存、evaluation keys 和通信分别是多少？

SecMoE/CryptoMoE 可以作为实验组织参考和相关工作，但不能与 CipherMoE 做绝对时间的
横向比较。公平的主 baseline 必须在相同的纯 FHE、非交互威胁模型下实现。

## 3. 当前 Basis 还缺什么

当前实现只完成：单 token、单个方阵线性层、外部已经提供 encrypted one-hot selector：

```text
Enc(x), Enc(one-hot route) -> Enc(W_e x)
```

要推理一个完整 MoE，至少还需要以下设施。

### 3.1 真实模型的 basis 预处理

- 从真实 checkpoint 读取每层每个 expert 的 `gate_proj/up_proj/down_proj`。
- 支持矩形矩阵，而不是只支持 `d × d` 方阵。
- 每层、每个 projection 独立求 basis 和系数，不能默认全模型共用一个 `r`。
- 比较全局 basis、expert 分组 basis、`shared base + low-rank delta`。
- 用 calibration activations 选择 `r`，同时记录权重误差、线性输出误差和任务准确率。

这是最先要做的实验。若真实模型只有接近 `N` 的 `r` 才能保持精度，当前 Basis 就没有
足够的计算优势。小 `r` 本质上也属于模型压缩，和“完全不修改原模型”的目标存在张力，
必须明确报告近似，而不能把它当作无损密码学变换。

### 3.2 完整私有 router

- CKKS 计算 router logits。
- 在不与客户端交互的情况下完成 top-k/argmax。
- 完成 CKKS 与 FHEW/TFHE 的转换，或者找到 CKKS-only 控制方案。
- 将 encrypted ID 转换成 Basis 所需的 selector/系数表示。
- 将 router、scheme switching 和 bootstrapping 成本计入端到端时间。

目前矩阵实验从 encrypted one-hot 开始，尚未覆盖这些成本。

### 3.3 完整 expert FFN

真实 expert 通常类似：

```text
down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

需要实现三个矩形线性层、SiLU/SwiGLU 近似、逐元素乘法和 scale/level 管理。top-1
可以沿用同一个隐藏 route；top-k 不能在非线性前简单合并多个 expert 权重，通常需要
分别计算 `k` 个 expert 流并在输出端按 router score 聚合，成本大约随 `k` 增长。

### 3.4 多 token 打包和布局

- 当前只有一个 token，且 selector 被广播到全部 slots。
- 完整 prefill/decode 中不同 token 会选择不同 expert。
- 需要联合设计 token、hidden dimension、basis rank 和 top-k 的 slot layout。
- 每层输出布局必须能直接进入下一层，避免昂贵的重复重打包。
- 应实现 batch MatMul/BSGS/hoisting；CryptoMoE 的实验表明 expert linear 往往是主要瓶颈。

### 3.5 深度、bootstrapping 和完整 Transformer 接口

- 计算整个 MoE 层所需的 multiplicative depth 和 modulus chain。
- 确定在哪些位置 bootstrap，以及 bootstrap 后误差和时间。
- 支持 residual、RMSNorm/LayerNorm，并与 attention 层布局衔接。
- 区分三个里程碑：完整 MoE 层、完整 Transformer block、完整模型，不能混称端到端。

### 3.6 测量和复现设施

- 分阶段计时：router、top-k、coefficient select、basis MatMul、activation、combine、bootstrap。
- 分开报告 setup、model encoding、input encryption、online evaluation 和 decryption。
- 记录 ciphertext/plaintext/evaluation-key 峰值内存，而不只记录进程 RSS。
- 记录 rotations、ct-pt、ct-ct、PBS、层数、level、scale、ring dimension 和线程数。
- 每项多次重复，报告 median、分位数或标准差。
- 保存完整配置、原始结果、失败/OOM 点和复现命令。

## 4. 建议的实验组织

### 阶段 A：真实权重结构分析，不运行 FHE

选择至少两个公开 MoE checkpoint，例如 OLMoE 和 DeepSeekMoE/QWenMoE。对每层、每个
projection 测试 `r=1...N`：

- 跨 expert singular-value energy；
- 权重重建误差；
- calibration activation 上的输出误差；
- 全局、分组和 shared-base-plus-delta 三种分解；
- 每层满足 1%/5% 输出误差所需的最小 `r`。

输出应该是 layer × projection × rank 的误差曲线，而不是只报告平均值。

### 阶段 B：Select-and-Apply primitive

在同一 OpenFHE 参数下比较：

1. Public routing；
2. Dense private routing；
3. Select-Weight full；
4. Select-Weight stream；
5. Basis；
6. FHEW fused LUT（单独作为有限精度标量证据）。

扫描变量：`N`、`d_in/d_out`、`r`、token 数、top-k、packing、rotation/basis block size。
输出 latency、操作数、HE 误差、basis 近似误差、峰值内存和失败点。

### 阶段 C：完整 MoE 层

依次加入：

1. router linear；
2. encrypted top-1；
3. 三个 expert projections；
4. SwiGLU；
5. top-k 和 score combine；
6. 多 token packing。

每加入一项都做组件消融和 latency breakdown，防止 primitive 加速被 router、PBS 或
activation 完全掩盖。

### 阶段 D：模型级评估

- 先在明文中验证 basis 版本的 perplexity/zero-shot accuracy。
- 在可运行的小模型上完成纯 FHE 端到端或至少一个完整 Transformer block。
- 对大模型若无法运行完整 FHE，诚实拆成“全模型明文质量 + 同尺寸密码学 kernel”，
  不把两者拼成虚假的端到端数字。
- 报告 latency/token、throughput、请求/响应字节数、总内存和任务准确率。

推荐采用 CryptoMoE 的展示顺序：

```text
单 primitive -> 单 MoE 层 -> 端到端/完整 block
             -> 参数消融 -> 组件消融 -> latency breakdown -> 扩展性/限制
```

## 5. Basis 是否需要像 Select-Weight 一样流式化

### 结论

**不需要照搬 Select-Weight 的“加密权重分块”，因为 Basis 根本不生成完整的
`Enc(W_e)`。当前 Basis 已经按 basis rank 流式执行。**

当前循环每次只保留：

1. 一个选中的 encrypted coefficient；
2. 一个 `B_r x`；
3. 一个门控后的 basis 输出；
4. 最终累加结果。

完成一个 `r` 后这些临时量即可释放，所以不会同时物化 `r` 个加密权重或 `r` 个输出。

### 真正需要优化的内存

当前实现会提前缓存全部 `d` 个输入旋转，并预编码全部 `r × d` 个 basis 对角线：

- `d` 个 rotated ciphertexts：在线密文内存，真实维度下可能很大；
- `r × d` 个 encoded plaintexts：不是加密中间权重，但模型内存也可能很大。

因此 Basis 应增加的是下面两类流式化：

1. **明文 basis streaming**：按 layer/rank block 从磁盘加载或编码 basis，降低模型常驻内存。
2. **rotation/diagonal tiling**：一次只缓存 `b` 个旋转，同时维护少量 basis partial sums。

一种不增加旋转数量的布局是：先保存 `r` 个 coefficient 和 `r` 个 basis partial output，
然后按对角线块生成旋转。峰值密文数量可从约 `O(d)` 变成 `O(b+r)`；代价是同时维护
`r` 个 partial outputs。若还要把 rank 分块，峰值可继续下降，但不同 rank block 可能需要
重算输入旋转，形成明确的内存—时间权衡。

需要增加消融：

| 方案 | 主要峰值 | 旋转代价 | 用途 |
| --- | --- | --- | --- |
| 当前：缓存全部 rotations | `O(d)` 密文 | `d-1` | 最快基准 |
| diagonal tile，保存全部 rank sums | `O(b+r)` 密文 | `d-1` | `r << d` 时优先 |
| rank + diagonal 双分块 | `O(b+r_block)` 密文 | 可能重算 | 极低内存场景 |
| BSGS/hoisting | 依布局而定 | 显著减少/摊销 | 真实大矩阵必须评估 |

所以，对问题 A 的补充回答是：Select-Weight streaming 解决的是选中加密权重的物化；
Basis 已经消除了这类对象，但仍需针对 rotations、basis plaintext 和多 token partial sums
设计独立的 streaming/tiling。两者优化的对象不同。

## 6. 下一步优先级

1. 实现真实 checkpoint 的 layer-wise/projection-wise basis 分析脚本。
2. 给 Basis 增加准确的 live-cipher/live-plaintext 内存统计和 diagonal tiling 消融。
3. 支持矩形矩阵、多个 token 和 BSGS/batch packing。
4. 实现完整 top-1 MoE 层，再扩展 top-k；不要直接跳到完整模型。
5. 只有确认真实模型存在足够小的有效 `r` 后，才投入完整纯 FHE 端到端实现。
