# Exact Select-and-Apply：实现与验证计划

## 1. 目标、语义和验收条件

最终接口是：

```text
SelectApply(Enc(e), Enc(x), {W_i}_{i=1}^N) -> Enc(W_e x)
```

- `e` 是秘密 expert ID，`x` 是秘密 activation；二者对服务器隐藏。
- `W_i` 是任意 pretrained MoE 的原始 plaintext expert weights。
- 不重新训练模型，不要求不同 experts 具有特殊结构。
- 不显式 materialize 完整 `Enc(W_e)`。
- 在线求值不执行 `N` 次完整 expert MatMul，也不能退化成
  `Enc(e) -> Enc(one-hot) -> Dense CKKS`。
- 输出选择语义 exact：应用的是原始 `W_e`，不允许用跨-expert 低秩近似替代它。

“exact”不表示真实数算术 bit-exact。定点编码/量化必须定义输入范围和溢出语义；CKKS
本身是近似算术。实验要把 HE 数值误差与模型结构近似误差分开。主方案的结构近似误差为 0。

服务器不能依据秘密 `e` 做可观察的普通分支或内存访问，所以“只执行被选 expert”的可操作
定义是：固定且数据无关的求值轨迹中，在线成本不包含 `N` 个完整 MatMul。模型的
`N × d_out × d_in` 个明文参数仍必须以某种形式存在；若把它们编码进 LUT、evaluation key
或预处理对象，必须把大小和生成成本报告出来。任意权重不可能仅凭“一个 PBS”免费消除这部分
信息量；时间、存储和 setup 之间的转移正是要验证的问题。

## 2. 两条严格分开的 proposal

### 2.1 `proposal/FHEW_SelectApply/`：exact 主线

用 FHEW/TFHE programmable bootstrapping、CMux、vertical packing 或 scheme switching，
让 encrypted small index 控制 plaintext linear operator。当前只有 1-bit expert ID、1-bit
scalar activation 的联合 LUT，它只证明“选择与标量应用可以融合”，尚未证明矩阵方案。

主线要依次解决：multi-bit ID、有限精度 scalar、packed vector、多输出复用、tiled matrix、
CKKS/FHEW 混合布局和完整 MoE 层。每一步都必须给出 PBS/CMux/rotation/ct-pt/ct-ct 数量，
不能只给 wall-clock latency。

### 2.2 `proposal/LowRank_Structured/`：结构化近似基线

它假设：

```text
W_e ≈ Σ_{j=1}^r c[e,j] B_j
```

`r<N` 时通常是近似方法，只有真实 weights 恰好位于该子空间才 exact。它可以作为
structured-model baseline 或可选优化，但不是 arbitrary pretrained MoE 的主方案。即使阶段 A
发现真实模型存在小 `r`，论文也只能把它报告为额外结构带来的优化；若低误差下普遍
`r≈N`，停止在它上面投入端到端工程。

## 3. 先检索开源实现，再补功能

每个功能缺口开始编码前执行同一流程：

1. 搜索当前依赖的源码、官方 examples/tests 和 API 文档。
2. 再搜索其他主流开源 FHE 库和论文作者官方仓库，不以二手博客为实现依据。
3. 在实验记录中固定 repository、commit/tag、license、目标 API、参数限制和复用边界。
4. 先写最小适配/正确性测试；只有没有合适原语或现有实现不能满足接口时才自研。
5. 复用代码时保留许可证和出处；对实验性 API 加版本锁定和回归测试。

2026-09-26 已完成的第一轮 source audit：

| 来源 | 固定版本 | 可复用设施 | 当前边界 |
| --- | --- | --- | --- |
| [OpenFHE](https://github.com/openfheorg/openfhe-development) | v1.5.1 / `1306d14f8c26bb6150d3e6ad54f28dfe1007689e`，BSD-2-Clause；已在 `thirdparty/openfhe/src/` | BinFHE `EvalFunc/EvalDecomp/EvalSign`；CKKS↔FHEW switching；RLWE/CKKS functional bootstrap | scalar LUT 不自动解决大矩阵；参数和 plaintext domain 有限制 |
| [OpenFHE scheme switching example](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/scheme-switching.cpp) | 同上 | `EvalCKKStoFHEW`、`EvalFHEWtoCKKS` 及中间 LUT | slot 转换成本必须计入在线路径 |
| [OpenFHE MultiValueBootstrapping](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/CKKS_FUNCTIONAL_BOOTSTRAPING.md) | 同上，experimental | 同一 input 上多个 LUT 共用 `EvalMVBPrecompute` 中间量 | RLWE coefficient encoding；当前 LUT input 最多约 14 bits；不能先假定适合 MoE |
| [TFHE-rs](https://github.com/zama-ai/tfhe-rs) | surveyed `079f46d08ffb8763de54b7794e850ddbd98fe63b`，BSD-3-Clause-Clear | PBS、WOPBS/vertical packing、radix integer、多输出 LUT、CMux，含 CPU/GPU 后端 | 与现有 OpenFHE CKKS 路径的集成成本高；先做独立原型比较 |
| [CryptoMoE](https://github.com/PKU-SEC-Lab/CryptoMoE) | `d3074967c86110ddda3c7e03f29106aff7e64a30`，Apache-2.0 | balanced-routing 模型改造和质量评测任务 | ciphertext/SPU implementation 仍为 TODO，不能复用其密码学 kernel |

后续 source audit 优先检查：TFHE-rs 的 `apply_many_lookup_table`、WOPBS
`circuit_bootstrapping_vertical_packing` 和 scalar CMux；OpenFHE 的 `EvalMVB*` 是否能让多个
输出函数真正复用选择；以及 private function evaluation / oblivious array access 文献中是否有
适合 public model、secret index 的构造。任何新原语先形成最小 benchmark 和复杂度表。

### 本轮研究优先级：P0 encrypted selector amortization

Stage C/D 以前的原型仍把 `(e,x)` 联合编码，且在 `N=2` 等小域内成立；
它们没有回答真实矩阵上的 control amortization。先单独做
`MultiOutputSelect(Enc(e), {V_i}) -> Enc(V_e)`，其中公开向量值可独立变化、
不依赖跨 expert 的低秩结构，不输入 activation，不把扩展 joint LUT 当成主线。
优先扫描 `m=1,4,8,16,32,64,128,256,512` 和小/大 `N`；记录
`T(m), T(m)/m`、实际 blind rotation、key switching、sample extraction、
公开 LUT、evaluation key、输入/输出密文字节数和正确性。
同一安全参数下把单 accumulator 能容纳的输出数与超过后必须额外执行的
PBS 分开计费。完整向量可在 P0 作为诊断输出，但不代表最终方案允许
物化完整 `Enc(W_e)`：P1/P2 必须证明 compact control 能直接控制
packed arithmetic，且不退化成 one-hot + Dense CKKS 或逐元素 PBS。
如果大 `N` 下 `T(m)` 基本随 `m` 线性增长，就按负面结果停止该
scalar-LUT 路线的模型级扩展，转向 RLWE coefficient packing、
scheme switching 或其他可复用的 encrypted-control 表示。

2026-09-27 已完成固定 TFHE-rs 参数下的 P0 `Enc(e)` 独立实验：
`N=2/8/64`、`m=1..512` 的九点扫描；`N=64,m=512` 实测
512 次 blind rotation / 101.84 秒，只有一个输出能共享单次旋转，
因此 **shortint many-LUT 在该参数下对 64 experts 的模型级扩展 no-go**。
原始结果和参数限制见 `results/p0_selector/report.md`。这不排除
其他 packed RLWE/GLWE 或 CKKS 兼容的 encrypted-control 构造；
在验证新表示前暂缓 P1/P2 和完整 MoE 集成。

## 4. 分阶段实现与验证

每个阶段结束都生成独立的 `results/stage_*/report.md`，报告通过项、失败项、原始配置、复现
命令和下一阶段 go/no-go；未经阶段验收不把 microbenchmark 称为完整 Select-and-Apply。

### 阶段 A：真实 checkpoint 的结构检验（辅助线）

目标是证伪/量化 LowRank 假设，不决定 exact 主线是否继续。

对 OLMoE、DeepSeekMoE、QwenMoE 的每个 MoE layer 和
`gate_proj/up_proj/down_proj`，把 experts 展平后做跨-expert SVD，并报告：

- 完整 rank 曲线和 singular-value energy；
- 权重 reconstruction relative Frobenius error；
- 真实 calibration activation 上的 relative output error；
- 达到 1% 和 0.1% 误差所需的 `r`；
- `r/N` 按 model、layer、projection 的分布；
- global、grouped、shared-base-plus-delta 的结构上限。

验收：全部权重元素参与最终 SVD；下载 revision、shard 数和总字节精确匹配 registry；不能
用抽样结果发布最终结论。若普遍 `r≈N`，LowRank 只保留历史 baseline；若 `r<<N`，也只进入
optional optimization，不改变 exact 主线定义。

### 阶段 B：multi-bit exact scalar FHEW

从任意 public scalar table `w[e]` 开始，实现：

```text
Enc(e), Enc(x), {w_i} -> Enc(w_e * x)
```

- `N=2,4,8,16,32,64`，ID 位宽为 `ceil(log2 N)`；处理非法 ID。
- 明确定点 domain、signed encoding、乘法后的 modulus/overflow。
- 比较 fused joint LUT、ID decomposition + CMux tree、TFHE-rs WOPBS/vertical packing。
- 统计 LUT/bootstrapping-key 大小、setup、PBS 数、在线 latency、峰值内存和错误率。
- 用穷举小 domain 与明文 reference 比较；主表采用至少 128-bit 安全参数。

Go/no-go：若 LUT domain 或 evaluation-key storage 随 `N × |x-domain|` 已不可接受，就先确定
矩阵扩展的资源下界，不能直接声称标量方案可扩展。

### 阶段 C：vector output 与一次选择复用

固定同一个 `Enc(e)`/`Enc(x)`，输出多个函数 `f_k(e,x)`，回答能否复用一次 encrypted
selection，而不是每个 weight/output 重做一次 PBS。

- 先做 `m=1,2,4,...` 个 scalar outputs 的 scaling curve。
- 对比独立 `EvalFunc`、OpenFHE `EvalMVB*`、TFHE-rs many-LUT/vertical packing。
- 分开记录共享 precompute 与每增加一个 output 的边际成本。
- 检查输出是多个 LWE、一个 packed RLWE，还是可直接进入 CKKS 的 slots。

Go/no-go：需要证明边际成本明显低于“每个输出一次完整 PBS”；否则真实 `d_out` 下会直接
失去可行性。MultiValueBootstrapping 是重点候选，但必须实测，不能从 API 名称推断收益。

### 阶段 D：tiled / packed matrix operator

扩展为有限精度矩阵：

```text
Enc(e), Enc(x_vector), {W_i} -> Enc(W_e x_vector)
```

- 支持矩形 `d_out × d_in` 和 tile，不只支持小方阵。
- 扫描 ID/activation/output packing；明确每个 tile 中 model data 的编码位置。
- 比较 output-stationary、input-stationary、diagonal/BSGS 等布局。
- 证明执行中不出现完整 `Enc(W_e)`，也不暗中计算 `N` 个完整 MatMul。
- 报告复杂度对 `N,d_in,d_out,bitwidth,tile` 的依赖，以及 model/LUT/eval-key storage。

这里必须先回答一个核心问题：arbitrary `W_i` 的公开参数怎样被固定电路访问。若每个输出元素
仍需要对 `N` 个 public weights 做线性选择，或为每个 weight 执行 PBS，方案只是换了表达形式，
没有解决主问题。

### 阶段 E：CKKS/FHEW 混合路径

目标是 CKKS 承担 packed arithmetic，FHEW/TFHE 承担 encrypted control，但禁止
FHEW 只生成 one-hot 后回到 Dense CKKS。

至少比较三种接口：

1. CKKS activation → FHEW components → fused/tiled operator → CKKS output；
2. FHEW 生成可复用的 compact control object，在 CKKS 中控制非 Dense 算子；
3. RLWE coefficient-packed functional bootstrap，中间直接做 CKKS leveled arithmetic。

计时必须包含 CKKS↔FHEW/RLWE 转换、布局转换、bootstrap 和所需 rotations；验证转换后的
range、precision 和 security 参数一致。

### 阶段 F：完整 MoE layer 与模型级验证

依次加入 router linear、encrypted top-k、三路 expert projection、SwiGLU、score combine、
multi-token packing、residual/norm/attention 接口。里程碑分为完整 MoE layer、完整
Transformer block、完整模型，不能混称端到端。

模型级质量先用原始明文模型确认任务基线，再验证密码学数值误差。LowRank 若保留，必须作为
单独的 approximate variant 报告 perplexity/zero-shot accuracy，不能与 exact 结果合并。

## 5. 统一 baseline、指标和实验矩阵

同口径 baseline：

1. Public Routing：公开 `e` 的性能下界；
2. Dense-FHE：计算所有 experts 再秘密 combine；
3. Select-Weight full：materialize 完整 `Enc(W_e)` 后 MatMul；
4. Select-Weight stream：分块选权重并立即应用；
5. LowRank Structured：只作为 approximate/structured baseline；
6. FHEW Exact SelectApply：主候选。

所有方法使用相同硬件、线程、安全级别、数值范围、输入、packing 约束和正确性阈值。至少扫描
`N,d_in,d_out,bitwidth,tokens,top-k,tile/packing`。统一记录：

- setup/preprocess/keygen/encrypt/online/decrypt latency，median 与分位数；
- PBS、CMux、rotation、ct-pt、ct-ct、bootstrap、scheme-switch 次数；
- ciphertext、plaintext model、LUT、evaluation key 和进程 RSS 峰值；
- HE numerical error、overflow/failure probability；仅 LowRank 另报 structural error；
- throughput、latency/token、请求/响应字节数和失败/OOM 点。

主结论需要同时满足：输出选择语义 exact；routing 不泄漏；不物化完整选中权重；在线求值不
包含 `N` 个完整 MatMul；在真实维度的外推或实测中，相对 Dense-FHE/Select-Weight 有明确
时间或内存收益。任何一项未满足，都应把结果表述为 partial primitive 或 negative result。
