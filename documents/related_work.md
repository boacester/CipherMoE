# SecMoE 与 CryptoMoE：实验组织和比较边界

本文只总结 related work 及其对实验设计的启示。CipherMoE 的实现与验证路线见
[implement.md](implement.md)。`idea.md` 是早期草稿，不作为已经成立的技术结论。

## 1. SecMoE

[SecMoE](SecMoE.pdf) 是交互式两方安全计算（2PC）方案。它用
Select-Then-Compute 先在秘密共享/电路中选择参数，再对被选条目执行安全计算，以避免
把所有 experts 的完整 FFN 都算一遍。它不是非交互纯 FHE。

论文实验按以下证据链组织：

| 层次 | 实验内容 |
| --- | --- |
| 环境 | 两个节点；每节点 64 vCPU、128 GB RAM；LAN 与 WAN |
| 数值 | `Z_(2^64)` 定点数，18-bit 小数精度 |
| 模型 | 由 T5-small 改造的 MoE-Small、Switch-Base |
| 扩展变量 | expert 数从 4/8 扩展到 128；top-1；8 tokens |
| baseline | Iron、BumbleBee；解释不比较 trusted-dealer/大客户端内存方案的原因 |
| 系统指标 | 端到端时间、通信量；LAN/WAN 分开报告；离线与在线总成本 |
| 模型指标 | CoLA、QNLI、RTE，与浮点明文结果比较 |
| 子协议 | 安全 GeLU 和通信量随 expert 数的扩展趋势 |

可借鉴之处是：先固定威胁模型和比较边界；同时报告系统成本与模型质量；不仅给单点结果，
还扫描 expert 数；明确是否计入离线阶段。

限制是：它依赖交互式 2PC、固定 top-1 和很小的 token 数，且 MoE-Small 由 dense 模型
改造而来。其绝对延迟不能直接作为 CipherMoE 的同口径 baseline。截至 2026-09-26，
论文和公开检索中未找到作者给出的官方实现，因此只能参考论文中的协议与实验方法，不能
把未核验的复现代码当作基础设施。论文入口：
[AAAI 版本](https://ojs.aaai.org/index.php/AAAI/article/download/39721/43682)、
[arXiv](https://arxiv.org/abs/2601.06790)。

## 2. CryptoMoE

[CryptoMoE](CryptoMoE.pdf) 也是 HE/MPC 混合的交互式 2PC。它通过 balanced routing
固定每个 expert 的 token 容量，并丢弃超额 token 以隐藏负载；还加入
confidence-aware token selection 和 batch MatMul。这会改变原模型的路由/输出，因此与
“不修改任意 pretrained MoE”的目标不同。

它的实验证据顺序很完整：

```text
单个 MoE 层 -> 完整模型 -> 参数消融 -> 组件消融
            -> latency breakdown -> 扩展性与失败边界
```

| 项目 | 设置 |
| --- | --- |
| 框架 | SecretFlow-SPU，无 trusted third party |
| 硬件 | Intel Xeon Platinum 8468，48 cores |
| 网络 | LAN 3 Gbps/0.2 ms；WAN 400 Mbps/40 ms |
| 模型 | DeepSeekMoE、OLMoE、QWenMoE |
| 任务 | 8 个 zero-shot commonsense reasoning 数据集 |
| baseline | 公开路由 Insecure、全 experts Dense、改造后的 CipherPrune |
| 扫描变量 | expert 容量 `t`、序列长度、batch size |
| 指标 | accuracy、amortized latency/token、communication/token |

它还逐项加入 balanced routing、confidence-aware selection 和 batch MatMul，并拆分
router、dispatch、expert linear、SiLU、combine、attention 的延迟。这种组织能暴露“某个
kernel 加速后，top-k 或网络通信成为新瓶颈”的情况，适合复用到 CipherMoE。

官方仓库是 [PKU-SEC-Lab/CryptoMoE](https://github.com/PKU-SEC-Lab/CryptoMoE)，
Apache-2.0。2026-09-26 检查的 commit 为
`d3074967c86110ddda3c7e03f29106aff7e64a30`。仓库目前公开的是 balanced-routing 的
模型改造和 lm-evaluation-harness 评估；README 中密码文/SPU 实现仍标为 TODO。因此可直接
参考其模型质量评测入口，但不能复用尚未公开的密文协议实现。

## 3. 与 exact Select-and-Apply 的关系

CipherMoE 的目标是：

```text
SelectApply(Enc(e), Enc(x), {W_i}) -> Enc(W_e x)
```

其中使用任意 pretrained MoE 的原始明文 expert weights，不重训、不要求跨-expert 结构，
隐藏路由，不物化完整 `Enc(W_e)`，在线成本不能退化为 `N` 次完整 expert MatMul。

这里的 exact 是“精确选择原始 `W_e`，不引入 LowRank 等模型近似”。浮点编码、定点量化、
CKKS 舍入和近似激活造成的数值误差必须另报，不能与结构近似混在一起。

三者的比较边界如下：

| 方案 | 密码学模型 | 是否修改/约束模型 | 能否作为绝对延迟 baseline |
| --- | --- | --- | --- |
| SecMoE | 交互式 2PC | 协议与模型设置受限 | 否；作为 related work 趋势参考 |
| CryptoMoE | HE/MPC 交互式 2PC | balanced routing 改变路由行为 | 否；复用实验组织和质量任务 |
| CipherMoE | 非交互 FHE 目标 | 任意原始 weights | 主实验须在统一 FHE 参数下自建 baseline |

因此论文主 baseline 应是同一硬件、安全级别、packing 和数值口径下的 Public Routing、
Dense-FHE、Select-Weight，以及 exact FHEW/TFHE candidate。SecMoE/CryptoMoE 的论文数字
只用于说明不同威胁模型下的设计空间。

## 4. 应复用的实验组织

1. 先测 primitive 正确性、复杂度和扩展趋势，再进入完整 MoE 层。
2. 分开 setup/model preprocessing、key generation、encryption、online evaluation、decryption。
3. 同时报告 latency、吞吐、通信、峰值内存、evaluation-key/model storage 和数值误差。
4. 做参数消融、组件消融与 latency breakdown，不用单个 kernel 数字代替端到端结论。
5. 最后报告完整 block/模型质量；跑不动的规模明确写成失败或 OOM 边界。
