# 阶段 A：真实 MoE 的跨 expert 低秩结构分析

## 实验口径

对每个真实 checkpoint 的每个 MoE 层和 `gate_proj/up_proj/down_proj`，把 `N` 个
expert 权重展平为 `N × P` 矩阵，纳入全部权重元素并用 FP32 Gram 矩阵做精确
expert-wise SVD。权重误差是相对 Frobenius 误差；输出误差使用校准文本捕获的
真实 activation。`r@1%`/`r@0.1%` 是达到相应相对误差所需的最小跨-expert rank，
格式为 `中位数 [最小, 最大]`。这里的 rank 不是单个权重矩阵自身的矩阵秩。

该实验只判断 `LowRank_Structured` 是否值得作为可选优化；无论结果如何，它都不
等同于任意 pretrained MoE 的 exact Select-and-Apply。

## Global SVD 结果

| 模型 | projection | 层数 | N | 权重 r@1% | 权重 r@0.1% | 输出 r@1% | 输出 r@0.1% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLMoE-1B-7B-0924 | `gate_proj` | 16 | 64 | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |
| OLMoE-1B-7B-0924 | `up_proj` | 16 | 64 | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |
| OLMoE-1B-7B-0924 | `down_proj` | 16 | 64 | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |
| Qwen1.5-MoE-A2.7B | `gate_proj` | 24 | 60 | 60 [60, 60] | 60 [60, 60] | 60 [60, 60] | 60 [60, 60] |
| Qwen1.5-MoE-A2.7B | `up_proj` | 24 | 60 | 60 [60, 60] | 60 [60, 60] | 60 [60, 60] | 60 [60, 60] |
| Qwen1.5-MoE-A2.7B | `down_proj` | 24 | 60 | 60 [60, 60] | 60 [60, 60] | 60 [60, 60] | 60 [60, 60] |
| deepseek-moe-16b-base | `gate_proj` | 27 | 64 | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |
| deepseek-moe-16b-base | `up_proj` | 27 | 64 | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |
| deepseek-moe-16b-base | `down_proj` | 27 | 64 | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |

## 结构变体（权重 r@1%）

| 模型 | projection | global | grouped-4 | shared+delta |
| --- | --- | ---: | ---: | ---: |
| OLMoE-1B-7B-0924 | `gate_proj` | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |
| OLMoE-1B-7B-0924 | `up_proj` | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |
| OLMoE-1B-7B-0924 | `down_proj` | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |
| Qwen1.5-MoE-A2.7B | `gate_proj` | 60 [60, 60] | 60 [60, 60] | 60 [60, 60] |
| Qwen1.5-MoE-A2.7B | `up_proj` | 60 [60, 60] | 60 [60, 60] | 60 [60, 60] |
| Qwen1.5-MoE-A2.7B | `down_proj` | 60 [60, 60] | 60 [60, 60] | 60 [60, 60] |
| deepseek-moe-16b-base | `gate_proj` | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |
| deepseek-moe-16b-base | `up_proj` | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |
| deepseek-moe-16b-base | `down_proj` | 64 [64, 64] | 64 [64, 64] | 64 [64, 64] |

完整逐 rank 曲线见 `weight_error_curves.png` 和 `output_error_curves.png`；逐层原始
谱与误差见 `weight_structure.jsonl`，阈值表见 `rank_thresholds.csv`。

## 对 LowRank Structured 的判断

global SVD 在 1% 权重误差下所需 `r/N` 的中位数为 1.000、90 分位为 1.000；在 0.1% 下分别为 1.000 和 1.000。每增加一个 rank，
该候选都需要再做一次完整明文基矩阵应用和一次密文门控。因此只有低误差下的
`r/N` 仍显著小于 1，结构压缩才可能产生密码学收益；若普遍 `r≈N`，停止把它
作为性能主线，只保留为 structured-model baseline。分组结果只是结构上限；若为了
隐藏 group ID 而计算所有组，总 basis 预算仍须全部计费。

**阶段结论（no-go）：全部 201 个 layer/projection 组在 1% 权重误差和
1% 真实激活输出误差下都需要 `r=N`，0.1% 阈值同样如此。在较宽松的 5% 阈值下，
权重误差仍有 201/201 组需要 `r=N`，真实激活输出误差有
200/201 组需要 `r=N`。因此，这三个 checkpoint 不支持
`r≪N` 的关键假设。停止把 `LowRank_Structured` 作为主方案，
仅保留为 structured-model baseline；主线转入不依赖跨-expert 结构的 exact
`FHEW_SelectApply`。**

## 限制

本阶段衡量单层线性 projection，不等同于完整模型的 perplexity 或下游准确率。
校准集输出误差用于检查权重范数指标是否误导，不替代模型级质量评估。它也不能
回答 exact FHEW Select-and-Apply 的可行性；后者不依赖跨-expert 低秩结构。
