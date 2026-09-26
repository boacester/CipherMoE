#!/usr/bin/env python3
"""Render a concise stage-A report from the raw weight-structure records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def first_rank(errors: list[float], threshold: float) -> int:
    return next(
        (index + 1 for index, error in enumerate(errors) if error <= threshold),
        len(errors),
    )


def rank_stats(rows: list[dict], method: str, metric: str, threshold: float) -> str:
    ranks = [first_rank(row["methods"][method][metric], threshold) for row in rows]
    return f"{np.median(ranks):.0f} [{min(ranks)}, {max(ranks)}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "stage_a")
    args = parser.parse_args()
    raw_path = args.results / "weight_structure.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No records in {raw_path}")
    if not all(row["exact"] for row in rows):
        raise SystemExit("Refusing to publish a final report from sampled-feature records")

    lines = [
        "# 阶段 A：真实 MoE 权重结构分析",
        "",
        "## 实验口径",
        "",
        "对每个真实 checkpoint 的每个 MoE 层和 `gate_proj/up_proj/down_proj`，把 `N` 个",
        "expert 权重展平为 `N × P` 矩阵，纳入全部权重元素并用 FP32 Gram 矩阵做特征分解。",
        "权重误差是相对",
        "Frobenius 误差；输出误差使用 64-token 校准文本捕获的每层真实 activation，每个",
        "projection 固定抽取 32 个输入，并让所有 experts 在同一输入集上比较。表中数字是",
        "达到 5% 相对误差所需的总 full-matrix basis 数，格式为 `中位数 [最小, 最大]`。",
        "",
        "- `global`：当前 Basis 直接使用的跨-expert SVD 最优基。",
        "- `grouped_g4`：按权重余弦距离把 experts 聚成 4 组，再在各组内分配总 basis 预算。",
        "- `shared_plus_delta`：一个共享均值基，再对跨-expert 残差做低秩分解。这里不是",
        "  对每个权重矩阵内部做低秩分解。",
        "",
        "## 结果",
        "",
        "| 模型 | projection | 层数 | N | global 权重 r@5% | global 输出 r@5% | grouped-4 权重 r@5% | shared+delta 权重 r@5% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    models = sorted({row["model"] for row in rows})
    projections = ("gate_proj", "up_proj", "down_proj")
    global_ratios = []
    missing_output = False
    for model in models:
        for projection in projections:
            subset = [
                row for row in rows
                if row["model"] == model and row["projection"] == projection
            ]
            if not subset:
                continue
            experts = subset[0]["experts"]
            weight_stat = rank_stats(subset, "global", "relative_weight_error", 0.05)
            grouped_stat = rank_stats(subset, "grouped_g4", "relative_weight_error", 0.05)
            shared_stat = rank_stats(subset, "shared_plus_delta", "relative_weight_error", 0.05)
            if all("relative_output_error" in row["methods"]["global"] for row in subset):
                output_stat = rank_stats(subset, "global", "relative_output_error", 0.05)
            else:
                output_stat = "未采集"
                missing_output = True
            global_ranks = [
                first_rank(row["methods"]["global"]["relative_weight_error"], 0.05)
                for row in subset
            ]
            global_ratios.extend(rank / experts for rank in global_ranks)
            lines.append(
                f"| {model} | `{projection}` | {len(subset)} | {experts} | {weight_stat} | "
                f"{output_stat} | {grouped_stat} | {shared_stat} |"
            )

    median_ratio = float(np.median(global_ratios))
    p90_ratio = float(np.quantile(global_ratios, 0.9))
    lines.extend([
        "",
        "完整逐 rank 曲线见 `weight_error_curves.png`，逐层原始谱和误差见",
        "`weight_structure.jsonl`，阈值表见 `rank_thresholds.csv`。",
        "",
        "## 对当前 Basis 的判断",
        "",
        f"在 5% 权重误差口径下，global basis 所需 rank/N 的中位数为 {median_ratio:.3f}，",
        f"90 分位为 {p90_ratio:.3f}。Basis 的每一个 rank 都需要一次完整明文基矩阵应用和",
        "一次密文门控；因此只有 `r` 显著小于 `N`，结构压缩才能稳定转化为密码学收益。",
        "分组结果只表示结构上限：若为了隐藏组 ID 而计算所有组，其总 basis 数仍必须全部",
        "计费。共享均值基的系数恒为 1，可以免去该基的秘密系数选择，但不能免去矩阵应用。",
    ])
    if missing_output:
        lines.extend([
            "",
            "> 部分或全部 activation-aware 输出误差尚未采集；在补齐校准运行前，本报告不能",
            "> 作为阶段 A 的最终结论。",
        ])
    lines.extend([
        "",
        "## 限制",
        "",
        "本阶段衡量单层线性 projection，不等同于完整 MoE 的 perplexity 或下游准确率。",
        "校准集很小，输出误差用于判断权重范数指标是否明显误导，不替代阶段 D 的模型质量",
        "评估。分组方案还没有把私有 group selection 的 FHE 成本加入；那部分属于阶段 B/C。",
        "",
    ])
    (args.results / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
