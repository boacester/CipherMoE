#!/usr/bin/env python3
"""Render a concise stage-A report from the raw weight-structure records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "stage_a_models.json"


def first_rank(errors: list[float], threshold: float) -> int:
    rank = next(
        (index + 1 for index, error in enumerate(errors) if error <= threshold),
        None,
    )
    if rank is None:
        raise ValueError(
            f"Full-rank error {errors[-1]} does not reach threshold {threshold}"
        )
    return rank


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

    registry = json.loads(REGISTRY.read_text())["models"]
    expected_names = {entry["name"] for entry in registry}
    actual_names = {row["model"] for row in rows}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise SystemExit(
            f"Incomplete Stage-A model set; missing={missing}, unexpected={unexpected}"
        )
    for entry in registry:
        model_rows = [row for row in rows if row["model"] == entry["name"]]
        keys = {(row["layer"], row["projection"]) for row in model_rows}
        expected_groups = entry["expected_moe_layers"] * 3
        if len(model_rows) != expected_groups or len(keys) != expected_groups:
            raise SystemExit(
                f"Incomplete or duplicate records for {entry['name']}: "
                f"records={len(model_rows)}, unique_groups={len(keys)}, "
                f"expected={expected_groups}"
            )
        if len({row["layer"] for row in model_rows}) != entry["expected_moe_layers"]:
            raise SystemExit(f"Unexpected MoE layer count for {entry['name']}")
        if {row["projection"] for row in model_rows} != {
            "gate_proj", "up_proj", "down_proj"
        }:
            raise SystemExit(f"Unexpected projection set for {entry['name']}")
        if any(row["experts"] != entry["expected_experts"] for row in model_rows):
            raise SystemExit(f"Unexpected expert count for {entry['name']}")
        if any(
            row["repo_id"] != entry["repo_id"]
            or row["revision"] != entry["revision"]
            for row in model_rows
        ):
            raise SystemExit(f"Checkpoint identity mismatch for {entry['name']}")
        if any(row["features_used"] != row["features_total"] for row in model_rows):
            raise SystemExit(f"Sampled feature record found for {entry['name']}")
        if any(row.get("calibration_samples", 0) <= 0 for row in model_rows):
            raise SystemExit(f"Missing activation calibration for {entry['name']}")

    lines = [
        "# 阶段 A：真实 MoE 的跨 expert 低秩结构分析",
        "",
        "## 实验口径",
        "",
        "对每个真实 checkpoint 的每个 MoE 层和 `gate_proj/up_proj/down_proj`，把 `N` 个",
        "expert 权重展平为 `N × P` 矩阵，纳入全部权重元素并用 FP32 Gram 矩阵做精确",
        "expert-wise SVD。权重误差是相对 Frobenius 误差；输出误差使用校准文本捕获的",
        "真实 activation。`r@1%`/`r@0.1%` 是达到相应相对误差所需的最小跨-expert rank，",
        "格式为 `中位数 [最小, 最大]`。这里的 rank 不是单个权重矩阵自身的矩阵秩。",
        "",
        "该实验只判断 `LowRank_Structured` 是否值得作为可选优化；无论结果如何，它都不",
        "等同于任意 pretrained MoE 的 exact Select-and-Apply。",
        "",
        "## Global SVD 结果",
        "",
        "| 模型 | projection | 层数 | N | 权重 r@1% | 权重 r@0.1% | 输出 r@1% | 输出 r@0.1% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    models = sorted({row["model"] for row in rows})
    projections = ("gate_proj", "up_proj", "down_proj")
    ratios_1pct = []
    ratios_0_1pct = []
    missing_output = False
    subsets: list[tuple[str, str, list[dict]]] = []
    for model in models:
        for projection in projections:
            subset = [
                row for row in rows
                if row["model"] == model and row["projection"] == projection
            ]
            if not subset:
                continue
            subsets.append((model, projection, subset))
            experts = subset[0]["experts"]
            weight_1 = rank_stats(subset, "global", "relative_weight_error", 0.01)
            weight_0_1 = rank_stats(subset, "global", "relative_weight_error", 0.001)
            if all("relative_output_error" in row["methods"]["global"] for row in subset):
                output_1 = rank_stats(subset, "global", "relative_output_error", 0.01)
                output_0_1 = rank_stats(subset, "global", "relative_output_error", 0.001)
            else:
                output_1 = output_0_1 = "未采集"
                missing_output = True
            ranks_1 = [
                first_rank(row["methods"]["global"]["relative_weight_error"], 0.01)
                for row in subset
            ]
            ranks_0_1 = [
                first_rank(row["methods"]["global"]["relative_weight_error"], 0.001)
                for row in subset
            ]
            ratios_1pct.extend(rank / experts for rank in ranks_1)
            ratios_0_1pct.extend(rank / experts for rank in ranks_0_1)
            lines.append(
                f"| {model} | `{projection}` | {len(subset)} | {experts} | {weight_1} | "
                f"{weight_0_1} | {output_1} | {output_0_1} |"
            )

    lines.extend([
        "",
        "## 结构变体（权重 r@1%）",
        "",
        "| 模型 | projection | global | grouped-4 | shared+delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for model, projection, subset in subsets:
        lines.append(
            f"| {model} | `{projection}` | "
            f"{rank_stats(subset, 'global', 'relative_weight_error', 0.01)} | "
            f"{rank_stats(subset, 'grouped_g4', 'relative_weight_error', 0.01)} | "
            f"{rank_stats(subset, 'shared_plus_delta', 'relative_weight_error', 0.01)} |"
        )

    med_1 = float(np.median(ratios_1pct))
    p90_1 = float(np.quantile(ratios_1pct, 0.9))
    med_0_1 = float(np.median(ratios_0_1pct))
    p90_0_1 = float(np.quantile(ratios_0_1pct, 0.9))
    full_weight_5pct = sum(
        first_rank(row["methods"]["global"]["relative_weight_error"], 0.05)
        == row["experts"]
        for row in rows
    )
    full_output_5pct = sum(
        first_rank(row["methods"]["global"]["relative_output_error"], 0.05)
        == row["experts"]
        for row in rows
    )
    global_full_rank_1pct = all(
        first_rank(row["methods"]["global"]["relative_weight_error"], 0.01)
        == row["experts"]
        and first_rank(row["methods"]["global"]["relative_output_error"], 0.01)
        == row["experts"]
        for row in rows
    )
    lines.extend([
        "",
        "完整逐 rank 曲线见 `weight_error_curves.png` 和 `output_error_curves.png`；逐层原始",
        "谱与误差见 `weight_structure.jsonl`，阈值表见 `rank_thresholds.csv`。",
        "",
        "## 对 LowRank Structured 的判断",
        "",
        f"global SVD 在 1% 权重误差下所需 `r/N` 的中位数为 {med_1:.3f}、90 分位为 "
        f"{p90_1:.3f}；在 0.1% 下分别为 {med_0_1:.3f} 和 {p90_0_1:.3f}。每增加一个 rank，",
        "该候选都需要再做一次完整明文基矩阵应用和一次密文门控。因此只有低误差下的",
        "`r/N` 仍显著小于 1，结构压缩才可能产生密码学收益；若普遍 `r≈N`，停止把它",
        "作为性能主线，只保留为 structured-model baseline。分组结果只是结构上限；若为了",
        "隐藏 group ID 而计算所有组，总 basis 预算仍须全部计费。",
    ])
    if global_full_rank_1pct:
        lines.extend([
            "",
            f"**阶段结论（no-go）：全部 {len(rows)} 个 layer/projection 组在 1% 权重误差和",
            "1% 真实激活输出误差下都需要 `r=N`，0.1% 阈值同样如此。在较宽松的 5% 阈值下，",
            f"权重误差仍有 {full_weight_5pct}/{len(rows)} 组需要 `r=N`，真实激活输出误差有",
            f"{full_output_5pct}/{len(rows)} 组需要 `r=N`。因此，这三个 checkpoint 不支持",
            "`r≪N` 的关键假设。停止把 `LowRank_Structured` 作为主方案，",
            "仅保留为 structured-model baseline；主线转入不依赖跨-expert 结构的 exact",
            "`FHEW_SelectApply`。**",
        ])
    if missing_output:
        lines.extend([
            "",
            "> 部分或全部 activation-aware 输出误差尚未采集；补齐校准运行前，本报告不能",
            "> 作为阶段 A 的最终结论。",
        ])
    lines.extend([
        "",
        "## 限制",
        "",
        "本阶段衡量单层线性 projection，不等同于完整模型的 perplexity 或下游准确率。",
        "校准集输出误差用于检查权重范数指标是否误导，不替代模型级质量评估。它也不能",
        "回答 exact FHEW Select-and-Apply 的可行性；后者不依赖跨-expert 低秩结构。",
        "",
    ])
    (args.results / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
