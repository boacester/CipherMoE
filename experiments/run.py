#!/usr/bin/env python3
"""Generate reproducible models, run each method in a fresh process, and report results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    experts: int
    dim: int
    rank: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profile", choices=("smoke", "main"), default="main")
    parser.add_argument("--skip-lut", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def cases(profile: str) -> list[Case]:
    if profile == "smoke":
        return [
            Case("structured_n4_d8_r2", "structured", 4, 8, 2, 1101),
            Case("random_n4_d8_r2", "random", 4, 8, 2, 1201),
        ]
    return [
        Case("structured_n8_d16_r2", "structured", 8, 16, 2, 2101),
        Case("random_n8_d16_r2", "random", 8, 16, 2, 2201),
        Case("random_n8_d16_r4", "random", 8, 16, 4, 2201),
        Case("random_n8_d16_r8", "random", 8, 16, 8, 2201),
    ]


def make_model(case: Case) -> dict[str, list]:
    rng = np.random.default_rng(case.seed)
    scale = 0.25 / np.sqrt(case.dim)
    inputs = rng.uniform(-0.5, 0.5, size=(4, case.dim))
    inputs[1] = 0.0
    inputs[2] = np.resize(np.array([-0.5, 0.5]), case.dim)

    if case.kind == "structured":
        basis = rng.normal(0.0, scale, size=(case.rank, case.dim, case.dim))
        coefficients = rng.normal(0.0, 0.7, size=(case.experts, case.rank))
        weights = np.einsum("er,rij->eij", coefficients, basis)
        approximation = weights.copy()
    else:
        weights = rng.normal(0.0, scale, size=(case.experts, case.dim, case.dim))
        flattened = weights.reshape(case.experts, -1)
        u, singular, vt = np.linalg.svd(flattened, full_matrices=False)
        basis = vt[: case.rank].reshape(case.rank, case.dim, case.dim)
        coefficients = u[:, : case.rank] * singular[: case.rank]
        approximation = np.einsum("er,rij->eij", coefficients, basis)

    return {
        "weights": weights.tolist(),
        "basis": basis.tolist(),
        "approximation": approximation.tolist(),
        "coefficients": coefficients.tolist(),
        "inputs": inputs.tolist(),
    }


def run_json(command: list[str], env: dict[str, str]) -> dict:
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"invalid JSON from {' '.join(command)}:\n{completed.stdout}\n{completed.stderr}"
        ) from error


def method_runs(dim: int) -> list[tuple[str, str, int | None]]:
    return [
        ("public", "public", None),
        ("dense", "dense", None),
        ("select_weight_full", "select_weight", dim),
        ("select_weight_stream", "select_weight", 1),
        ("low_rank_basis", "low_rank_basis", None),
    ]


def write_csv(records: list[dict], path: Path) -> None:
    fields = [
        "case",
        "kind",
        "label",
        "method",
        "n",
        "dim",
        "rank",
        "block_size",
        "median_ms",
        "max_he_error",
        "relative_output_error",
        "ct_pt",
        "ct_ct",
        "rotations",
        "additions",
        "selected_weight_peak_bytes",
        "peak_rss_kib",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def plot(records: list[dict], output_dir: Path) -> None:
    case_names = list(dict.fromkeys(record["case"] for record in records))
    labels = ["public", "dense", "select_weight_full", "select_weight_stream", "low_rank_basis"]
    fig, axes = plt.subplots(len(case_names), 1, figsize=(9, 3.2 * len(case_names)), squeeze=False)
    for axis, case_name in zip(axes[:, 0], case_names):
        by_label = {r["label"]: r for r in records if r["case"] == case_name}
        present = [label for label in labels if label in by_label]
        axis.bar(present, [by_label[label]["median_ms"] for label in present])
        axis.set_yscale("log")
        axis.set_ylabel("median online ms (log)")
        axis.set_title(case_name)
        axis.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(output_dir / "latency.png", dpi=180)
    plt.close(fig)


def write_report(records: list[dict], lut: dict | None, metadata: dict, path: Path) -> None:
    by_case_label = {(record["case"], record["label"]): record for record in records}
    lines = [
        "# CipherMoE 实验报告",
        "",
        f"生成时间：{metadata['timestamp']}  ",
        f"CPU：{metadata['cpu']}  ",
        f"OpenMP 线程：{metadata['threads']}；每项在线重复：{metadata['repeats']}  ",
        "CKKS 安全级别：HEStd_128_classic；FHEW 参数：STD128。",
        "",
        "## CKKS 矩阵实验",
        "",
        "| case | method | median ms | rel. output error | ct-pt | ct-ct | rotations | selected-weight peak MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| {case} | {label} | {median_ms:.3f} | {relative_output_error:.3e} | "
            "{ct_pt} | {ct_ct} | {rotations} | {peak:.2f} |".format(
                peak=record["selected_weight_peak_bytes"] / (1024 * 1024), **record
            )
        )
    if lut:
        lines.extend(
            [
                "",
                "## FHEW 标量联合查表",
                "",
                "该微基准从分别加密的 1-bit expert ID 与 1-bit activation 开始；线性联合编码计入在线时间。",
                "",
                "| variant | median ms | EvalFunc | inferred internal PBS |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for result in lut["results"]:
            lines.append(
                f"| {result['name']} | {result['median_ms']:.3f} | "
                f"{result['eval_func_calls']} | {result['internal_pbs']} |"
            )
    structured = "structured_n8_d16_r2"
    random_r2 = "random_n8_d16_r2"
    random_r4 = "random_n8_d16_r4"
    random_r8 = "random_n8_d16_r8"
    required = [
        (structured, "dense"),
        (structured, "select_weight_stream"),
        (structured, "low_rank_basis"),
        (random_r2, "low_rank_basis"),
        (random_r4, "low_rank_basis"),
        (random_r8, "dense"),
        (random_r8, "low_rank_basis"),
        (structured, "select_weight_full"),
    ]
    if all(key in by_case_label for key in required):
        dense = by_case_label[(structured, "dense")]["median_ms"]
        select = by_case_label[(structured, "select_weight_stream")]["median_ms"]
        basis = by_case_label[(structured, "low_rank_basis")]["median_ms"]
        exact_dense = by_case_label[(random_r8, "dense")]["median_ms"]
        exact_basis = by_case_label[(random_r8, "low_rank_basis")]["median_ms"]
        full_peak = by_case_label[(structured, "select_weight_full")]["selected_weight_peak_bytes"]
        stream_peak = by_case_label[(structured, "select_weight_stream")]["selected_weight_peak_bytes"]
        lines.extend(
            [
                "",
                "## 主要结论",
                "",
                f"- 对精确 rank-2 共享结构，LowRank Structured 相对 Dense 加速 {dense / basis:.2f}×，"
                f"相对流式 Select-Weight 加速 {select / basis:.2f}×，且不物化选中权重。",
                f"- 对随机权重，rank-2/rank-4 虽更快，但相对输出误差仍为 "
                f"{by_case_label[(random_r2, 'low_rank_basis')]['relative_output_error']:.3f}/"
                f"{by_case_label[(random_r4, 'low_rank_basis')]['relative_output_error']:.3f}；"
                f"rank-8 达到近似精确时反而比 Dense 慢 {exact_basis / exact_dense:.2f}×。",
                f"- Select-Weight 的 block=1 将显式选中权重峰值从 "
                f"{full_peak / 2**20:.0f} MiB 降到 {stream_peak / 2**20:.0f} MiB，"
                "但密码学操作数不变，因此只解决存储而非计算。",
            ]
        )
        if lut:
            lut_by_name = {item["name"]: item for item in lut["results"]}
            fused = lut_by_name["fused_negacyclic"]["median_ms"]
            staged = lut_by_name["select_then_apply_negacyclic"]["median_ms"]
            lines.append(
                f"- FHEW 半域负循环融合相对两阶段查表加速 {staged / fused:.2f}×；"
                "它证明标量融合有效，但尚未解决高维打包与 CKKS/FHEW 转换成本。"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `relative_output_error` 对 low_rank_basis 是低秩近似相对输出误差；其他方法应为 0（仅另报 CKKS 数值误差）。",
            "- `selected-weight peak` 只统计显式选中权重密文载荷，不等同于进程 RSS。",
            "- CKKS 从加密 one-hot 开始；FHEW 从加密 ID 开始，二者不是端到端同口径结果。",
            "- 当前矩阵规模用于验证 primitive 和趋势，不代表完整 MoE 层或真实模型权重。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    if args.repeats < 1 or args.threads < 1:
        raise SystemExit("--repeats and --threads must be positive")
    matrix_binary = args.build_dir / "matrix_benchmark"
    lut_binary = args.build_dir / "lut_benchmark"
    if not matrix_binary.exists():
        raise SystemExit(f"missing {matrix_binary}; build the project first")
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "models"
    raw_dir = output_dir / "raw"
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    openfhe_lib = str(ROOT / "thirdparty" / "openfhe" / "install" / "lib")
    env["LD_LIBRARY_PATH"] = openfhe_lib + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    records: list[dict] = []
    failures: list[str] = []
    for case in cases(args.profile):
        model_path = model_dir / f"{case.name}.json"
        model_path.write_text(json.dumps(make_model(case), separators=(",", ":")))
        for label, method, block in method_runs(case.dim):
            command = [str(matrix_binary), str(model_path), method, str(args.repeats)]
            if block is not None:
                command.append(str(block))
            print(f"running {case.name}: {label}", flush=True)
            try:
                result = run_json(command, env)
            except RuntimeError as error:
                if not args.keep_going:
                    raise
                failures.append(str(error))
                continue
            result.update(case=case.name, kind=case.kind, label=label)
            records.append(result)
            (raw_dir / f"{case.name}__{label}.json").write_text(json.dumps(result, indent=2))

    lut_result = None
    if not args.skip_lut:
        if not lut_binary.exists():
            failures.append(f"missing {lut_binary}")
        else:
            print("running FHEW LUT benchmark", flush=True)
            try:
                lut_result = run_json([str(lut_binary), str(args.repeats)], env)
                (raw_dir / "fused_lut.json").write_text(json.dumps(lut_result, indent=2))
            except RuntimeError as error:
                if not args.keep_going:
                    raise
                failures.append(str(error))

    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "cpu": cpu,
        "threads": args.threads,
        "repeats": args.repeats,
        "profile": args.profile,
        "failures": failures,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    write_csv(records, output_dir / "summary.csv")
    if records:
        plot(records, output_dir)
    write_report(records, lut_result, metadata, output_dir / "report.md")
    if failures:
        (output_dir / "failures.log").write_text("\n\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
