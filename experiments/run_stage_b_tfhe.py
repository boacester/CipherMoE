#!/usr/bin/env python3
"""Explore TFHE-rs WOPBS separately; legacy parameters are not security-matched."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "## TFHE-rs WOPBS-only（探索性参数）"


def report(output: Path) -> None:
    path = output / "report.md"
    if not path.exists():
        return
    rows = json.loads((output / "tfhe_summary.json").read_text())
    failures = json.loads((output / "tfhe_failures.json").read_text())
    content = path.read_text().split(MARKER)[0]
    content = content.replace(
        "- TFHE-rs WOPBS / vertical packing 尚未集成或实测，不能写成优于/劣于 OpenFHE。",
        "- TFHE-rs WOPBS-only 的 legacy 参数已独立实测，但未证明与 OpenFHE STD128 安全等价；"
        "多输出 vertical packing 仍未实测。",
    )
    content = content.replace(
        "当前仅证明部分 exact 标量原语可计算；没有完成 TFHE-rs 对照和所有 N 的穷举。",
        "当前仅证明部分 exact 标量原语可计算；TFHE-rs WOPBS-only 已单列探索，"
        "未完成同口径安全参数对照与所有 N 的穷举。",
    )
    content = content.replace("；  多输出", "；多输出")
    if "CMUX 由客户端分别加密的 ID bits" not in content and "CMUX 输入是客户端" not in content:
        content = content.replace(
            "  leaf functions 并对相同子表达式做缓存，不是可扩展的矩阵方案。",
            "  leaf functions 并对相同子表达式做缓存，不是可扩展的矩阵方案。\n"
            "- CMUX 输入是客户端分别加密的 ID bits 和 x bits，没有把单个 `Enc(e)` "
            "同态分解成 bits 的成本；输出 4 个 bit 密文，而 fused LUT 输出 1 个标量密文。",
        )
    lines = [
        MARKER, "",
        "固定 Zama tfhe-rs `079f46d08ffb8763de54b7794e850ddbd98fe63b` (BSD-3-Clause-Clear)，",
        "`shortint` + `experimental` 的 `WopbsKey::new_wopbs_key_only_for_wopbs`、",
        "`generate_lut_without_padding` / `programmable_bootstrapping_without_padding`。",
        "选择 ID 与 x 分开加密，在线先线性合成为 `4e+code`，再做 WOPBS。",
        "这是 single-output WOPBS，并非多输出 vertical packing 对照。", "",
        "| N | p | 检查/错误 | online median (ms) | key (GiB) | LUT (KiB) | setup (s) | peak RSS (GiB) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['n']} | {row['p']} | {row['verified']}/{row['failures']} | "
            f"{row['median_ms']:.1f} | {row['evaluation_key_bytes']/2**30:.2f} | "
            f"{row['lut_bytes']/2**10:.1f} | {row['keygen_ms']/1000:.1f} | "
            f"{row['peak_rss_kib']/2**20:.2f} |"
        )
    lines += ["", "TFHE-rs 的 N=2/4 各穷举了 8/16 个输入组合；其余规模为抽样。",
              "TFHE-rs 表采用 *legacy WOPBS-only* 参数；其安全性未与 128-bit 主表",
              "同口径认证，且多输出 vertical packing 未测试，因此不能从两个表推导跨库加速倍数。",
              "key_bytes 是 WopbsKey 的 bincode 序列化大小（含内部 server keys），",
              "LUT 单列；固定权重、定点域与 OpenFHE 实验一致。", ""]
    lines += [
        "## 复用原语与边界", "",
        "| 源码 | 固定版本/许可 | 复用 API | 限制 |",
        "| --- | --- | --- | --- |",
        "| OpenFHE `thirdparty/openfhe/src` | v1.5.1 / `1306d14`, BSD-2-Clause | "
        "`EvalFunc`、`EvalBinGate(CMUX)`、`GetMaxPlaintextSpace` | "
        "一般 LUT 2 次 PBS；CMUX 内部 3 次 NAND/PBS；p <= q/256 |",
        "| Zama tfhe-rs（Cargo.lock） | `079f46d`, BSD-3-Clause-Clear | "
        "`WopbsKey::new_wopbs_key_only_for_wopbs`、`programmable_bootstrapping_without_padding` | "
        "`experimental` feature；legacy 参数安全级别待核验；vertical packing 未实测 |",
        "", "上述均调用官方提供的原语，未自行实现 PBS 内核。",
        "默认权重表仅有五种数值；对于重复权重，CMUX 缓存会复用布尔子表达式，",
        "因此成本不能外推到高精度真实模型权重。", "",
    ]
    base = output / "raw/n4_fused.json"
    if base.exists():
        measured = json.loads(base.read_text())
        lines += [
            "## 资源下界与失败边界", "",
            "任意 public scalar table 至少包含 N 个独立权重；本实验的 fused LUT",
            "需要容纳 4N 个 `(e,x)` 输入。OpenFHE v1.5.1 的 `GetMaxPlaintextSpace=q/256`，",
            "因此该直接编码至少需要 `p>=4N, q>=1024N`；若用负循环半域扩展，",
            "需要 `p>=8N, q>=2048N`。由 q=4096/8192 两点实测 key 大小几乎翻倍，",
            "按 ring dimension 线性外推（**非实测**）：", "",
            "| N | fused q 下限 | 估计 OpenFHE key (GiB) | 32 GiB 进程预算 |",
            "| ---: | ---: | ---: | --- |",
        ]
        for n in (16, 32, 64):
            q = 1024 * n
            estimated = measured["evaluation_key_bytes"] * q / measured["q"] / 2**30
            lines.append(f"| {n} | {q} | {estimated:.1f} | 超出 |")
        lines += [
            "", "这些估计仅适用于这里的 OpenFHE 直接联合编码，不能用于宣称 TFHE-rs",
            "也具有相同 key 增长。对任意权重矩阵，公开模型至少需存放",
            "`N*d_in*d_out*weight_bits/8` 字节；若逐元素/输出独立执行本标量原语，",
            "LUT/PBS 成本还会乘上元素/输出数。必须先实测一次选择的多输出复用，",
            "才有进入矩阵 Stage D 的依据。", "",
        ]
    custom = output / "custom_tables.json"
    if custom.exists():
        lines += ["## 自定义公开权重表", ""]
        for item in json.loads(custom.read_text()):
            if "error" in item:
                lines.append(f"- `{item['case']}`: 失败 `{item['error']}`")
            else:
                lines.append(f"- `{item['case']}` `[{item['weights']}]`: "
                             f"检查 {item['verified']}、错误 {item['failures']}。")
        lines += ["", "复现：`.venv/bin/python experiments/test_stage_b_tables.py`。", ""]
    probe = output / "cmux_probes.json"
    if probe.exists():
        lines += ["## 大 N 的低样本 CMUX 探针", "",
                  "主扫描的 4 次检查 + 1 次计时设为 360 秒/项；下表是独立运行的",
                  "1 次检查 + 1 次计时（500 秒/项），不能与多次重复的 median 同等解释。",
                  "", "| N | 验证/错误 | 单次在线 (ms) | PBS |",
                  "| ---: | ---: | ---: | ---: |"]
        for item in json.loads(probe.read_text()):
            lines.append(f"| {item['n']} | {item['verified']}/{item['failures']} | "
                         f"{item['median_ms']:.1f} | {item['pbs_per_eval']} |")
        errors_path = output / "cmux_probe_failures.json"
        if errors_path.exists():
            for item in json.loads(errors_path.read_text()):
                lines.append(f"- N={item['n']} 探针失败：`{item['error']}`")
        lines += ["", "复现：`.venv/bin/python experiments/probe_stage_b_cmux.py`。", ""]
    if failures:
        lines += ["TFHE-rs 失败项："] + [f"- `{item['case']}`: {item['error']}" for item in failures]
    lines += ["复现：`RUSTUP_TOOLCHAIN=1.91.1 CARGO_TARGET_DIR=$PWD/build/tfhe-target "
              "cargo build --release --manifest-path experiments/tfhe_stage_b/Cargo.toml`，",
              "然后运行 `.venv/bin/python experiments/run_stage_b_tfhe.py`。", ""]
    path.write_text(content.rstrip() + "\n\n" + "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage_b")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--memory-gib", type=int, default=32)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(output)
        return
    binary = ROOT / "build/tfhe-target/release/ciphermoe-stage-b-wopbs"
    results: list[dict] = []
    failures: list[dict] = []
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"

    def limit_memory() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib * 2**30,) * 2)

    for n in (2, 4, 8, 16, 32, 64):
        checks = 4 * n if n <= 4 else 8
        case = f"n{n}"
        print(f"running tfhe-rs {case}", flush=True)
        try:
            run = subprocess.run(
                [str(binary), str(n), str(checks), "2"], cwd=ROOT, env=env,
                text=True, capture_output=True, timeout=args.timeout,
                preexec_fn=limit_memory, check=False,
            )
            if run.returncode:
                raise RuntimeError(f"exit={run.returncode}: {run.stderr[-1500:]}")
            result = json.loads(run.stdout)
            results.append(result)
            (output / "raw" / f"tfhe_{case}.json").write_text(json.dumps(result, indent=2) + "\n")
        except (subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
            failures.append({"case": case, "error": str(error)[:1500]})
        (output / "tfhe_summary.json").write_text(json.dumps(results, indent=2) + "\n")
        (output / "tfhe_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    report(output)


if __name__ == "__main__":
    main()
