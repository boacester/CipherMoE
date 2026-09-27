#!/usr/bin/env python3
"""Resource-bounded Stage B experiments, preserving successes and failures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage_b")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--memory-gib", type=int, default=32)
    parser.add_argument("--cases", nargs="*", help="N:method:checks:repeats")
    args = parser.parse_args()
    cases = args.cases or [
        "2:fused:8:2", "2:negacyclic:8:2", "2:cmux:8:2",
        "4:fused:16:2", "4:negacyclic:16:2", "4:cmux:4:2",
        "8:fused:8:2", "8:cmux:4:1", "16:cmux:4:1",
        "32:cmux:4:1", "64:cmux:4:1",
    ]
    output = args.output.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = str(ROOT / "thirdparty/openfhe/install/lib") + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )

    def limit_memory() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib * 2**30,) * 2)

    rows: list[dict] = []
    failures: list[dict] = []
    for case in cases:
        n, method, checks, repeats = case.split(":")
        command = [str(ROOT / "build/stage_b_scalar"), n, method, checks, repeats]
        print(f"running {case}", flush=True)
        try:
            run = subprocess.run(
                command, cwd=ROOT, env=env, text=True, capture_output=True,
                timeout=args.timeout, preexec_fn=limit_memory, check=False,
            )
            if run.returncode:
                raise RuntimeError(f"exit={run.returncode}; stderr={run.stderr[-2000:]}")
            result = json.loads(run.stdout)["result"]
            (raw / f"n{n}_{method}.json").write_text(json.dumps(result, indent=2) + "\n")
            rows.append(result)
            if result["failures"]:
                failures.append({"case": case, "error": f"{result['failures']} wrong answers"})
        except (subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError) as error:
            failures.append({"case": case, "error": str(error)[:2000]})
        (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    if rows:
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[k for k in rows[0] if k != "online_ms"])
            writer.writeheader()
            writer.writerows({k: v for k, v in row.items() if k != "online_ms"} for row in rows)

    cpu = next((line.split(":", 1)[1].strip() for line in
                Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": cpu,
        "openfhe_version": "v1.5.1 / 1306d14f8c26bb6150d3e6ad54f28dfe1007689e",
        "security": "STD128, GINX, HEStd_128_classic ring parameter selection",
        "omp_threads": 1,
        "timeout_seconds_per_case": args.timeout,
        "memory_limit_gib_per_case": args.memory_gib,
        "cases": cases,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = [
        "# Stage B: multi-bit exact scalar FHEW（有限资源扫描）", "",
        "## 口径", "",
        "OpenFHE v1.5.1, STD128/GINX；单线程；key 和 model 是服务器公开参数，",
        "只有 ID、activation 和输出是密文。ID 采用 ceil(log2 N) bits，",
        "N 均为 2 的幂，因此该位宽没有非法码点；外部非法 ID 应在客户端加密前拒绝，",
        "服务器不能对加密 ID 做明文合法性分支。", "",
        "x 的 2-bit 定点编码为 code=0..3，对应 x=(code-2)/2；",
        "w[e] 是 [-2,2] 中由公开公式 ((7e+3)%5)-2 得到的整数，",
        "程序也接受用户提供的任意 [-2,2] 整数表；默认表仅有五种权重，",
        "大 N 时存在重复，CMUX 的子表达式复用收益不能外推到高精度权重。",
        "即结果的实数值为 (w[e]*(code-2))/2。为避免混淆，原始整数乘积",
        "保存在 Z_16 中并以 [-8,7] 补码解码；所测输入下范围 [-4,4]，",
        "没有溢出。Fused 直接在 Z_p 做多位联合编码 (4e+code)；",
        "negacyclic 在 LUT 内使用 +5 offset 并在解密后去除；CMUX 输出 4 个 bit，",
        "按 Z_16 解码。输入加密与解密不计入 online_ms；PBS 数据 OpenFHE 源码统计。", "",
        "## 实测", "",
        "| N | 路径 | p / ring | 检查/错误 | online median (ms) | PBS/CMUX | key (GiB) | LUT (KiB) | setup (s) | peak RSS (GiB) |",
        "| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['n']} | {row['method']} | {row['p']}/{row['ring_dim']} | "
            f"{row['verified']}/{row['failures']} | {row['median_ms']:.1f} | "
            f"{row['pbs_per_eval']}/{row['cmux_calls_per_eval']} | "
            f"{row['evaluation_key_bytes']/2**30:.2f} | "
            f"{row['lut_bytes']/2**10:.1f} | {row['keygen_ms']/1000:.1f} | "
            f"{row['peak_rss_kib']/2**20:.2f} |"
        )
    lines += ["", "## 未完成 / 失败项", ""]
    lines += [f"- `{item['case']}`: {item['error']}" for item in failures]
    lines += [
        "- 以上只有 N=2 所有路径以及 N=4 fused/negacyclic（若列出）做了完整枚举；",
        "  其他为抽样，无穷举正确性结论。",
        "- TFHE-rs WOPBS / vertical packing 尚未集成或实测，不能写成优于/劣于 OpenFHE。",
        "- Fused 和 CMUX 的 ring dimension 不同；本表可比安全级别与输入输出语义，",
        "  但延迟不能被解释成同一参数下的纯算法加速比。",
        "- CMUX 当前是输出 bit 的 Boolean truth-table/CMUX tree；它使用 N 个公开",
        "  leaf functions 并对相同子表达式做缓存，不是可扩展的矩阵方案。",
        "- CMUX 由客户端分别加密的 ID bits 与 x bits 开始（输入 IDbits+2 个密文），",
        "  不包括由单个 `Enc(e)` 同态分解 ID 的代价；输出 4 个 bit 密文，而 LUT",
        "  输入 2 个标量密文、输出 1 个标量密文。这不是完全同口径的接口比较。",
        "- `evaluation_key_bytes` 为 refresh + switching keys 的 portable binary 序列化大小；",
        "  LUT 是 `q * sizeof(NativeInteger)`，不是 evaluation key；peak RSS 是进程高水位。",
        "", "## 阶段决策", "",
        "当前仅证明部分 exact 标量原语可计算；没有完成 TFHE-rs 对照和所有 N 的穷举。",
        "直接联合 LUT 的明文空间至少为 N*4，要求 ring dimension >= 256*N*4；",
        "依赖半域 negacyclic 时还要翻倍。Key storage 与 setup 已成为矩阵扩展的关键障碍，",
        "不能将本结果宣称为完整 SelectApply 或 Stage C 的 go。先评估任意权重矩阵",
        "所需 public data / LUT / evaluation-key 信息下界，再验证复用选择的接口。", "",
        "## 复现", "",
        "```bash", "cmake --build build --target stage_b_scalar -j8",
        ".venv/bin/python experiments/run_stage_b.py", "```", "",
        "原始配置见 `metadata.json`，每项原始结果见 `raw/`，失败记录见 `failures.json`。", "",
    ]
    (output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
