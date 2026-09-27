#!/usr/bin/env python3
"""Benchmark exact e-only public-vector selection, saving each finished size."""

import argparse
import json
import os
import resource
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/tfhe-target/release/p0_selector"
SIZES = (1, 4, 8, 16, 32, 64, 128, 256, 512)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/p0_selector")
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--memory-gib", type=int, default=32)
    parser.add_argument("--experts", type=int, nargs="*", default=[2, 8, 64])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--skip-custom", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    (output / "raw").mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps({
        "date": date.today().isoformat(),
        "tfhe_commit": "079f46d08ffb8763de54b7794e850ddbd98fe63b",
        "parameter": "V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128",
        "expert_counts": args.experts, "sizes": SIZES, "repeats": args.repeats,
        "memory_gib": args.memory_gib, "timeout_s_per_process": args.timeout,
        "threads": 1, "weight_values": "u4 unsigned, range 0..15",
        "input": "only Enc(e), public arbitrary weight vectors; no activation",
    }, indent=2) + "\n")
    rows: list[dict] = []
    failures: list[dict] = []
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"

    def limit_memory() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib * 2**30,) * 2)

    def save() -> None:
        rows.sort(key=lambda row: (row["n"], row["m"], row["custom_weights"]))
        (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")

    cases: list[tuple[int, Path | None]] = [(n, None) for n in args.experts]
    if not args.skip_custom:
        cases.append((2, ROOT / "experiments/p0_custom_vectors.json"))
    for n, custom in cases:
        label = f"n{n}" + ("_custom" if custom else "")
        print(f"P0: {label}", flush=True)
        command = [str(BINARY), str(n), str(args.repeats)]
        if custom:
            command.append(str(custom))
        stdout: str | bytes = ""
        error = ""
        try:
            run = subprocess.run(
                command, cwd=ROOT, env=env, capture_output=True, text=True,
                check=False, timeout=args.timeout, preexec_fn=limit_memory,
            )
            stdout = run.stdout
            if run.returncode:
                error = f"exit={run.returncode}: {run.stderr[-1800:]}"
        except subprocess.TimeoutExpired as expired:
            stdout = expired.stdout or b""
            error = f"timeout after {args.timeout}s"
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        for line in stdout.splitlines():
            try:
                row = json.loads(line)
                if row["n"] != n or row["failures"] != 0:
                    raise ValueError("invalid N or incorrect output")
            except (ValueError, KeyError) as exc:
                failures.append({"case": label, "error": f"invalid result: {exc}", "line": line[:300]})
                continue
            rows.append(row)
            name = f"{label}_m{row['m']}"
            (output / "raw" / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")
            print(f"  m={row['m']}: {row['online_median_ms']:.1f} ms, {row['blind_rotations']} rotations", flush=True)
            save()
        if error:
            failures.append({"case": label, "error": error})
        if not custom and not error and len([row for row in rows if row["n"] == n and not row["custom_weights"]]) != len(SIZES):
            failures.append({"case": label, "error": "process returned fewer than nine sizes"})
        save()


if __name__ == "__main__":
    main()
