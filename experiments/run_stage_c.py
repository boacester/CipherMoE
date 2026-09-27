#!/usr/bin/env python3
"""Measure reusable same-input LUT outputs with pinned TFHE-rs parameters."""

import argparse
import json
import os
import resource
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--memory-gib", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", type=str, help="rerun a single N,m case, preserving other results")
    args = parser.parse_args()
    output = ROOT / "results/stage_c"
    (output / "raw").mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps({
        "date": date.today().isoformat(), "tfhe_commit": "079f46d08ffb8763de54b7794e850ddbd98fe63b",
        "parameter": "V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128",
        "threads": 1, "timeout_s": args.timeout, "memory_gib": args.memory_gib,
        "repeats": args.repeats, "cases": [[n, m] for n in (2, 4, 8) for m in (1, 2, 4, 8) if 4*n*m <= 64],
    }, indent=2) + "\n")
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"

    def limit_memory() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib * 2**30,) * 2)

    rows = json.loads((output / "summary.json").read_text()) if args.only else []
    failures = json.loads((output / "failures.json").read_text()) if args.only else []
    only_case = tuple(map(int, args.only.split(","))) if args.only else None
    if only_case is not None and only_case not in {(n, m) for n in (2, 4, 8) for m in (1, 2, 4, 8) if 4*n*m <= 64}:
        parser.error("--only must be an allowed N,m case")
    for n in (2, 4, 8):
        for m in (1, 2, 4, 8):
            if 4*n*m > 64 or (only_case is not None and only_case != (n, m)):
                continue
            name = f"n{n}_m{m}"
            print(f"stage C: {name}", flush=True)
            rows = [row for row in rows if (row["n"], row["m"]) != (n, m)]
            failures = [row for row in failures if row["case"] != name]
            try:
                result = subprocess.run(
                    [str(ROOT / "build/tfhe-target/release/stage_c"), str(n), str(m), str(args.repeats)],
                    cwd=ROOT, env=env, capture_output=True, text=True, check=False,
                    timeout=args.timeout, preexec_fn=limit_memory,
                )
                if result.returncode:
                    raise RuntimeError(f"exit {result.returncode}: {result.stderr[-1600:]}")
                row = json.loads(result.stdout)
                if row["failures"]:
                    raise RuntimeError(f"{row['failures']} incorrect outputs")
                rows.append(row)
                rows.sort(key=lambda row: (row["n"], row["m"]))
                (output / "raw" / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")
            except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                failures.append({"case": name, "error": str(error)[:1800]})
            (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")


if __name__ == "__main__":
    main()
