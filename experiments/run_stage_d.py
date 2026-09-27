#!/usr/bin/env python3
"""Tiny rectangular matrix/tile comparisons, not a model-scale claim."""

import json
import os
import resource
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/stage_d"
CASES = [
    (2, 3, 2, 32, None), (3, 4, 2, 16, None), (3, 4, 4, 16, None),
    (2, 3, 2, 16, "1,0,-1,1,0,1,0,-1,1,-1,1,0"),
]


def limit_memory() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (32 * 2**30,) * 2)


def main() -> None:
    (OUTPUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "date": date.today().isoformat(), "tfhe_commit": "079f46d08ffb8763de54b7794e850ddbd98fe63b",
        "parameter": "V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128",
        "cases_d_in_d_out_tile_checks_weights": CASES, "memory_gib": 32, "timeout_s": 600, "threads": 1,
    }, indent=2) + "\n")
    rows, failures = [], []
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    for d_in, d_out, tile, checks, weights in CASES:
        name = f"in{d_in}_out{d_out}_tile{tile}" + ("_custom" if weights else "")
        print(f"stage D: {name}", flush=True)
        try:
            run = subprocess.run(
                [str(ROOT / "build/tfhe-target/release/stage_d"), str(d_in), str(d_out), str(tile), str(checks)]
                + ([weights] if weights else []),
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
                preexec_fn=limit_memory, check=False,
            )
            if run.returncode:
                raise RuntimeError(f"exit {run.returncode}: {run.stderr[-1600:]}")
            row = json.loads(run.stdout)
            if row["failures"]:
                raise RuntimeError(f"{row['failures']} incorrect outputs")
            rows.append(row)
            (OUTPUT / "raw" / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            failures.append({"case": name, "error": str(error)[:1800]})
        (OUTPUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        (OUTPUT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")


if __name__ == "__main__":
    main()
