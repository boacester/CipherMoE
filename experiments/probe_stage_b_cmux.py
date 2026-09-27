#!/usr/bin/env python3
"""Small-sample CMUX probes for cases that exceed the main scan's time limit."""

import json
import os
import resource
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/stage_b"


def main() -> None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = str(ROOT / "thirdparty/openfhe/install/lib") + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )

    def limit_memory() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (32 * 2**30,) * 2)

    rows = []
    failures = []
    for n in (32, 64):
        print(f"running N={n} CMUX probe (one check + one sample)", flush=True)
        try:
            run = subprocess.run(
                [str(ROOT / "build/stage_b_scalar"), str(n), "cmux", "1", "1"],
                cwd=ROOT, env=env, text=True, capture_output=True,
                timeout=500, preexec_fn=limit_memory, check=False,
            )
            if run.returncode:
                raise RuntimeError(f"exit={run.returncode}: {run.stderr[-1000:]}")
            data = json.loads(run.stdout)["result"]
            rows.append(data)
            (OUTPUT / "raw" / f"n{n}_cmux_probe.json").write_text(json.dumps(data, indent=2) + "\n")
        except (subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
            failures.append({"n": n, "error": str(error)[:1000]})
        (OUTPUT / "cmux_probes.json").write_text(json.dumps(rows, indent=2) + "\n")
        (OUTPUT / "cmux_probe_failures.json").write_text(json.dumps(failures, indent=2) + "\n")


if __name__ == "__main__":
    main()
