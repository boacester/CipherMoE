#!/usr/bin/env python3
"""Check non-default public scalar tables without overwriting the main scan."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/stage_b/custom_tables.json"


def main() -> None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = str(ROOT / "thirdparty/openfhe/install/lib") + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    cases = [
        ("tfhe_wopbs_legacy", [
            str(ROOT / "build/tfhe-target/release/ciphermoe-stage-b-wopbs"),
            "4", "16", "1", "2,-1,-2,0",
        ]),
        ("openfhe_fused", [
            str(ROOT / "build/stage_b_scalar"), "2", "fused", "8", "1", "2,-1",
        ]),
        ("openfhe_cmux", [
            str(ROOT / "build/stage_b_scalar"), "2", "cmux", "8", "1", "2,-1",
        ]),
    ]
    records = []
    for name, command in cases:
        print(f"running custom table {name}", flush=True)
        try:
            run = subprocess.run(
                command, cwd=ROOT, env=env, text=True, capture_output=True,
                timeout=300, check=False,
            )
            if run.returncode:
                records.append({"case": name, "error": run.stderr.strip()[-800:]})
            else:
                result = json.loads(run.stdout)
                if "result" in result:
                    result = result["result"]
                records.append({"case": name, "weights": command[-1],
                                "verified": result["verified"], "failures": result["failures"]})
        except (subprocess.TimeoutExpired, ValueError) as error:
            records.append({"case": name, "error": str(error)[:800]})
        OUTPUT.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
