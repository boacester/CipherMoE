#!/usr/bin/env python3
"""Download the revision-pinned MoE checkpoints used by stage A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "stage_a_models.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        help="Registry model name to download; repeat as needed (default: all).",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    entries = json.loads(REGISTRY.read_text())["models"]
    selected = set(args.model or [entry["name"] for entry in entries])
    unknown = selected - {entry["name"] for entry in entries}
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(sorted(unknown))}")

    for entry in entries:
        if entry["name"] not in selected:
            continue
        destination = ROOT / entry["local_dir"]
        print(f"Downloading {entry['repo_id']}@{entry['revision']} -> {destination}")
        snapshot_download(
            repo_id=entry["repo_id"],
            revision=entry["revision"],
            local_dir=destination,
            max_workers=args.max_workers,
        )
        shards = sorted(destination.glob("*.safetensors"))
        total_bytes = sum(path.stat().st_size for path in shards)
        if len(shards) != entry["expected_shards"] or total_bytes != entry["expected_weight_bytes"]:
            raise RuntimeError(
                f"Checkpoint validation failed for {entry['name']}: "
                f"{len(shards)} shards/{total_bytes} bytes, expected "
                f"{entry['expected_shards']} shards/{entry['expected_weight_bytes']} bytes"
            )
        print(f"Validated {len(shards)} shards and {total_bytes} weight bytes")


if __name__ == "__main__":
    main()
