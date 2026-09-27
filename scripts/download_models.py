#!/usr/bin/env python3
"""Download and cryptographically validate revision-pinned stage-A checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "stage_a_models.json"
STOP_REQUESTED = threading.Event()


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def segment_bounds(total: int, count: int, index: int) -> tuple[int, int]:
    start = total * index // count
    end = total * (index + 1) // count - 1
    return start, end


def segment_paths(
    destination: Path, shard: dict, segment_count: int
) -> list[Path]:
    segment_dir = destination / ".segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    return [
        segment_dir / f"{shard['name']}.segments{segment_count}.part{index:03d}"
        for index in range(segment_count)
    ]


def prepare_segments(destination: Path, shard: dict, segment_count: int) -> bool:
    target = destination / shard["name"]
    if target.exists() and target.stat().st_size == shard["bytes"]:
        print(f"Already complete by size: {target.name}", flush=True)
        return False
    parts = segment_paths(destination, shard, segment_count)
    if target.exists():
        if any(path.exists() for path in parts):
            raise RuntimeError(f"Both partial target and segments exist for {target}")
        first_start, first_end = segment_bounds(shard["bytes"], segment_count, 0)
        if target.stat().st_size > first_end - first_start + 1:
            raise RuntimeError(
                f"Existing prefix of {target} is too large for {segment_count} segments"
            )
        target.replace(parts[0])
    return True


def download_modelscope_segment(
    entry: dict,
    shard: dict,
    destination: Path,
    segment_count: int,
    segment_index: int,
) -> None:
    start, end = segment_bounds(shard["bytes"], segment_count, segment_index)
    expected = end - start + 1
    part = segment_paths(destination, shard, segment_count)[segment_index]
    url = (
        f"https://modelscope.cn/models/{entry['modelscope_repo_id']}/resolve/"
        f"{entry['modelscope_revision']}/{shard['name']}"
    )
    for attempt in range(1, 21):
        if STOP_REQUESTED.is_set():
            raise InterruptedError("Download interrupted; segment state is resumable")
        present = part.stat().st_size if part.exists() else 0
        if present == expected:
            return
        if present > expected:
            raise RuntimeError(f"Oversized segment {part}: {present}, expected {expected}")
        with part.open("ab") as output:
            result = subprocess.run(
                [
                    "curl",
                    "--location",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "60",
                    "--range",
                    f"{start + present}-{end}",
                    url,
                ],
                stdout=output,
                check=False,
            )
        if STOP_REQUESTED.is_set():
            raise InterruptedError("Download interrupted; segment state is resumable")
        current = part.stat().st_size
        if current > expected:
            raise RuntimeError(
                f"Server ignored byte range for {part}: {current}, expected {expected}"
            )
        if current == expected:
            return
        print(
            f"Retry {attempt}/20 for {shard['name']} segment {segment_index}: "
            f"curl={result.returncode}, bytes={current}/{expected}",
            flush=True,
        )
    raise RuntimeError(f"Could not complete {part} after 20 attempts")


def assemble_shard(destination: Path, shard: dict, segment_count: int) -> None:
    target = destination / shard["name"]
    if target.exists() and target.stat().st_size == shard["bytes"]:
        return
    parts = segment_paths(destination, shard, segment_count)
    for index, part in enumerate(parts):
        start, end = segment_bounds(shard["bytes"], segment_count, index)
        expected = end - start + 1
        if not part.exists() or part.stat().st_size != expected:
            raise RuntimeError(f"Incomplete segment {part}; expected {expected} bytes")
    assembling = target.with_name(target.name + ".assembling")
    with assembling.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if assembling.stat().st_size != shard["bytes"]:
        raise RuntimeError(f"Assembled size mismatch for {assembling}")
    assembling.replace(target)
    for part in parts:
        part.unlink()
    print(f"Assembled {target.name}", flush=True)


def download_from_modelscope(
    entry: dict, destination: Path, max_workers: int, segment_count: int
) -> None:
    if shutil.which("curl") is None:
        raise RuntimeError("ModelScope resumable download requires curl")
    # Keep tokenizer/config/code pinned to the canonical Hugging Face revision.
    snapshot_download(
        repo_id=entry["repo_id"],
        revision=entry["revision"],
        local_dir=destination,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth"],
        max_workers=max_workers,
    )
    pending = [
        shard
        for shard in entry["shards"]
        if prepare_segments(destination, shard, segment_count)
    ]
    STOP_REQUESTED.clear()
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(
            download_modelscope_segment,
            entry,
            shard,
            destination,
            segment_count,
            segment_index,
        ): (shard, segment_index)
        for shard in pending
        for segment_index in range(segment_count)
    }
    try:
        for future in as_completed(futures):
            shard, segment_index = futures[future]
            future.result()
            print(
                f"Downloaded {entry['name']}/{shard['name']} "
                f"segment {segment_index + 1}/{segment_count}",
                flush=True,
            )
    except KeyboardInterrupt:
        STOP_REQUESTED.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    for shard in pending:
        assemble_shard(destination, shard, segment_count)


def validate(entry: dict, destination: Path) -> None:
    expected_names = {shard["name"] for shard in entry["shards"]}
    actual_names = {path.name for path in destination.glob("*.safetensors")}
    if actual_names != expected_names:
        raise RuntimeError(
            f"Checkpoint shard set mismatch for {entry['name']}: "
            f"actual={sorted(actual_names)}, expected={sorted(expected_names)}"
        )
    total_bytes = 0
    for shard in entry["shards"]:
        path = destination / shard["name"]
        size = path.stat().st_size
        if size != shard["bytes"]:
            raise RuntimeError(
                f"Size mismatch for {path}: {size}, expected {shard['bytes']}"
            )
        actual_hash = digest(path)
        if actual_hash != shard["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: {actual_hash}, expected {shard['sha256']}"
            )
        total_bytes += size
        print(f"Verified {entry['name']}/{path.name}", flush=True)
    if len(actual_names) != entry["expected_shards"] or total_bytes != entry["expected_weight_bytes"]:
        raise RuntimeError(
            f"Checkpoint validation failed for {entry['name']}: "
            f"{len(actual_names)} shards/{total_bytes} bytes, expected "
            f"{entry['expected_shards']} shards/{entry['expected_weight_bytes']} bytes"
        )
    print(f"Validated {len(actual_names)} shards and {total_bytes} weight bytes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        help="Registry model name to download; repeat as needed (default: all).",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--segments-per-shard",
        type=int,
        default=1,
        help="ModelScope byte ranges per shard; max-workers caps global concurrency.",
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
        help="Weight source. Both are checked against the same pinned SHA-256 values.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; validate all selected local shards.",
    )
    args = parser.parse_args()
    if args.max_workers < 1 or args.segments_per_shard < 1:
        raise SystemExit("--max-workers and --segments-per-shard must be positive")

    entries = json.loads(REGISTRY.read_text())["models"]
    selected = set(args.model or [entry["name"] for entry in entries])
    unknown = selected - {entry["name"] for entry in entries}
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(sorted(unknown))}")

    for entry in entries:
        if entry["name"] not in selected:
            continue
        destination = ROOT / entry["local_dir"]
        destination.mkdir(parents=True, exist_ok=True)
        if not args.verify_only:
            if args.source == "huggingface":
                print(
                    f"Downloading {entry['repo_id']}@{entry['revision']} -> {destination}",
                    flush=True,
                )
                snapshot_download(
                    repo_id=entry["repo_id"],
                    revision=entry["revision"],
                    local_dir=destination,
                    max_workers=args.max_workers,
                )
            else:
                print(
                    f"Downloading weights from ModelScope "
                    f"{entry['modelscope_repo_id']}@{entry['modelscope_revision']} -> {destination}",
                    flush=True,
                )
                download_from_modelscope(
                    entry, destination, args.max_workers, args.segments_per_shard
                )
        validate(entry, destination)


if __name__ == "__main__":
    main()
