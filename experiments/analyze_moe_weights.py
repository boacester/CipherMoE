#!/usr/bin/env python3
"""Stage A: analyze cross-expert weight structure in real MoE checkpoints.

The script treats the flattened weights of one layer/projection as an N x P
matrix.  It computes the exact N x N Gram matrix, so the singular spectrum and
best cross-expert rank-r reconstruction error are exact up to float32 dot
products without retaining the whole checkpoint in RAM.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "stage_a_models.json"
EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
DTYPES = {
    "F64": ("<f8", 8),
    "F32": ("<f4", 4),
    "F16": ("<f2", 2),
    "BF16": ("<u2", 2),
    "I64": ("<i8", 8),
    "I32": ("<i4", 4),
    "I16": ("<i2", 2),
    "I8": ("i1", 1),
    "U8": ("u1", 1),
    "BOOL": ("?", 1),
}


@dataclass(frozen=True)
class TensorInfo:
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


class SafeTensorStore:
    """Minimal read-only safetensors reader with explicit BF16 conversion."""

    def __init__(self, model_dir: Path):
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing {index_path}; checkpoint is incomplete")
        index = json.loads(index_path.read_text())
        self.model_dir = model_dir
        self.weight_map: dict[str, str] = index["weight_map"]
        self._headers: dict[Path, tuple[int, dict]] = {}

    def required_shards(self) -> list[Path]:
        return sorted({self.model_dir / name for name in self.weight_map.values()})

    def validate(self) -> None:
        missing = [path for path in self.required_shards() if not path.exists()]
        if missing:
            names = ", ".join(path.name for path in missing)
            raise FileNotFoundError(f"Missing checkpoint shard(s): {names}")

    def _header(self, shard: Path) -> tuple[int, dict]:
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        with shard.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"Invalid safetensors header in {shard}")
            header_length = struct.unpack("<Q", length_bytes)[0]
            header = json.loads(handle.read(header_length))
        value = (8 + header_length, header)
        self._headers[shard] = value
        return value

    def info(self, name: str) -> TensorInfo:
        shard = self.model_dir / self.weight_map[name]
        data_start, header = self._header(shard)
        record = header[name]
        begin, end = record["data_offsets"]
        dtype = record["dtype"]
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype {dtype} for {name}")
        return TensorInfo(
            shard=shard,
            dtype=dtype,
            shape=tuple(record["shape"]),
            offset=data_start + begin,
            nbytes=end - begin,
        )

    def read_float32(self, name: str, indices: np.ndarray | None = None) -> np.ndarray:
        info = self.info(name)
        np_dtype, itemsize = DTYPES[info.dtype]
        count = math.prod(info.shape)
        if count * itemsize != info.nbytes:
            raise ValueError(f"Inconsistent tensor size for {name}")
        raw = np.memmap(
            info.shard,
            mode="r",
            dtype=np_dtype,
            offset=info.offset,
            shape=(count,),
        )
        selected = raw if indices is None else raw[indices]
        if info.dtype == "BF16":
            bits = np.asarray(selected, dtype=np.uint16).astype(np.uint32)
            values = np.left_shift(bits, 16).view(np.float32)
        else:
            values = np.asarray(selected, dtype=np.float32)
        return values.copy()


def discover_expert_groups(store: SafeTensorStore) -> dict[tuple[int, str], dict[int, str]]:
    groups: dict[tuple[int, str], dict[int, str]] = {}
    for name in store.weight_map:
        match = EXPERT_PATTERN.match(name)
        if match is None:
            continue
        key = (int(match["layer"]), match["projection"])
        groups.setdefault(key, {})[int(match["expert"])] = name
    if not groups:
        raise ValueError(f"No routed expert tensors found in {store.model_dir}")
    return groups


def eigendecomposition(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram = (gram + gram.T) * 0.5
    values, vectors = np.linalg.eigh(gram.astype(np.float64))
    values = values[::-1]
    vectors = vectors[:, ::-1]
    tolerance = max(float(values[0]), 1.0) * 1e-10
    values[(values < 0.0) & (values > -tolerance)] = 0.0
    if np.any(values < 0.0):
        raise ValueError(f"Gram matrix has a materially negative eigenvalue: {values[-1]}")
    return values, vectors


def eigenvalues(gram: np.ndarray) -> np.ndarray:
    return eigendecomposition(gram)[0]


def relative_errors(values: np.ndarray, total: float) -> list[float]:
    retained = np.cumsum(values, dtype=np.float64)
    residual = np.maximum(total - retained, 0.0)
    return np.sqrt(residual / total).tolist()


def output_errors(vectors: np.ndarray, output_gram: np.ndarray) -> list[float]:
    total = float(np.trace(output_gram))
    captured = np.sum(vectors * (output_gram @ vectors), axis=0)
    residual = np.maximum(total - np.cumsum(captured, dtype=np.float64), 0.0)
    return np.sqrt(residual / total).tolist()


def grouped_decomposition(
    gram: np.ndarray, group_count: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    count = gram.shape[0]
    norms = np.sqrt(np.maximum(np.diag(gram), np.finfo(np.float64).tiny))
    cosine = gram / np.outer(norms, norms)
    distance = np.clip(1.0 - cosine, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(
        linkage(squareform(distance, checks=False), method="average"),
        t=min(group_count, count),
        criterion="maxclust",
    )
    spectrum_parts = []
    vector_parts = []
    sizes = []
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        sizes.append(int(len(members)))
        values, local_vectors = eigendecomposition(gram[np.ix_(members, members)])
        vectors = np.zeros((count, len(members)), dtype=np.float64)
        vectors[members] = local_vectors
        spectrum_parts.append(values)
        vector_parts.append(vectors)
    values = np.concatenate(spectrum_parts)
    vectors = np.concatenate(vector_parts, axis=1)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order], sizes


def analyze_gram(
    gram: np.ndarray,
    group_counts: Iterable[int],
    output_gram: np.ndarray | None = None,
) -> dict:
    count = gram.shape[0]
    total = float(np.trace(gram))
    global_values, global_vectors = eigendecomposition(gram)
    methods: dict[str, dict] = {
        "global": {
            "relative_weight_error": relative_errors(global_values, total),
            "eigenvalues": global_values.tolist(),
        }
    }

    centering = np.eye(count) - np.ones((count, count)) / count
    centered_values, centered_vectors = eigendecomposition(centering @ gram @ centering)
    mean_energy = float(np.sum(gram) / count)
    shared_captured = mean_energy + np.concatenate(
        ([0.0], np.cumsum(centered_values, dtype=np.float64))
    )
    shared_captured = shared_captured[:count]
    methods["shared_plus_delta"] = {
        "relative_weight_error": np.sqrt(
            np.maximum(total - shared_captured, 0.0) / total
        ).tolist(),
        "mean_energy_fraction": mean_energy / total,
        "delta_eigenvalues": centered_values.tolist(),
    }

    shared_vectors = np.column_stack(
        (np.ones(count, dtype=np.float64) / math.sqrt(count), centered_vectors[:, : count - 1])
    )

    for group_count in group_counts:
        values, vectors, sizes = grouped_decomposition(gram, group_count)
        methods[f"grouped_g{group_count}"] = {
            "relative_weight_error": relative_errors(values, total),
            "eigenvalues": values.tolist(),
            "group_sizes": sizes,
        }
        if output_gram is not None:
            methods[f"grouped_g{group_count}"]["relative_output_error"] = output_errors(
                vectors, output_gram
            )
    if output_gram is not None:
        methods["global"]["relative_output_error"] = output_errors(
            global_vectors, output_gram
        )
        methods["shared_plus_delta"]["relative_output_error"] = output_errors(
            shared_vectors, output_gram
        )
    return methods


def load_existing(path: Path) -> set[tuple[str, int, str]]:
    completed: set[tuple[str, int, str]] = set()
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add((row["model"], row["layer"], row["projection"]))
    return completed


def write_summary(raw_path: Path, output_dir: Path) -> None:
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    summary_path = output_dir / "rank_thresholds.csv"
    thresholds = (0.10, 0.05, 0.01)
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "model", "layer", "projection", "method", "experts", "out_features",
            "in_features", "features_used", "features_total", "exact", "seconds",
            "rank_at_10pct", "rank_at_5pct", "rank_at_1pct",
            "output_rank_at_10pct", "output_rank_at_5pct", "output_rank_at_1pct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for method, result in row["methods"].items():
                errors = result["relative_weight_error"]
                ranks = []
                for threshold in thresholds:
                    ranks.append(next((i + 1 for i, error in enumerate(errors) if error <= threshold), ""))
                output_errors_for_method = result.get("relative_output_error", [])
                output_ranks = []
                for threshold in thresholds:
                    output_ranks.append(next(
                        (
                            i + 1 for i, error in enumerate(output_errors_for_method)
                            if error <= threshold
                        ),
                        "",
                    ))
                writer.writerow({
                    "model": row["model"],
                    "layer": row["layer"],
                    "projection": row["projection"],
                    "method": method,
                    "experts": row["experts"],
                    "out_features": row["shape"][0],
                    "in_features": row["shape"][1],
                    "features_used": row["features_used"],
                    "features_total": row["features_total"],
                    "exact": row["exact"],
                    "seconds": f"{row['seconds']:.3f}",
                    "rank_at_10pct": ranks[0],
                    "rank_at_5pct": ranks[1],
                    "rank_at_1pct": ranks[2],
                    "output_rank_at_10pct": output_ranks[0],
                    "output_rank_at_5pct": output_ranks[1],
                    "output_rank_at_1pct": output_ranks[2],
                })

    models = sorted({row["model"] for row in rows})
    projections = ("gate_proj", "up_proj", "down_proj")
    colors = {
        "global": "#1f77b4",
        "shared_plus_delta": "#ff7f0e",
        "grouped_g4": "#2ca02c",
    }

    def plot_metric(metric: str, ylabel: str, filename: str) -> None:
        fig, axes = plt.subplots(
            len(models), 3, figsize=(13, 3.6 * len(models)), squeeze=False
        )
        plotted = False
        for model_index, model in enumerate(models):
            for projection_index, projection in enumerate(projections):
                axis = axes[model_index][projection_index]
                subset = [
                    row for row in rows
                    if row["model"] == model and row["projection"] == projection
                ]
                if not subset:
                    axis.set_visible(False)
                    continue
                methods = [
                    name for name in colors
                    if name in subset[0]["methods"]
                    and all(metric in row["methods"][name] for row in subset)
                ]
                if not methods:
                    axis.set_visible(False)
                    continue
                plotted = True
                for method in methods:
                    curves = np.asarray([
                        row["methods"][method][metric] for row in subset
                    ])
                    ranks = np.arange(1, curves.shape[1] + 1)
                    median = np.median(curves, axis=0)
                    low, high = np.quantile(curves, [0.1, 0.9], axis=0)
                    axis.plot(ranks, median, label=method, color=colors[method])
                    axis.fill_between(ranks, low, high, color=colors[method], alpha=0.15)
                axis.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
                axis.set_title(f"{model} / {projection}")
                axis.set_xlabel("total full-matrix basis budget")
                axis.set_ylabel(ylabel)
                axis.set_ylim(bottom=0.0)
                axis.grid(alpha=0.2)
                axis.legend(fontsize=8)
        if plotted:
            fig.tight_layout()
            fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    plot_metric("relative_weight_error", "relative weight error", "weight_error_curves.png")
    plot_metric("relative_output_error", "relative output error", "output_error_curves.png")

    metadata = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "records": len(rows),
        "exact_records": sum(bool(row["exact"]) for row in rows),
        "calibrated_records": sum(bool(row["calibration_samples"]) for row in rows),
        "analysis_seconds": sum(float(row["seconds"]) for row in rows),
        "models": {
            row["model"]: {"repo_id": row["repo_id"], "revision": row["revision"]}
            for row in rows
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="Registry model name; default: all")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stage_a")
    parser.add_argument("--groups", default="2,4,8")
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Uniformly sample this many flattened weights (0 means exact/all).",
    )
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="Directory containing <model>.npz captured by capture_moe_activations.py.",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text())["models"]
    requested = set(args.model or [entry["name"] for entry in registry])
    unknown = requested - {entry["name"] for entry in registry}
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(sorted(unknown))}")
    group_counts = [int(value) for value in args.groups.split(",") if value]

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "weight_structure.jsonl"
    completed = set() if args.no_resume else load_existing(raw_path)
    mode = "w" if args.no_resume else "a"
    rng = np.random.default_rng(args.seed)

    with raw_path.open(mode) as output:
        for entry in registry:
            if entry["name"] not in requested:
                continue
            model_dir = ROOT / entry["local_dir"]
            calibration = None
            if args.calibration_dir is not None:
                calibration_path = args.calibration_dir.resolve() / f"{entry['name']}.npz"
                if calibration_path.exists():
                    calibration = np.load(calibration_path)
                else:
                    print(f"warning: missing calibration file {calibration_path}")
            store = SafeTensorStore(model_dir)
            store.validate()
            groups = discover_expert_groups(store)
            moe_layers = {layer for layer, _ in groups}
            expected_group_count = entry["expected_moe_layers"] * 3
            if len(moe_layers) != entry["expected_moe_layers"] or len(groups) != expected_group_count:
                raise ValueError(
                    f"Unexpected MoE layer/projection count for {entry['name']}: "
                    f"{len(moe_layers)} layers and {len(groups)} groups"
                )
            if any(len(experts) != entry["expected_experts"] for experts in groups.values()):
                raise ValueError(f"Unexpected routed expert count for {entry['name']}")
            for (layer, projection), expert_names in sorted(groups.items()):
                key = (entry["name"], layer, projection)
                if key in completed:
                    print(f"skip {key}: already present")
                    continue
                expert_ids = sorted(expert_names)
                if expert_ids != list(range(len(expert_ids))):
                    raise ValueError(f"Non-contiguous expert IDs for {key}: {expert_ids}")
                info = store.info(expert_names[expert_ids[0]])
                shape = info.shape
                if len(shape) != 2:
                    raise ValueError(f"Expected matrix tensor for {key}, got {shape}")
                total_features = math.prod(shape)
                feature_indices = None
                if args.max_features and args.max_features < total_features:
                    feature_indices = np.sort(
                        rng.choice(total_features, size=args.max_features, replace=False)
                    )
                used_features = total_features if feature_indices is None else len(feature_indices)
                print(
                    f"analyze {entry['name']} layer={layer} projection={projection} "
                    f"experts={len(expert_ids)} features={used_features}/{total_features}",
                    flush=True,
                )
                started = time.perf_counter()
                weights = np.empty((len(expert_ids), used_features), dtype=np.float32)
                for row_index, expert_id in enumerate(expert_ids):
                    name = expert_names[expert_id]
                    if store.info(name).shape != shape:
                        raise ValueError(f"Mismatched shape for {name}")
                    weights[row_index] = store.read_float32(name, feature_indices)
                gram = weights @ weights.T
                if feature_indices is not None:
                    gram *= total_features / used_features
                output_gram = None
                calibration_samples = 0
                if calibration is not None and feature_indices is None:
                    calibration_key = (
                        f"mlp_{layer}" if projection in {"gate_proj", "up_proj"}
                        else f"down_{layer}"
                    )
                    if calibration_key in calibration:
                        inputs = np.asarray(calibration[calibration_key], dtype=np.float32)
                        if inputs.ndim != 2 or inputs.shape[1] != shape[1]:
                            raise ValueError(
                                f"Calibration shape {inputs.shape} does not match {key} {shape}"
                            )
                        matrices = weights.reshape(len(expert_ids), *shape)
                        outputs = matrices @ inputs.T
                        flattened_outputs = outputs.reshape(len(expert_ids), -1)
                        output_gram = flattened_outputs @ flattened_outputs.T
                        calibration_samples = int(inputs.shape[0])
                        del matrices, outputs, flattened_outputs, inputs
                del weights
                methods = analyze_gram(
                    gram.astype(np.float64),
                    group_counts,
                    None if output_gram is None else output_gram.astype(np.float64),
                )
                elapsed = time.perf_counter() - started
                record = {
                    "model": entry["name"],
                    "repo_id": entry["repo_id"],
                    "revision": entry["revision"],
                    "layer": layer,
                    "projection": projection,
                    "experts": len(expert_ids),
                    "shape": list(shape),
                    "dtype": info.dtype,
                    "features_used": used_features,
                    "features_total": total_features,
                    "exact": feature_indices is None,
                    "calibration_samples": calibration_samples,
                    "seconds": elapsed,
                    "methods": methods,
                }
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                output.flush()
                completed.add(key)
                print(f"done {key} in {elapsed:.1f}s", flush=True)

    write_summary(raw_path, output_dir)


if __name__ == "__main__":
    main()
