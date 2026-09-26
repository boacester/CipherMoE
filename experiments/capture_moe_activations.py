#!/usr/bin/env python3
"""Capture small, deterministic calibration input sets for each MoE projection."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "stage_a_models.json"
DEFAULT_TEXT = ROOT / "experiments" / "calibration_text.txt"


class Reservoir:
    def __init__(self, size: int, seed: int):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.rows: list[np.ndarray] = []
        self.seen = 0

    def add(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().reshape(-1, tensor.shape[-1]).float().cpu().numpy()
        for row in values:
            self.seen += 1
            if len(self.rows) < self.size:
                self.rows.append(row.copy())
                continue
            replace = int(self.rng.integers(0, self.seen))
            if replace < self.size:
                self.rows[replace] = row.copy()

    def array(self) -> np.ndarray:
        if not self.rows:
            raise RuntimeError("Calibration hook captured no rows")
        return np.stack(self.rows).astype(np.float32, copy=False)


def find_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return list(layers)
    raise RuntimeError("Could not locate transformer layers")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="Registry model name; default: all")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stage_a" / "calibration")
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--samples-per-layer", type=int, default=32)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    registry = json.loads(REGISTRY.read_text())["models"]
    requested = set(args.model or [entry["name"] for entry in registry])
    unknown = requested - {entry["name"] for entry in registry}
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(sorted(unknown))}")
    text = args.text.read_text()
    text = "\n".join([text] * 8)
    args.output.mkdir(parents=True, exist_ok=True)

    metadata_path = args.output / "metadata.json"
    metadata = {
        "text": str(args.text.resolve()),
        "sequence_length": args.sequence_length,
        "samples_per_layer": args.samples_per_layer,
        "seed": args.seed,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "models": [],
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        settings = ("text", "sequence_length", "samples_per_layer", "seed")
        if any(existing.get(key) != metadata[key] for key in settings):
            raise SystemExit(
                f"Existing calibration metadata in {metadata_path} uses different settings"
            )
        metadata = existing
    for model_index, entry in enumerate(registry):
        if entry["name"] not in requested:
            continue
        destination = args.output / f"{entry['name']}.npz"
        if destination.exists():
            print(f"skip {entry['name']}: {destination} already exists")
            continue
        model_dir = ROOT / entry["local_dir"]
        print(f"load {entry['name']} from {model_dir}", flush=True)
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.sequence_length,
            add_special_tokens=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        load_seconds = time.perf_counter() - load_started
        model.eval()
        layers = find_layers(model)
        mlp_inputs = {
            index: Reservoir(args.samples_per_layer, args.seed + model_index * 1000 + index)
            for index in range(len(layers))
        }
        down_inputs = {
            index: Reservoir(args.samples_per_layer, args.seed + 500_000 + model_index * 1000 + index)
            for index in range(len(layers))
        }
        handles = []
        for layer_index, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue

            def capture_mlp(module, inputs, index=layer_index):
                del module
                mlp_inputs[index].add(inputs[0])

            handles.append(mlp.register_forward_pre_hook(capture_mlp))
            experts = getattr(mlp, "experts", None)
            if experts is None:
                continue
            for expert in experts:
                down_proj = getattr(expert, "down_proj", None)
                if down_proj is None:
                    continue

                def capture_down(module, inputs, index=layer_index):
                    del module
                    down_inputs[index].add(inputs[0])

                handles.append(down_proj.register_forward_pre_hook(capture_down))

        print(f"forward {entry['name']} with {encoded['input_ids'].shape[-1]} tokens", flush=True)
        forward_started = time.perf_counter()
        with torch.inference_mode():
            model(**encoded, use_cache=False)
        forward_seconds = time.perf_counter() - forward_started
        for handle in handles:
            handle.remove()

        arrays = {}
        for layer_index in range(len(layers)):
            if mlp_inputs[layer_index].rows:
                arrays[f"mlp_{layer_index}"] = mlp_inputs[layer_index].array()
            if down_inputs[layer_index].rows:
                arrays[f"down_{layer_index}"] = down_inputs[layer_index].array()
        np.savez_compressed(destination, **arrays)
        model_metadata = {
            "name": entry["name"],
            "repo_id": entry["repo_id"],
            "revision": entry["revision"],
            "tokens": int(encoded["input_ids"].shape[-1]),
            "layers": len(layers),
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "arrays": {name: list(value.shape) for name, value in arrays.items()},
        }
        metadata["models"] = [
            item for item in metadata["models"] if item["name"] != entry["name"]
        ]
        metadata["models"].append(model_metadata)
        print(f"saved {destination}", flush=True)
        del model, tokenizer, encoded, arrays
        gc.collect()

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
