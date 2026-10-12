"""Offline checkpoint/profile restoration audit. Does not load a model or use Metal."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from qwenlean.inference.context import PROFILES, context_contract
from qwenlean.utils.io import digest, write_json


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-weight-files", action="store_true")
    parser.add_argument("--adapter-path")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("Refuse overwriting an existing audit")
    config = yaml.safe_load(Path(args.config).read_text())
    path = Path(snapshot_download(
        config["model"], revision=config["revision"], local_files_only=True,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "tokenizer*", "*.txt"],
    ))
    original = json.loads((path / "config.json").read_text())
    text = original.get("text_config", original)
    files = [path / "config.json", path / "tokenizer_config.json", path / "tokenizer.json"]
    files += list(path.glob("*.jinja"))
    if args.verify_weight_files:
        weights = sorted(path.glob("*.safetensors"))
        if not weights:
            raise ValueError("No cached original weights to verify")
        files += weights
    if args.adapter_path:
        adapter = Path(args.adapter_path)
        meta = json.loads((adapter / "adapter_config.json").read_text())
        if meta.get("revision") != config["revision"] or meta.get("base_model") != config["model"]:
            raise ValueError("Adapter pin mismatch")
        files += [adapter / "adapter_config.json", adapter / "adapters.safetensors"]
    before = {str(p): file_hash(p) for p in files}
    profiles = [context_contract({**config, "context_profile": p}, text)
                for p in ("mac-128k", "mac-100k", "native-262k")]
    after = {str(p): file_hash(p) for p in files}
    if before != after or digest(original) != digest(json.loads((path / "config.json").read_text())):
        raise RuntimeError("Checkpoint/adapter/tokenizer/positional configuration changed")
    package = importlib.metadata.distribution("mlx-lm")
    sources = ["mlx_lm/generate.py", "mlx_lm/models/qwen3_5.py", "mlx_lm/models/cache.py"]
    report = {
        "model": config["model"], "revision": config["revision"], "profiles": profiles,
        "original_text_config": text, "model_config_hash": digest(original),
        "file_hashes_before": before, "file_hashes_after": after,
        "all_files_unchanged": before == after, "weights_verified": args.verify_weight_files,
        "adapter_verified": bool(args.adapter_path),
        "mlx_lm_version": package.version,
        "installed_source_hashes": {s: file_hash(package.locate_file(s)) for s in sources},
        "configured_lengths": PROFILES, "model_loaded": False,
        "behavioral_context_lengths_verified": [],
        "interpretation": "CPU file/profile integrity audit only; no inference or memory savings demonstrated",
    }
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("all_files_unchanged", "weights_verified", "adapter_verified", "model_loaded")}, indent=2))


if __name__ == "__main__":
    main()
