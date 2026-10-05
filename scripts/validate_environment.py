"""Validate the pinned environment without overwriting historical probe artifacts."""

import argparse
import json
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from qwenlean.utils.environment import environment_manifest
from qwenlean.utils.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default="reports/environment-current.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    path = hf_hub_download(
        config["model"], "config.json", revision=config["revision"], local_files_only=args.offline
    )
    report = {
        **environment_manifest(check_mps=True),
        "model": config["model"],
        "revision": config["revision"],
        "model_config": json.loads(Path(path).read_text()),
    }
    if Path(args.output).exists():
        raise FileExistsError("Refusing to overwrite an existing environment artifact")
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "model_config"}, indent=2))


if __name__ == "__main__":
    main()
