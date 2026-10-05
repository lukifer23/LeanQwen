"""Compare a real MLX greedy trace to the saved exact-checkpoint MPS diagnostic."""

import json
from pathlib import Path

import yaml

from qwenlean.inference.mlx_backend import MLXBackend
from qwenlean.utils.io import write_json
from qwenlean.utils.process_lock import model_process_lock


def main():
    reference = json.loads(Path("reports/transformers_probe.json").read_text())
    if not reference["status"].startswith("inference_and_adapter_attachment_ok"):
        raise ValueError("No valid MPS diagnostic")
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    config["name"] = "MLX-MPS-greedy-parity"
    config["max_output_tokens"] = len(reference["token_ids"])
    config["sampling"].update(temperature=0.0, presence_penalty=0.0, repetition_penalty=1.0)
    backend = MLXBackend(config)
    result = backend.generate("Compute 17 * 23.", 42)
    actual = result["token_ids"]
    expected = reference["token_ids"]
    prefix = 0
    for a, b in zip(actual, expected):
        if a != b:
            break
        prefix += 1
    report = {
        "config": config,
        "generation": result,
        "reference": "reports/transformers_probe.json",
        "same_revision": reference["revision"] == config["revision"],
        "matching_prefix_tokens": prefix,
        "reference_tokens": len(expected),
        "identical_token_sequences": actual == expected,
        "note": "One diagnostic greedy generation, not numerical equivalence on all inputs; no model instances overlap",
    }
    write_json("reports/backend_parity.json", report)
    print(
        "Greedy parity:",
        prefix,
        "/",
        len(expected),
        "tokens; identical:",
        actual == expected,
        flush=True,
    )


if __name__ == "__main__":
    with model_process_lock():
        main()
