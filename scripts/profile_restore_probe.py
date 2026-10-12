"""One leased live instance: profile switching must preserve all loaded tensors/config."""

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from qwenlean.utils.io import digest, write_json
from qwenlean.utils.process_lock import model_process_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--adapter-path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("Refuse overwriting restoration evidence")
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg.update(max_output_tokens=64, context_profile="mac-128k")
    if args.adapter_path:
        cfg["adapter_path"] = args.adapter_path
    from mlx.utils import tree_flatten

    from qwenlean.inference.mlx_backend import MLXBackend
    from qwenlean.training.integrity import tensor_fingerprint

    with model_process_lock(), MLXBackend(cfg) as backend:
        before = tensor_fingerprint(tree_flatten(backend.model.parameters()))
        args_before = digest(asdict(backend.model.args))
        records = [backend.generate("Compute 17 * 23.", cfg["seed"],
                   {**cfg, "context_profile": profile}) for profile in ("mac-128k", "native-262k")]
        after = tensor_fingerprint(tree_flatten(backend.model.parameters()))
        args_after = digest(asdict(backend.model.args))
        result = {"model_identifier": backend.model_info, "weights_before": before,
                  "weights_after": after, "all_loaded_tensors_unchanged": before == after,
                  "live_model_args_hash_before": args_before, "live_model_args_hash_after": args_after,
                  "live_positional_and_architecture_args_unchanged": args_before == args_after,
                  "short_emitted_ids_identical": records[0]["token_ids"] == records[1]["token_ids"],
                  "records": records, "adapter_loaded": bool(args.adapter_path),
                  "full_262k_inference_verified": False,
                  "interpretation": "Live short-request restoration/integrity check; does not validate 262K-length inference"}
    write_json(args.output, result)
    if not all(result[k] for k in ("all_loaded_tensors_unchanged", "live_positional_and_architecture_args_unchanged", "short_emitted_ids_identical")):
        raise RuntimeError("Profile restoration/inference identity failed; evidence retained")
    if any(r['termination_reason'] not in {'eos', 'max_output_tokens'} for r in records):
        raise RuntimeError("Restoration generation failed; evidence retained")
    print("Live profile restoration preserved all loaded tensors, configuration and short output IDs.")


if __name__ == "__main__":
    main()
