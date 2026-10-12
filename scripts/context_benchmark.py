"""Bounded full-history context probes; exact paired tasks, durable per-job resume."""

import argparse
import json
from pathlib import Path

import psutil
import yaml

from qwenlean.datasets.context_suite import QUESTIONS, VERSION, build_context_task
from qwenlean.evaluation.single_job import evaluate_single
from qwenlean.inference.context import DEFAULT_PROFILE, PROFILES
from qwenlean.utils.io import digest, write_json
from qwenlean.utils.process_lock import model_process_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nonthinking.yaml")
    parser.add_argument("--context-profile", choices=list(PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--lengths", nargs="+", type=int, default=[2048, 8192, 32768])
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--reserve", type=int, default=256)
    parser.add_argument("--adapter-path")
    parser.add_argument("--state", required=True)
    parser.add_argument("--min-available-gib", type=float, default=4)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config.update(name="context-stock" if not args.adapter_path else "context-adapted",
                  context_profile=args.context_profile, prefill_step_size=args.prefill_step_size,
                  enable_thinking=False, max_output_tokens=args.reserve, prompt_policy="P2",
                  min_available_ram_bytes=int(args.min_available_gib * 2**30),
                  sampling={"temperature": 0, "presence_penalty": 0, "repetition_penalty": 1},
                  loop_guard={"enabled": False})
    if args.adapter_path:
        config["adapter_path"] = args.adapter_path
    if (args.lengths != sorted(set(args.lengths)) or
            any(n > PROFILES[args.context_profile] for n in args.lengths)):
        parser.error("Use unique progressive lengths within the selected combined context limit")
    root = Path(args.state)
    root.mkdir(parents=True, exist_ok=True)
    contract = {"config": config, "lengths": args.lengths, "task_version": VERSION,
                "min_available_gib": args.min_available_gib}
    path = root / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Context benchmark resume contract differs")
    write_json(path, contract)
    from qwenlean.inference.mlx_backend import MLXBackend

    results = []
    with model_process_lock(), MLXBackend(config) as backend:
        for length in args.lengths:
            for kind in QUESTIONS:
                if psutil.virtual_memory().available < args.min_available_gib * 2**30:
                    write_json(root / "deferred.json", {"length": length, "task": kind,
                               "reason": "available_memory_below_declared_floor",
                               "available_bytes": psutil.virtual_memory().available})
                    raise RuntimeError("Memory floor reached; context escalation deferred")
                task = build_context_task(backend.tokenizer, length, args.reserve, kind)
                # evaluate() independently renders and tokenizes the same official
                # template; verify construction before allowing model generation.
                row = evaluate_single(backend, task, config, root / "jobs" / task["task_id"])
                if row["prompt_token_ids"] != task["prompt_token_ids"]:
                    raise ValueError("Actual generation prompt differs from constructed stimulus")
                results.append(row)
                write_json(root / "progress.json", {"completed": len(results), "planned": len(args.lengths) * len(QUESTIONS)})
    summary = {"status": "complete", "task_version": VERSION, "contract_hash": digest(contract),
               "config": config, "training_permitted": False, "test_used": False,
               "by_combined_budget": {str(n): {
                   "correct": sum(r["correct"] for r in results if r["sequence_budget"] == n),
                   "count": sum(r["sequence_budget"] == n for r in results),
                   "input_tokens": n - args.reserve,
                   "mlx_peak_bytes": max(r["mlx_peak_bytes"] for r in results if r["sequence_budget"] == n),
                   "rss_peak_observed_bytes": max(r["rss_peak_observed_bytes"] for r in results if r["sequence_budget"] == n),
               } for n in args.lengths}}
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
