"""Small inspectable CLI: configurations, data and run artifacts are explicit."""

import argparse
import json
from pathlib import Path

import yaml

from qwenlean.datasets.procedural import build_pools
from qwenlean.datasets.prompts import POLICIES, render_task
from qwenlean.evaluation.prompt_comparison import intervention_comparison
from qwenlean.evaluation.runner import evaluate, validate_resume
from qwenlean.evaluation.summary import paired_comparison, summarize
from qwenlean.utils.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(prog="qwenlean")
    subs = parser.add_subparsers(dest="command", required=True)
    build = subs.add_parser("build-dataset", help="Build deterministic TRAIN/DEV/TEST pools")
    build.add_argument("--output", default="data/splits")
    build.add_argument("--per-family", type=int, default=10)
    build.add_argument("--version", choices=["v1", "v2"], default="v1")
    build.add_argument("--per-bin", type=int, default=2)
    ev = subs.add_parser("eval", help="Run real model evaluation")
    ev.add_argument("--config", required=True)
    ev.add_argument("--dataset", default="data/splits/dev.jsonl")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--runs-dir", default="runs")
    ev.add_argument(
        "--resume", help="Resume a matching incomplete run; never regenerate its prefix"
    )
    ev.add_argument("--replicates", type=int)
    ev.add_argument("--prompt-policy", choices=list(POLICIES))
    gen = subs.add_parser("generate", help="Save real inference for a supplied prompt")
    gen.add_argument("--config", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--output", required=True)
    analyze = subs.add_parser("analyze")
    analyze.add_argument("run")
    analyze.add_argument("--output", help="Save a new summary explicitly; default only prints")
    compare = subs.add_parser("compare")
    compare.add_argument("runs", nargs="+")
    compare.add_argument("--output", default="reports/comparison.json")
    prompt_compare = subs.add_parser(
        "compare-prompts", help="Task/replicate paired prompt intervention"
    )
    prompt_compare.add_argument("left")
    prompt_compare.add_argument("right")
    prompt_compare.add_argument("--left-policy", required=True, choices=list(POLICIES))
    prompt_compare.add_argument("--right-policy", required=True, choices=list(POLICIES))
    prompt_compare.add_argument("--output", required=True)
    train = subs.add_parser("train-sft", help="Real masked LoRA training from an approved TRAIN pool")
    train.add_argument("--config", required=True)
    train.add_argument("--dataset", required=True)
    train.add_argument("--quality-manifest", required=True)
    train.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build-dataset":
        if args.per_family <= 0:
            parser.error("--per-family must be positive")
        if args.version == "v2":
            from qwenlean.datasets.procedural_v2 import build_pools as build_v2

            pools = build_v2(args.output, args.per_bin)
        else:
            pools = build_pools(args.output, args.per_family)
        print({s: len(r) for s, r in pools.items()})
    elif args.command == "train-sft":
        from qwenlean.training.mlx_sft import train

        config = yaml.safe_load(Path(args.config).read_text())
        result = train(config, args.dataset, args.quality_manifest, args.output)
        print(json.dumps(result, indent=2))
    elif args.command in {"eval", "generate"}:
        config = yaml.safe_load(Path(args.config).read_text())
        if config.get("backend") != "mlx":
            parser.error("This milestone implements only the validated MLX backend")
        if args.command == "eval":
            if args.replicates is not None:
                config["replicates"] = args.replicates
            policy = args.prompt_policy or config.get("prompt_policy")
            tasks = read_jsonl(args.dataset)
            if args.limit:
                tasks = tasks[: args.limit]
            if policy:
                config["prompt_policy"] = policy
                tasks = [render_task(t, policy) for t in tasks]
            if args.resume:
                validate_resume(args.resume, tasks, config)
        from qwenlean.inference.mlx_backend import MLXBackend
        from qwenlean.utils.process_lock import model_process_lock

        with model_process_lock():
            backend = MLXBackend(config)
            if args.command == "eval":
                evaluate(backend, tasks, config, root=args.runs_dir, resume=args.resume)
            else:
                write_json(args.output, backend.generate(args.prompt, config["seed"]))
    elif args.command == "analyze":
        summary = summarize(read_jsonl(Path(args.run) / "samples.jsonl"))
        if args.output:
            write_json(args.output, summary)
        print(json.dumps(summary, indent=2))
    elif args.command == "compare-prompts":

        def records(path, policy):
            p = Path(path)
            rows = read_jsonl(p / "samples.jsonl" if p.is_dir() else p)
            return [r for r in rows if r.get("prompt_policy") == policy]

        result = intervention_comparison(
            records(args.left, args.left_policy), records(args.right, args.right_policy)
        )
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        records = [read_jsonl(Path(r) / "samples.jsonl") for r in args.runs]
        output = {
            "runs": {r: summarize(data) for r, data in zip(args.runs, records)},
            "paired_against_first": {
                r: paired_comparison(records[0], data)
                for r, data in zip(args.runs[1:], records[1:])
            },
        }
        write_json(args.output, output)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
