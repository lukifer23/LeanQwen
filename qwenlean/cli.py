"""Small inspectable CLI: configurations, data and run artifacts are explicit."""

import argparse
import json
from pathlib import Path

import yaml

from qwenlean.datasets.procedural import build_pools
from qwenlean.evaluation.runner import evaluate
from qwenlean.evaluation.summary import paired_comparison, summarize
from qwenlean.utils.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(prog="qwenlean")
    subs = parser.add_subparsers(dest="command", required=True)
    build = subs.add_parser("build-dataset", help="Build deterministic TRAIN/DEV/TEST pools")
    build.add_argument("--output", default="data/splits")
    build.add_argument("--per-family", type=int, default=10)
    ev = subs.add_parser("eval", help="Run real model evaluation")
    ev.add_argument("--config", required=True)
    ev.add_argument("--dataset", default="data/splits/dev.jsonl")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--runs-dir", default="runs")
    gen = subs.add_parser("generate", help="Save real inference for a supplied prompt")
    gen.add_argument("--config", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--output", required=True)
    analyze = subs.add_parser("analyze")
    analyze.add_argument("run")
    compare = subs.add_parser("compare")
    compare.add_argument("runs", nargs="+")
    compare.add_argument("--output", default="reports/comparison.json")
    args = parser.parse_args()
    if args.command == "build-dataset":
        if args.per_family <= 0:
            parser.error("--per-family must be positive")
        pools = build_pools(args.output, args.per_family)
        print({s: len(r) for s, r in pools.items()})
    elif args.command in {"eval", "generate"}:
        config = yaml.safe_load(Path(args.config).read_text())
        if config.get("backend") != "mlx":
            parser.error("This milestone implements only the validated MLX backend")
        from qwenlean.inference.mlx_backend import MLXBackend
        from qwenlean.utils.process_lock import model_process_lock

        with model_process_lock():
            backend = MLXBackend(config)
            if args.command == "eval":
                tasks = read_jsonl(args.dataset)
                if args.limit:
                    tasks = tasks[: args.limit]
                evaluate(backend, tasks, config, root=args.runs_dir)
            else:
                write_json(args.output, backend.generate(args.prompt, config["seed"]))
    elif args.command == "analyze":
        summary = summarize(read_jsonl(Path(args.run) / "samples.jsonl"))
        write_json(Path(args.run) / "summary.json", summary)
        print(json.dumps(summary, indent=2))
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
