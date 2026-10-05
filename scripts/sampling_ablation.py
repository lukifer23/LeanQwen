"""Staged DEV-only experiment: three single-variable changes, then one full finalist."""

import argparse
from collections import Counter
from pathlib import Path

import yaml

from qwenlean.evaluation.runner import evaluate
from qwenlean.evaluation.summary import paired_comparison, summarize
from qwenlean.inference.mlx_backend import MLXBackend
from qwenlean.utils.io import digest, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    args = parser.parse_args()
    baseline = Path(args.baseline_run)
    if not (baseline / "completion.json").exists():
        raise ValueError("Baseline must be complete")
    import json

    if json.loads((baseline / "completion.json").read_text())["status"] != "complete":
        raise ValueError("Baseline must be complete")
    tasks = read_jsonl("data/splits/dev.jsonl")
    if any(t["split"] != "dev" for t in tasks):
        raise ValueError("Sweep is DEV only")
    selected = []
    counts = Counter()
    for task in tasks:
        if counts[task["family"]] < 2:
            selected.append(task)
            counts[task["family"]] += 1
    baseline_records = read_jsonl(baseline / "samples.jsonl")
    chosen_ids = {t["sample_id"] for t in selected}
    pilot_a = [r for r in baseline_records if r["sample_id"] in chosen_ids]
    if len(pilot_a) != len(selected):
        raise ValueError("Baseline does not contain every pilot sample")
    base = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    backend = MLXBackend(base)
    runs = []
    # Predeclared broad stage. Never scan TEST to choose configurations.
    for name in ["cooler", "presence-zero", "repetition-105"]:
        config = yaml.safe_load(Path(f"configs/sampling/{name}.yaml").read_text())
        run, summary = evaluate(backend, selected, config, label=f"pilot-{name}")
        records = read_jsonl(run / "samples.jsonl")
        runs.append(
            {
                "name": name,
                "path": str(run),
                "config": config,
                "summary": summary,
                "paired_against_A": paired_comparison(pilot_a, records),
            }
        )
    # Correctness first, then clean termination, then median reasoning tokens.
    ranked = sorted(
        runs,
        key=lambda r: (
            -r["summary"]["accuracy"],
            r["summary"]["max_output_rate"],
            r["summary"]["reasoning_tokens"]["p50"],
        ),
    )
    finalist = ranked[0]
    config = finalist["config"]
    config["name"] = f"B-candidate-{finalist['name']}"
    Path("configs/sampling/finalist.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    run, summary = evaluate(backend, tasks, config)
    records = read_jsonl(run / "samples.jsonl")
    original = {
        r["sample_id"]: r["token_ids"] for r in read_jsonl(Path(finalist["path"]) / "samples.jsonl")
    }
    repeated = [r for r in records if r["sample_id"] in original]
    reproducibility = {
        "repeated_pilot_samples": len(repeated),
        "identical_token_sequences": sum(
            r["token_ids"] == original[r["sample_id"]] for r in repeated
        ),
    }
    report = {
        "baseline_run": str(baseline),
        "pilot_A": summarize(pilot_a),
        "pilot_sample_ids": [t["sample_id"] for t in selected],
        "pilot_dataset_hash": digest(selected),
        "pilot_candidates": runs,
        "selection_rule": "accuracy descending, cap rate ascending, median reasoning ascending",
        "finalist_run": str(run),
        "finalist_summary": summary,
        "full_paired_against_A": paired_comparison(baseline_records, records),
        "repeatability": reproducibility,
        "test_evaluated": False,
    }
    write_json("reports/sampling_measurements.json", report)
    print("FINALIST", run, flush=True)


if __name__ == "__main__":
    from qwenlean.utils.process_lock import model_process_lock

    with model_process_lock():
        main()
