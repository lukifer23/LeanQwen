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
    parser.add_argument(
        "--pilot-runs", nargs=3, help="Reuse complete pilots or resume ordered partial pilots"
    )
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
    for index, name in enumerate(["cooler", "presence-zero", "repetition-105"]):
        config = yaml.safe_load(Path(f"configs/sampling/{name}.yaml").read_text())
        reuse = Path(args.pilot_runs[index]) if args.pilot_runs else None
        if (
            reuse
            and (reuse / "completion.json").exists()
            and json.loads((reuse / "completion.json").read_text())["status"] == "complete"
        ):
            run = reuse
            if json.loads((run / "config.json").read_text()) != config:
                raise ValueError("Reused pilot config mismatch")
            if json.loads((run / "dataset.json").read_text())["sha256"] != digest(selected):
                raise ValueError("Reused pilot dataset mismatch")
        else:
            run, _ = evaluate(backend, selected, config, label=f"pilot-{name}", resume=reuse)

        # Baseline generation already records its explicit scoring contract.
        # Historical runs must never be rewritten by a new scorer.
        summary = json.loads((run / "summary.json").read_text())
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
    report = {
        "baseline_run": str(baseline),
        "pilot_A": summarize(pilot_a),
        "pilot_sample_ids": [t["sample_id"] for t in selected],
        "pilot_dataset_hash": digest(selected),
        "pilot_candidates": runs,
        "selection_rule": "Require nonzero pilot accuracy at least as high as A; then accuracy descending, cap rate ascending, median reasoning ascending",
        "finalist_run": None,
        "test_evaluated": False,
    }
    eligible = [
        r
        for r in ranked
        if r["summary"]["accuracy"] > 0
        and r["summary"]["accuracy"] >= report["pilot_A"]["accuracy"]
    ]
    if not eligible:
        report["decision"] = (
            "No eligible finalist: do not promote shortening when every candidate lacks correct final answers"
        )
        write_json("reports/sampling_measurements.json", report)
        print(report["decision"], flush=True)
        return
    finalist = eligible[0]
    config = {**finalist["config"], "name": f"B-candidate-{finalist['name']}"}
    Path("configs/sampling/finalist.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    run, summary = evaluate(backend, tasks, config)
    records = read_jsonl(run / "samples.jsonl")
    original = {
        r["sample_id"]: r["token_ids"] for r in read_jsonl(Path(finalist["path"]) / "samples.jsonl")
    }
    repeated = [r for r in records if r["sample_id"] in original]
    report.update(
        {
            "decision": "Promote best eligible pilot to full DEV evaluation",
            "finalist_run": str(run),
            "finalist_summary": summary,
            "full_paired_against_A": paired_comparison(baseline_records, records),
            "repeatability": {
                "repeated_pilot_samples": len(repeated),
                "identical_token_sequences": sum(
                    r["token_ids"] == original[r["sample_id"]] for r in repeated
                ),
            },
        }
    )
    write_json("reports/sampling_measurements.json", report)
    print("FINALIST", run, flush=True)


if __name__ == "__main__":
    from qwenlean.utils.process_lock import model_process_lock

    with model_process_lock():
        main()
