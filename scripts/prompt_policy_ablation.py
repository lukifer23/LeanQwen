"""One serialized, resumable 12-task × 3-seed × 4-policy DEV intervention."""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from qwenlean.datasets.prompts import render_task
from qwenlean.evaluation.prompt_comparison import intervention_comparison
from qwenlean.evaluation.runner import evaluate, validate_resume
from qwenlean.evaluation.summary import summarize
from qwenlean.utils.io import read_jsonl, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock


def pilot_tasks():
    selected, counts = [], Counter()
    for t in read_jsonl("data/splits/dev.jsonl"):
        if counts[t["family"]] < 2:
            selected.append(t)
            counts[t["family"]] += 1
    if len(selected) != 12 or any(t["split"] != "dev" for t in selected):
        raise ValueError("Expected the historical balanced 12-task DEV subset")
    return selected


def publish(run):
    if json.loads((run / "completion.json").read_text())["status"] != "complete":
        raise ValueError("Cannot publish incomplete prompt study")
    rows = read_jsonl(run / "samples.jsonl")
    config = json.loads((run / "config.json").read_text())
    policies = config["prompt_policies"]
    subsets = {p: [r for r in rows if r["prompt_policy"] == p] for p in policies}
    summary = {p: summarize(data) for p, data in subsets.items()}
    pairs = {p: intervention_comparison(subsets["P3"], subsets[p]) for p in policies if p != "P3"}
    # Correctness first, then observed termination, then total compute. Provisional DEV choice.
    ranked = sorted(
        policies,
        key=lambda p: (
            -summary[p]["task_mean_accuracy"],
            summary[p]["max_output_rate"],
            summary[p]["format_meta_density"]["mean"],
            summary[p]["total_output_tokens"]["mean"],
            p,
        ),
    )
    selected = ranked[0]
    result = {
        "run": str(run),
        "config": config,
        "dataset_hash": json.loads((run / "dataset.json").read_text())["sha256"],
        "by_policy": summary,
        "paired_against_P3": pairs,
        "selected_policy": selected,
        "selection_rule": "task accuracy descending, cap rate ascending, format-meta density ascending, mean total output ascending; DEV only",
        "scoring_version": rows[0]["scoring_version"],
        "unique_tasks": 12,
        "trajectories": len(rows),
        "test_evaluated": False,
        "manual_calibration_status": "pending; heuristic statistics require review",
    }
    write_json("reports/prompt_policy_measurements.json", result)
    write_jsonl("reports/prompt_policy_records.jsonl", rows)
    lines = [
        "# Prompt-policy ablation",
        "",
        f"Completed {len(rows)} trajectories on twelve DEV tasks, three matched seeds, four policies. Original weights, official thinking sampler, guard off, 2048 total-output cap. No TEST selection.",
        "",
        "| Policy | Correct / 36 | EOS | Cap | Median / P95 reasoning | Mean total tokens | Format compliance | Format-meta word density |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p, s in summary.items():
        fmt = "n/a" if s["format_compliance_rate"] is None else f"{s['format_compliance_rate']:.1%}"
        lines.append(
            f"| {p} | {sum(r['correct'] for r in subsets[p])} / {len(subsets[p])} | {s['termination_counts'].get('eos', 0)} | {s['termination_counts'].get('max_output_tokens', 0)} | {s['reasoning_tokens']['p50']:.0f} / {s['reasoning_tokens']['p95']:.0f} | {s['total_output_tokens']['mean']:.1f} | {fmt} | {s['format_meta_density']['mean']:.1%} |"
        )
    lines += [
        "",
        f"Provisional DEV policy choice: {selected}. Selection rule: "
        + result["selection_rule"]
        + ".",
        "",
        "P0 is bare task; P1 requests step-by-step and boxed answer; P2 requests a final integer-only line; P3 reproduces the historical FINAL wording.",
        "",
        "## Seed variation and uncertainty",
        "",
        "Replicates are averaged within each task; task bootstrap intervals use twelve clusters, not 36 independent problems. Same task/replicate receives the same seed across policies. Numerical outcome and format adherence are separate. The format-meta metric is a transparent line-cue heuristic, not a semantic oracle; manual validation is recorded separately. Latency is from sequential interleaved policy runs during development, with uncontrolled thermals/background activity.",
        "",
    ]
    for p, s in summary.items():
        lines.append(
            f"{p}: {s['within_task_variation']['tasks_with_mixed_correctness']}/12 tasks have both correct and incorrect replicates; mean within-task total-token SD {s['within_task_variation']['mean_total_token_std']:.1f}. Task-bootstrap accuracy interval: {s['task_accuracy_bootstrap_ci95']}."
        )
    lines += [
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv sync --frozen",
        "uv run --frozen python scripts/prompt_policy_ablation.py",
        "# Interrupted run: reuse exactly its saved config and dataset.",
        "uv run --frozen python scripts/prompt_policy_ablation.py --resume runs/<run>",
        "```",
        "",
        f"Full local source: `{run}`. Inspectable raw text, tokens, policies, seeds and metrics are committed in prompt_policy_records.jsonl. Historical v1 stimuli and v5 results are unchanged.",
        "",
    ]
    Path("reports/prompt_policy_ablation.md").write_text("\n".join(lines))
    print("Prompt study published; provisional policy", selected, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/diagnostics/prompt-policies.yaml")
    p.add_argument("--resume")
    p.add_argument("--publish-run")
    args = p.parse_args()
    if args.publish_run:
        publish(Path(args.publish_run))
        return
    cfg = yaml.safe_load(Path(args.config).read_text())
    tasks = [render_task(t, policy) for t in pilot_tasks() for policy in cfg["prompt_policies"]]
    if args.resume:
        validate_resume(args.resume, tasks, cfg)
    from qwenlean.inference.mlx_backend import MLXBackend

    with model_process_lock():
        backend = MLXBackend(cfg)
        run, _ = evaluate(backend, tasks, cfg, resume=args.resume)
    publish(run)


if __name__ == "__main__":
    main()
