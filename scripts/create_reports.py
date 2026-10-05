"""Generate summaries and plots solely from completed, measured run artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qwenlean.evaluation.summary import paired_comparison, summarize
from qwenlean.utils.io import read_jsonl, write_json


def table_row(name, s):
    t = s["reasoning_tokens"]
    return (
        f"| {name} | {s['accuracy']:.1%} | {t['p50']:.0f} | {t['p95']:.0f} | "
        f"{s['loop_rate']:.1%} | {s['max_output_rate']:.1%} | {s['latency_s']['mean']:.2f} s |"
    )


def table(entries):
    return "\n".join(
        [
            "| Variant | Accuracy | Median Think Tokens | P95 Think Tokens | Exact Loop Rate | Max-Out Rate | Mean Latency |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[table_row(n, s) for n, s in entries],
        ]
    )


def verify_complete(path):
    completion = json.loads((path / "completion.json").read_text())
    if completion["status"] != "complete":
        raise ValueError(f"Cannot report incomplete run: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--sampling", default="reports/sampling_measurements.json")
    args = parser.parse_args()
    baseline = Path(args.baseline_run)
    verify_complete(baseline)
    a = read_jsonl(baseline / "samples.jsonl")
    sa = summarize(a)
    source = {
        str(baseline): {
            "sample_count": len(a),
            "config": json.loads((baseline / "config.json").read_text()),
        }
    }
    write_json("reports/baseline_summary.json", sa)
    lines = [
        "# Baseline A — original Qwen weights, official thinking sampler",
        "",
        "Measured on 60 DEV problems, ten each across six deterministic families. TEST was not evaluated.",
        "Pinned original bf16 text weights, MLX on M3 Pro. Total-output cap: 2048 tokens. No runtime guard.",
        "",
        table([("A: official thinking", sa)]),
        "",
        f"Accuracy 95% sample-bootstrap interval: {sa['accuracy_bootstrap_ci95']}.",
        f"Reasoning mean/p75/p90/p99: {sa['reasoning_tokens']['mean']:.1f} / {sa['reasoning_tokens']['p75']:.1f} / {sa['reasoning_tokens']['p90']:.1f} / {sa['reasoning_tokens']['p99']:.1f}.",
        f"Aggregate output throughput (including prefill): {sa['aggregate_output_tokens_per_second']:.2f} tokens/s.",
        f"MLX allocator peak: {sa['mlx_peak_bytes'] / 1e9:.3f} GB. Sampled process RSS peak: {sa['rss_peak_observed_bytes'] / 1e9:.3f} GB.",
        "Memory measures overlap and are not total-machine usage.",
        f"Mean repeated-content density: {sa['repetition_density']['mean']:.2%}. Longest repeated span: {max(r['repetition']['longest_repeated_span_words'] for r in a)} words.",
        f"Candidate correct-conclusion found in {sa['candidate_conclusion_found']}/{len(a)} traces; this is a heuristic, not proof of solution.",
        "",
        "## By task family",
        "",
        "| Family | Correct / N | Accuracy | Median / P95 Reasoning |",
        "|---|---:|---:|---:|",
    ]
    for family, fs in sa["by_family"].items():
        lines.append(
            f"| {family} | {sum(r['correct'] for r in a if r['family'] == family)} / {fs['n']} | {fs['accuracy']:.1%} | {fs['reasoning_tokens']['p50']:.0f} / {fs['reasoning_tokens']['p95']:.0f} |"
        )
    lines += [
        "",
        "## Inspection candidates",
        "",
        "The following are actual trace excerpts. Excerpts can suggest failure modes; attribution requires manual inspection. Full outputs and token IDs remain in the local run directory.",
        "",
    ]
    selected = {}
    correct = [r for r in a if r["correct"]]
    if correct:
        selected["short_success"] = min(correct, key=lambda r: r["reasoning_tokens"])
        selected["long_success"] = max(correct, key=lambda r: r["reasoning_tokens"])
    capped = [r for r in a if r["termination_reason"] == "max_output_tokens"]
    if capped:
        selected["capped"] = max(capped, key=lambda r: r["repetition"]["repetition_density"])
    selected["largest_literal_repeat"] = max(
        a, key=lambda r: r["repetition"]["longest_repeated_span_words"]
    )
    candidates = [
        r
        for r in a
        if not r["correct"] and r["answer_distance"]["tokens_after_candidate"] is not None
    ]
    if candidates:
        selected["candidate_conclusion_then_failure"] = max(
            candidates, key=lambda r: r["answer_distance"]["tokens_after_candidate"]
        )
    for label, r in selected.items():
        lines += [
            f"### {label}: `{r['sample_id']}`",
            "",
            r["prompt"],
            "",
            f"Ground truth: {r['expected']}; final score: {r['correct']}; reasoning tokens: {r['reasoning_tokens']}; termination: {r['termination_reason']}.",
            "",
            "Reasoning beginning:",
            "",
            "```text",
            r["reasoning"][:700],
            "```",
            "",
            "Reasoning ending:",
            "",
            "```text",
            r["reasoning"][-700:],
            "```",
            "",
            "Final answer:",
            "",
            "```text",
            r["final"],
            "```",
            "",
        ]
    write_json("reports/inspection_records.json", selected)
    lines += [
        "## Limits",
        "",
        "This is a small procedural DEV suite, not a general reasoning benchmark. The cap censors long traces. Exact cycle detection misses paraphrased/reconsideration loops. Lexical cosine is only a semantic redundancy proxy. One generation seed per problem does not measure sampler seed variance. The FINAL instruction may itself influence format deliberation.",
        "",
        "## Reproduction",
        "",
        f"Raw run: `{baseline}`.",
        "```bash",
        "uv sync --frozen --extra compat",
        "uv run --frozen --extra compat qwenlean eval --config configs/baseline.yaml",
        "```",
        "",
    ]
    Path("reports/baseline.md").write_text("\n".join(lines))
    entries = [("A", a, sa)]
    if Path(args.sampling).exists():
        sweep = json.loads(Path(args.sampling).read_text())
        bpath = Path(sweep["finalist_run"])
        verify_complete(bpath)
        b = read_jsonl(bpath / "samples.jsonl")
        sb = summarize(b)
        pair = paired_comparison(a, b)
        entries.append(("B candidate", b, sb))
        source[str(bpath)] = {
            "sample_count": len(b),
            "config": json.loads((bpath / "config.json").read_text()),
        }
        report = [
            "# Initial sampling ablation",
            "",
            "Predeclared staged DEV-only design. Broad stage: 12 balanced problems (two per family) selected before inspecting results. Baseline pilot measurements reuse the corresponding original A samples. Change one parameter at a time; everything else stays fixed.",
            "",
            table(
                [
                    ("A pilot", sweep["pilot_A"]),
                    *[(r["name"], r["summary"]) for r in sweep["pilot_candidates"]],
                ]
            ),
            "",
            f"Selection rule: {sweep['selection_rule']}. Selected {sweep['finalist_run']}.",
            "The selected candidate was then evaluated on all 60 DEV problems. This is sampler selection evidence, not an untouched TEST result.",
            "",
            table([("A official", sa), ("B candidate", sb)]),
            "",
            f"Paired accuracy change: {pair['accuracy_delta']:+.1%}; 95% paired bootstrap interval {pair['paired_accuracy_delta_ci95']}. Gained {pair['gained']}, lost {pair['lost']}.",
            f"Token-identical pilot repeats: {sweep['repeatability']['identical_token_sequences']}/{sweep['repeatability']['repeated_pilot_samples']}.",
            "",
            "## Interpretation",
            "",
            "Inspect correctness, cap rate, task-family regressions and actual traces together. Shortening a capped wrong trace does not preserve capability. No model weights were changed. Broad pilots are too small for stable claims; full DEV results still require an independent confirmation set and multiple generation seeds.",
            "",
            "## Reproduction",
            "",
            "```bash",
            f"uv run --frozen --extra compat python scripts/sampling_ablation.py --baseline-run {baseline}",
            f"uv run --frozen --extra compat python scripts/create_reports.py --baseline-run {baseline}",
            "```",
            "",
        ]
        Path("reports/sampling_ablation.md").write_text("\n".join(report))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, rows, s in entries:
        x = np.sort([r["reasoning_tokens"] for r in rows])
        axes[0].step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=label)
        ci = s["accuracy_bootstrap_ci95"]
        axes[1].errorbar(
            s["reasoning_tokens"]["mean"],
            s["accuracy"],
            yerr=[[s["accuracy"] - ci[0]], [ci[1] - s["accuracy"]]],
            fmt="o",
            label=label,
        )
    axes[0].set(
        xlabel="Reasoning tokens (2048 total-output cap)", ylabel="Empirical cumulative probability"
    )
    axes[1].set(xlabel="Mean reasoning tokens", ylabel="Accuracy", ylim=(0, 1.05))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig("reports/reasoning_accuracy.png", dpi=180)
    plt.close(fig)
    # Compact per-sample evidence persists in Git; bulky full run outputs stay local.
    evidence = {
        label: [
            {
                k: r[k]
                for k in [
                    "sample_id",
                    "family",
                    "correct",
                    "reasoning_tokens",
                    "final_tokens",
                    "total_output_tokens",
                    "latency_s",
                    "termination_reason",
                    "seed",
                ]
            }
            for r in rows
        ]
        for label, rows, _ in entries
    }
    write_json("reports/per_sample_measurements.json", evidence)
    write_json("reports/run_sources.json", source)


if __name__ == "__main__":
    main()
