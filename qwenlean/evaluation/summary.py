"""Measured aggregates, bootstrap uncertainty, and paired sampler comparisons."""

import math
from collections import Counter, defaultdict

import numpy as np

from qwenlean.metrics.repetition import find_loop


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        **{f"p{p}": float(np.percentile(values, p)) for p in [50, 75, 90, 95, 99]},
    }


def bootstrap_mean_ci(values, seed=90210, n=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return list(map(float, np.percentile(means, [2.5, 97.5])))


def wilson_interval(correct, n):
    z = 1.959963984540054
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(records):
    if not records:
        raise ValueError("Cannot summarize an empty run")
    n = len(records)
    correct = [r["correct"] for r in records]
    reasoning = [r["reasoning_tokens"] for r in records]
    groups = defaultdict(list)
    for r in records:
        groups[r.get("task_id", r.get("sample_id", "unit-fixture"))].append(r)
    task_rates = [sum(r["correct"] for r in rows) / len(rows) for rows in groups.values()]
    applicable_format = [
        r["format_compliant"] for r in records if r.get("format_compliant") is not None
    ]
    termination = Counter(r["termination_reason"] for r in records)
    output_loop_evidence = [
        find_loop(r["token_ids"], **r["generation_parameters"].get("metrics", {}).get("loop", {}))
        for r in records
    ]
    family = {}
    for name in sorted({r["family"] for r in records}):
        subset = [r for r in records if r["family"] == name]
        family[name] = {
            "n": len(subset),
            "total_generations": len(subset),
            "unique_tasks": len({r.get("task_id", r.get("sample_id")) for r in subset}),
            "accuracy": sum(r["correct"] for r in subset) / len(subset),
            "reasoning_tokens": distribution([r["reasoning_tokens"] for r in subset]),
        }
    return {
        "n": n,
        "total_generations": n,
        "unique_tasks": len(groups),
        "accuracy_unit": "task_mean"
        if any(len(g) > 1 for g in groups.values())
        else "one_trajectory_per_task",
        "task_mean_accuracy": float(np.mean(task_rates)),
        "task_accuracy_bootstrap_ci95": bootstrap_mean_ci(task_rates),
        "within_task_variation": {
            "tasks_with_mixed_correctness": sum(0 < p < 1 for p in task_rates),
            "fraction_tasks_with_mixed_correctness": sum(0 < p < 1 for p in task_rates)
            / len(groups),
            "mean_total_token_std": float(
                np.mean(
                    [np.std([r["total_output_tokens"] for r in rows]) for rows in groups.values()]
                )
            ),
            "per_task_accuracy": {
                tid: sum(r["correct"] for r in rows) / len(rows) for tid, rows in groups.items()
            },
        },
        "accuracy": sum(correct) / n,
        "strict_final_accuracy": sum(r.get("strict_final_correct", r["correct"]) for r in records)
        / n,
        "format_compliance_rate": sum(applicable_format) / len(applicable_format)
        if applicable_format
        else None,
        "format_applicable_generations": len(applicable_format),
        "scoring_versions": sorted({r.get("scoring_version", "legacy") for r in records}),
        "accuracy_bootstrap_ci95": bootstrap_mean_ci(task_rates),
        "accuracy_wilson_ci95": wilson_interval(sum(correct), n) if len(groups) == n else None,
        "reasoning_tokens": distribution(reasoning),
        "final_tokens": distribution([r["final_tokens"] for r in records]),
        "total_output_tokens": distribution([r["total_output_tokens"] for r in records]),
        "excessive_reasoning_rate": {
            str(t): sum(x > t for x in reasoning) / n for t in [512, 1024, 2048, 4096]
        },
        "output_loop_rate": sum(e is not None for e in output_loop_evidence) / n,
        "loop_rate": sum(r["repetition"]["loop_evidence"] is not None for r in records) / n,
        "repetition_density": distribution(
            [r["repetition"]["repetition_density"] for r in records]
        ),
        "longest_repeated_span_words": distribution(
            [r["repetition"]["longest_repeated_span_words"] for r in records]
        ),
        "lexical_redundancy_density": distribution(
            [r["repetition"]["lexical_redundancy"]["density"] for r in records]
        ),
        "termination_counts": dict(termination),
        "max_output_rate": termination.get("max_output_tokens", 0) / n,
        "generation_error_count": sum(
            r["termination_reason"] not in {"eos", "max_output_tokens", "runtime_loop_guard"}
            for r in records
        ),
        "format_meta_density": distribution(
            [r.get("format_meta", {}).get("word_density", 0) for r in records]
        ),
        "unclosed_thinking_rate": sum(r["parse_status"] == "unclosed_thinking" for r in records)
        / n,
        "latency_s": distribution([r["latency_s"] for r in records]),
        "aggregate_output_tokens_per_second": sum(r["total_output_tokens"] for r in records)
        / sum(r["latency_s"] for r in records),
        "mlx_peak_bytes": max(r["mlx_peak_bytes"] for r in records),
        "rss_peak_observed_bytes": max(r["rss_peak_observed_bytes"] for r in records),
        "candidate_conclusion_found": sum(
            r["answer_distance"]["tokens_after_candidate"] is not None for r in records
        ),
        "candidate_correct_then_wrong_final": sum(
            r["answer_distance"]["tokens_after_candidate"] is not None
            and not r["correct"]
            and r["termination_reason"] == "eos"
            for r in records
        ),
        "accuracy_over_median_reasoning_tokens": sum(correct) / n / max(1, np.median(reasoning)),
        "by_family": family,
    }


def exact_mcnemar_p(gained, lost):
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def paired_comparison(a, b):
    if {r.get("scoring_version", "legacy") for r in a} != {
        r.get("scoring_version", "legacy") for r in b
    }:
        raise ValueError(
            "Paired comparison requires the same scoring version; reanalyze raw outputs first"
        )

    def key(r):
        return r["sample_id"], r.get("replicate", 0)

    if len({key(r) for r in a}) != len(a) or len({key(r) for r in b}) != len(b):
        raise ValueError("Duplicate sample/replicate IDs in paired comparison")
    if {key(r) for r in a} != {key(r) for r in b}:
        raise ValueError("Paired comparison requires exactly the same sample IDs and replicates")
    lookup = {key(r): r for r in b}
    if any(
        r["prompt"] != lookup[key(r)]["prompt"]
        or r["expected"] != lookup[key(r)]["expected"]
        or r["seed"] != lookup[key(r)]["seed"]
        for r in a
    ):
        raise ValueError("Paired prompts, ground truth and seeds must match")
    differences = [int(lookup[key(r)]["correct"]) - int(r["correct"]) for r in a]
    grouped = defaultdict(list)
    for r, d in zip(a, differences):
        grouped[r.get("task_id", r["sample_id"])].append(d)
    task_differences = [np.mean(v) for v in grouped.values()]
    return {
        "accuracy_delta": float(np.mean(task_differences)),
        "paired_accuracy_delta_ci95": bootstrap_mean_ci(task_differences),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(
            sum(d == 1 for d in differences), sum(d == -1 for d in differences)
        )
        if len(grouped) == len(a)
        else None,
        "gained": sum(d == 1 for d in differences),
        "lost": sum(d == -1 for d in differences),
        "reasoning_median_delta": float(
            np.median([r["reasoning_tokens"] for r in b])
            - np.median([r["reasoning_tokens"] for r in a])
        ),
        "unique_tasks": len(grouped),
        "paired_trajectories": len(a),
        "uncertainty_unit": "task",
    }
