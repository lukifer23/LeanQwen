"""Controlled prompt interventions, paired by task and replicate, clustered by task."""

from collections import defaultdict

import numpy as np

from qwenlean.evaluation.summary import bootstrap_mean_ci
from qwenlean.utils.io import digest


def intervention_comparison(a, b):
    def keyed(rows):
        keys = [(r["task_id"], r["replicate"]) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate task/replicate; select a single prompt policy per variant")
        if len({r["prompt_policy"] for r in rows}) != 1:
            raise ValueError("Select exactly one policy per variant")
        return dict(zip(keys, rows))

    left, right = keyed(a), keyed(b)
    if left.keys() != right.keys() or not left:
        raise ValueError("Prompt comparison needs identical task/replicate sets")
    changes = defaultdict(list)
    deltas = {}
    for key, r in left.items():
        other = right[key]

        def invariant(row):
            return {
                k: row[k]
                for k in ["task_id", "task_content_hash", "expected", "seed", "scoring_version"]
            } | {
                "model_identifier": row["model_identifier"],
                "sampler": row["generation_parameters"]["sampling"],
                "cap": row["generation_parameters"]["max_output_tokens"],
                "thinking": row["generation_parameters"].get("enable_thinking", True),
                "guard": row["generation_parameters"].get("loop_guard", {}),
            }

        if digest(invariant(r)) != digest(invariant(other)):
            raise ValueError(
                "Prompt intervention changed model/weights/task/seed/sampler/cap/scoring/guard"
            )
        if (
            r["prompt_policy"] == other["prompt_policy"]
            or r["rendered_prompt_id"] == other["rendered_prompt_id"]
        ):
            raise ValueError("Prompt intervention needs distinct rendered policies")
        fields = {
            "accuracy": lambda x: int(x["correct"]),
            "format_compliance": lambda x: x["format_compliant"],
            "eos_rate": lambda x: int(x["termination_reason"] == "eos"),
            "cap_rate": lambda x: int(x["termination_reason"] == "max_output_tokens"),
            "reasoning_tokens": lambda x: x["reasoning_tokens"],
            "total_tokens": lambda x: x["total_output_tokens"],
            "latency_s": lambda x: x["latency_s"],
            "repeated_content_density": lambda x: x["repetition"]["repetition_density"],
            "lexical_density": lambda x: x["repetition"]["lexical_redundancy"]["density"],
            "semantic_density": lambda x: x.get("semantic", {}).get("density"),
            "format_meta_density": lambda x: x.get("format_meta", {}).get("word_density", 0),
        }
        delta = {
            name: fn(other) - fn(r) if fn(other) is not None and fn(r) is not None else None
            for name, fn in fields.items()
        }
        changes[key[0]].append(delta)
    for name in next(iter(changes.values()))[0]:
        values = [
            np.mean([d[name] for d in rows])
            for rows in changes.values()
            if all(d[name] is not None for d in rows)
        ]
        deltas[name] = (
            {
                "mean_paired_task_delta": float(np.mean(values)),
                "task_bootstrap_ci95": bootstrap_mean_ci(values),
            }
            if values
            else None
        )
    return {
        "unique_tasks": len(changes),
        "paired_trajectories": len(left),
        "left_policy": a[0]["prompt_policy"],
        "right_policy": b[0]["prompt_policy"],
        "deltas": deltas,
        "per_task_deltas": {
            tid: {
                name: float(np.mean([d[name] for d in rows]))
                if all(d[name] is not None for d in rows)
                else None
                for name in rows[0]
            }
            for tid, rows in changes.items()
        },
        "uncertainty_unit": "task; replicate outcomes are averaged within task",
    }
