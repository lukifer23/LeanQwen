"""Recompute deterministic metrics from preserved emitted IDs; never regenerate outputs."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

from qwenlean.evaluation.summary import summarize
from qwenlean.inference.parsing import parse_tokens
from qwenlean.metrics.answer_distance import conclusion_distance
from qwenlean.metrics.repetition import repetition_metrics
from qwenlean.scoring.exact import SCORING_VERSION, score
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl


def recompute(run, output, version=SCORING_VERSION):
    run = Path(run)
    if json.loads((run / "completion.json").read_text())["status"] != "complete":
        raise ValueError("Only complete runs can be reanalyzed")
    config = json.loads((run / "config.json").read_text())
    t = TokenizerWrapper(
        AutoTokenizer.from_pretrained(
            config["model"], revision=config["revision"], local_files_only=True
        )
    )
    t.add_eos_token("<|endoftext|>")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = read_jsonl(run / "samples.jsonl")
    records = []
    for r in original:
        p = parse_tokens(r["token_ids"], t, config.get("enable_thinking", True))
        fields = asdict(p)
        fields.pop("reasoning_ids")
        records.append(
            {
                **r,
                **fields,
                **score(p.final, r["expected"], r.get("prompt_policy", "P3"), version),
                "repetition": repetition_metrics(
                    p.reasoning, p.reasoning_ids, config.get("metrics", {})
                ),
                "answer_distance": conclusion_distance(
                    p.reasoning, p.reasoning_ids, r["expected"], t
                ),
                "metric_revision": version,
            }
        )
    write_jsonl(output / "samples.jsonl", records)
    write_json(output / "summary.json", summarize(records))
    write_json(
        output / "reanalysis.json",
        {
            "original_samples_hash": digest(original),
            "reanalyzed_samples_hash": digest(records),
            "accuracy_changed_samples": sum(
                a["correct"] != b["correct"] for a, b in zip(original, records)
            ),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "reason": "Separate explicit final-conclusion correctness from format compliance; preserve raw generations",
            "generation_unchanged": True,
            "source_run": str(run),
            "source_scoring_versions": sorted({r.get("scoring_version", "legacy") for r in original}),
            "derived_scoring_version": version,
        },
    )
    print(run, "rescored", len(records), "correct", sum(r["correct"] for r in records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--output", required=True, help="New derived directory; originals never modified")
    parser.add_argument("--scoring-version", default=SCORING_VERSION)
    args = parser.parse_args()
    recompute(args.run, args.output, args.scoring_version)


if __name__ == "__main__":
    main()
