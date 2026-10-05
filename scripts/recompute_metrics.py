"""Recompute deterministic metrics from preserved emitted IDs; never regenerate outputs."""

import argparse
import json
import shutil
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


def recompute(run):
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
    original = read_jsonl(run / "samples.jsonl")
    backup = run / "samples.original.jsonl"
    if not backup.exists():
        shutil.copyfile(run / "samples.jsonl", backup)
    before_version = original[0].get("scoring_version", "legacy")
    version_backup = run / f"samples.before-{before_version}.jsonl"
    if not version_backup.exists():
        shutil.copyfile(run / "samples.jsonl", version_backup)
    records = []
    for r in original:
        p = parse_tokens(r["token_ids"], t, config.get("enable_thinking", True))
        fields = asdict(p)
        fields.pop("reasoning_ids")
        records.append(
            {
                **r,
                **fields,
                **score(p.final, r["expected"]),
                "repetition": repetition_metrics(
                    p.reasoning, p.reasoning_ids, config.get("metrics", {})
                ),
                "answer_distance": conclusion_distance(
                    p.reasoning, p.reasoning_ids, r["expected"], t
                ),
                "metric_revision": SCORING_VERSION,
            }
        )
    write_jsonl(run / "samples.jsonl", records)
    write_json(run / "summary.json", summarize(records))
    write_json(
        run / "reanalysis.json",
        {
            "original_samples_hash": digest(read_jsonl(backup)),
            "reanalyzed_samples_hash": digest(records),
            "accuracy_changed_samples": sum(
                a["correct"] != b["correct"] for a, b in zip(original, records)
            ),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "reason": "Separate explicit final-conclusion correctness from format compliance; preserve raw generations",
            "generation_unchanged": True,
        },
    )
    print(run, "rescored", len(records), "correct", sum(r["correct"] for r in records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    recompute(parser.parse_args().run)


if __name__ == "__main__":
    main()
