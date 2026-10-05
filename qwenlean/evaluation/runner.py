"""Backend-independent evaluation and append-only raw sample records."""

import importlib.metadata as md
import json
from dataclasses import asdict

from qwenlean.datasets.provenance import validate_provenance
from qwenlean.evaluation.summary import summarize
from qwenlean.inference.parsing import parse_tokens
from qwenlean.metrics.answer_distance import conclusion_distance
from qwenlean.metrics.repetition import repetition_metrics
from qwenlean.scoring.exact import score
from qwenlean.utils.io import create_run, digest, write_json


def evaluate(backend, tasks, config, label=None, root="runs"):
    if not tasks:
        raise ValueError("No evaluation samples")
    for task in tasks:
        validate_provenance(task)
    run = create_run(root, label or config["name"], config)
    write_json(
        run / "environment.json",
        {
            "packages": {
                p: md.version(p) for p in ["mlx", "mlx-lm", "transformers", "numpy", "qwenlean"]
            }
        },
    )
    write_json(
        run / "dataset.json",
        {
            "sha256": digest(tasks),
            "sample_ids": [r["sample_id"] for r in tasks],
            "split": tasks[0]["split"],
            "count": len(tasks),
        },
    )
    records = []
    with (run / "samples.jsonl").open("w") as handle:
        for i, task in enumerate(tasks):
            seed = (config["seed"] + int(digest(task["sample_id"])[:8], 16)) % (2**32)
            generation = backend.generate(task["prompt"], seed, config)
            parsed = parse_tokens(
                generation["token_ids"], backend.tokenizer, config.get("enable_thinking", True)
            )
            metrics = repetition_metrics(
                parsed.reasoning, parsed.reasoning_ids, config.get("metrics", {})
            )
            distance = conclusion_distance(
                parsed.reasoning, parsed.reasoning_ids, task["expected"], backend.tokenizer
            )
            fields = asdict(parsed)
            fields.pop("reasoning_ids")
            record = {
                **task,
                **generation,
                **fields,
                **score(parsed.final, task["expected"]),
                "seed": seed,
                "generation_parameters": config,
                "config_hash": digest(config),
                "repetition": metrics,
                "answer_distance": distance,
                "provenance": {
                    **task["provenance"],
                    "prompt_source_id": task["sample_id"],
                    "source_id": f"{run.name}:{task['sample_id']}",
                    "response_origin": "qwen_self",
                    "model_revision": config["revision"],
                    "license": "Apache-2.0",
                    "training_permitted": task["split"] == "train",
                },
            }
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            records.append(record)
            print(
                f"{run.name} {i + 1}/{len(tasks)} {task['family']} "
                f"correct={record['correct']} think={parsed.reasoning_tokens} "
                f"stop={record['termination_reason']} {record['latency_s']:.2f}s",
                flush=True,
            )
    summary = summarize(records)
    write_json(run / "summary.json", summary)
    write_json(run / "completion.json", {"status": "complete", "samples": len(records)})
    print(f"RUN: {run}", flush=True)
    return run, summary
