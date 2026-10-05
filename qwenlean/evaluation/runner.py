"""Durable trajectories with explicit task/policy/replicate contracts and safe resume."""

import json
import os
import subprocess
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

from qwenlean.datasets.prompts import replicate_seed
from qwenlean.datasets.provenance import validate_provenance
from qwenlean.evaluation.summary import summarize
from qwenlean.inference.parsing import parse_tokens
from qwenlean.metrics.answer_distance import conclusion_distance
from qwenlean.metrics.format_meta import format_meta_reasoning
from qwenlean.metrics.repetition import repetition_metrics
from qwenlean.scoring.exact import LEGACY_SCORING_VERSION, SCORING_VERSION, score
from qwenlean.utils.environment import environment_manifest
from qwenlean.utils.io import create_run, digest, read_jsonl, write_json

VALID_TERMINATIONS = {"eos", "max_output_tokens", "runtime_loop_guard"}


def job_plan(tasks, config):
    count = config.get("replicates", 1)
    if type(count) is not int or count <= 0:
        raise ValueError("replicates must be a positive integer")
    if len({t["sample_id"] for t in tasks}) != len(tasks):
        raise ValueError("Dataset contains duplicate rendered prompt IDs")
    groups = OrderedDict()
    for task in tasks:
        groups.setdefault(task.get("task_id", task["sample_id"]), []).append(task)
    return [(t, rep) for group in groups.values() for rep in range(count) for t in group]


def scoring_contract(tasks, config):
    default = (
        SCORING_VERSION if any("prompt_policy" in t for t in tasks) else LEGACY_SCORING_VERSION
    )
    return {
        "version": config.get("scoring_version", default),
        "model_revision": config.get("revision"),
        "prompt_policy_versions": sorted({t.get("prompt_policy_version", "legacy") for t in tasks}),
    }


def validate_resume(run, tasks, config):
    run = Path(run)
    if digest(json.loads((run / "config.json").read_text())) != digest(config):
        raise ValueError("Resume configuration mismatch")
    if json.loads((run / "dataset.json").read_text())["sha256"] != digest(tasks):
        raise ValueError("Resume dataset mismatch")
    plan = job_plan(tasks, config)
    records = read_jsonl(run / "samples.jsonl") if (run / "samples.jsonl").exists() else []
    expected = [(t["sample_id"], rep) for t, rep in plan[: len(records)]]
    actual = [(r["sample_id"], r.get("replicate", 0)) for r in records]
    if len(records) > len(plan) or actual != expected or len(set(actual)) != len(actual):
        raise ValueError("Resume samples must be an exact ordered prefix without duplicates")
    contract_path = run / "scoring_contract.json"
    if not contract_path.exists() or json.loads(contract_path.read_text()) != scoring_contract(
        tasks, config
    ):
        raise ValueError("Resume scoring contract/model revision mismatch or unavailable contract")
    completion = (
        json.loads((run / "completion.json").read_text())
        if (run / "completion.json").exists()
        else {}
    )
    if completion.get("status") == "complete" or len(records) == len(plan):
        raise ValueError("Completed run cannot be resumed")
    if completion.get("status") == "aborted":
        raise ValueError("An explicitly aborted historical run cannot be resumed")
    for r, (task, rep) in zip(records, plan):
        if (
            r.get("config_hash") != digest(config)
            or r.get("prompt") != task["prompt"]
            or r.get("expected") != task["expected"]
            or r.get("seed") != replicate_seed(config["seed"], task, rep)
            or r.get("scoring_version") != scoring_contract(tasks, config)["version"]
            or r.get("model_identifier", {}).get("revision") != config["revision"]
        ):
            raise ValueError("Resume record violates config/model/dataset/seed/scoring contract")
    return records


def evaluate(backend, tasks, config, label=None, root="runs", resume=None):
    if not tasks:
        raise ValueError("No evaluation samples")
    for task in tasks:
        validate_provenance(task)
    plan = job_plan(tasks, config)
    contract = scoring_contract(tasks, config)
    if resume:
        run = Path(resume)
        records = validate_resume(run, tasks, config)
        saved_model = json.loads((run / "model.json").read_text()) if (run / "model.json").exists() else (records[0]["model_identifier"] if records else None)
        if saved_model is not None and digest(saved_model) != digest(backend.model_info):
            raise ValueError("Resume loaded model/adapter/precision/template/context fingerprint changed")
        event = {
            "existing_generations": len(records),
            "remaining_generations": len(plan) - len(records),
            "configuration_unchanged": True,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
        with (run / "resumptions.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
    else:
        run = create_run(root, label or config["name"], config)
        write_json(run / "environment.json", environment_manifest())
        write_json(run / "model.json", backend.model_info)
        write_json(run / "scoring_contract.json", contract)
        write_json(
            run / "dataset.json",
            {
                "sha256": digest(tasks),
                "sample_ids": [t["sample_id"] for t in tasks],
                "split": tasks[0]["split"],
                "count": len(tasks),
                "generations": len(plan),
                "job_plan_hash": digest([(t["sample_id"], rep) for t, rep in plan]),
            },
        )
        records = []
    write_json(
        run / "completion.json",
        {"status": "running", "generations": len(records), "planned": len(plan)},
    )
    try:
        with (run / "samples.jsonl").open("a" if resume else "w") as handle:
            for i, (task, rep) in enumerate(plan[len(records) :], start=len(records)):
                seed = replicate_seed(config["seed"], task, rep)
                generation = backend.generate(task["prompt"], seed, config)
                parsed = parse_tokens(
                    generation["token_ids"], backend.tokenizer, config.get("enable_thinking", True)
                )
                fields = asdict(parsed)
                fields.pop("reasoning_ids")
                gid = f"{run.name}:{task['sample_id']}:replicate-{rep}"
                record = {
                    **task,
                    **generation,
                    **fields,
                    **score(
                        parsed.final,
                        task["expected"],
                        task.get("prompt_policy", "P3"),
                        contract["version"],
                    ),
                    "format_meta": format_meta_reasoning(parsed.reasoning),
                    "generation_id": gid,
                    "task_id": task.get("task_id", task["sample_id"]),
                    "replicate": rep,
                    "seed": seed,
                    "generation_parameters": config,
                    "config_hash": digest(config),
                    "repetition": repetition_metrics(
                        parsed.reasoning, parsed.reasoning_ids, config.get("metrics", {})
                    ),
                    "answer_distance": conclusion_distance(
                        parsed.reasoning, parsed.reasoning_ids, task["expected"], backend.tokenizer
                    ),
                    "provenance": {
                        **task["provenance"],
                        "prompt_source_id": task.get("task_id", task["sample_id"]),
                        "source_id": gid,
                        "response_origin": "qwen_self",
                        "model_revision": config["revision"],
                        "license": "Apache-2.0",
                        "training_permitted": task["split"] == "train",
                    },
                }
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                records.append(record)
                print(
                    f"{run.name} {i + 1}/{len(plan)} {task['family']} {task.get('prompt_policy', 'legacy')} rep={rep} correct={record['correct']} think={parsed.reasoning_tokens} stop={record['termination_reason']} {record['latency_s']:.2f}s",
                    flush=True,
                )
                if record["termination_reason"] not in VALID_TERMINATIONS:
                    raise RuntimeError(
                        f"Generation failed explicitly: {record['termination_reason']}; preserved {gid}"
                    )
    except BaseException as exc:
        write_json(
            run / "completion.json",
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "generations": len(records),
                "planned": len(plan),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    summary = summarize(records)
    write_json(run / "summary.json", summary)
    write_json(
        run / "completion.json",
        {
            "status": "complete",
            "samples": len(records),
            "unique_tasks": len({r["task_id"] for r in records}),
        },
    )
    print(f"RUN: {run}", flush=True)
    return run, summary
