"""Adversarial state/serialization fixtures, never benchmark or model evidence."""

from types import SimpleNamespace

import pytest

from qwenlean.datasets.procedural import generate
from qwenlean.datasets.prompts import POLICIES, render_task, replicate_seed
from qwenlean.evaluation.runner import scoring_contract, validate_resume
from qwenlean.inference.streaming import consume_stream, validate_eos_ids
from qwenlean.scoring.exact import SCORING_VERSION, score
from qwenlean.utils.environment import package_versions
from qwenlean.utils.io import digest, write_json, write_jsonl


def stream(tokens, *, fail=False, close_error=False):
    try:
        for token in tokens:
            yield SimpleNamespace(
                token=token, finish_reason="length" if token == tokens[-1] else None
            )
        if fail:
            raise ValueError("explicit test stream failure")
    finally:
        if close_error:
            raise ValueError("explicit test close failure")


def consume(iterator, max_tokens=10):
    return consume_stream(iterator, eos_ids={99}, max_tokens=max_tokens, thinking=True, closing=98)


def test_stream_empty_exception_and_close_are_explicit():
    assert consume(stream([]))["termination_reason"] == "empty_generation"
    r = consume(stream([1, 2], fail=True))
    assert r["token_ids"] == [1, 2] and r["termination_reason"] == "stream_error"
    assert consume(stream([99], close_error=True))["termination_reason"] == "cleanup_error"
    assert consume(stream([99]))["termination_reason"] == "eos"
    assert consume(stream([1, 2]))["termination_reason"] == "unexpected_length_stop"
    assert consume(stream([1, 2]), 2)["termination_reason"] == "max_output_tokens"
    unknown = (SimpleNamespace(token=1, finish_reason="unexpected"),)
    assert consume(iter(unknown))["termination_reason"] == "unknown_finish_reason"
    assert consume(iter([SimpleNamespace(token=1)]))["termination_reason"] == "iterator_exhausted"


@pytest.mark.parametrize("ids", [None, [], [-1], ["99"], [True], [100]])
def test_bad_eos_metadata_fails_closed(ids):
    with pytest.raises(ValueError, match="EOS"):
        validate_eos_ids(ids, 100)


def test_optional_dependency_manifest():
    v = package_versions(["numpy", "qwenlean-nonexistent-optional-package"])
    assert v["numpy"] and v["qwenlean-nonexistent-optional-package"] is None


def test_task_policy_and_seeds_do_not_rewrite_v1():
    source = generate("dev")[0]
    variants = [render_task(source, p) for p in POLICIES]
    assert variants[-1]["prompt"] == source["prompt"]
    assert len({v["task_id"] for v in variants}) == 1
    assert len({v["rendered_prompt_id"] for v in variants}) == 4
    assert len({replicate_seed(7, v, 2) for v in variants}) == 1
    assert len({replicate_seed(7, variants[0], rep) for rep in range(3)}) == 3
    assert "task_id" not in source


@pytest.mark.parametrize(
    "final,expected,correct",
    [
        (r"Answer: \boxed{42}", "42", True),
        (r"Answer: \[\boxed{42}\]", "42", True),
        (r"Example: \boxed{42}", "42", False),
        (r"\boxed{42} but I will reconsider", "42", False),
        ("Maybe the answer is 42.", "42", False),
        ("Work: 42.\nNow solve the remaining steps.", "42", False),
        ("FINAL: 42\nFINAL: 43", "43", False),
        ("Computation\n42", "42", True),
        (r"\boxed{\frac{1}{2}}", "1/2", True),
    ],
)
def test_policy_scoring_adversarial(final, expected, correct):
    assert score(final, expected, "P0")["correct"] is correct


def test_policy_compliance_is_independent():
    assert score(r"\boxed{43}", "42", "P1")["format_compliant"]
    assert not score(r"\boxed{43}", "42", "P1")["correct"]
    assert score("42", "42", "P0")["format_compliant"] is None
    assert score("Steps\n42", "42", "P2")["format_compliant"]
    assert not score("42.", "42", "P2")["format_compliant"]
    assert (
        score("FINAL: 42", "42", version="terminal_numeric_conclusions_v5")["scoring_version"]
        == "terminal_numeric_conclusions_v5"
    )


def test_resume_contract_prefix_completed_and_tampered_record(tmp_path):
    tasks = [render_task(t, "P0") for t in generate("dev")[:2]]
    config = {
        "name": "fixture",
        "model": "fixture",
        "revision": "pinned",
        "seed": 7,
        "replicates": 2,
    }
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "dataset.json", {"sha256": digest(tasks)})
    write_json(tmp_path / "scoring_contract.json", scoring_contract(tasks, config))
    first = {
        "sample_id": tasks[0]["sample_id"],
        "replicate": 0,
        "config_hash": digest(config),
        "prompt": tasks[0]["prompt"],
        "expected": tasks[0]["expected"],
        "seed": replicate_seed(7, tasks[0], 0),
        "scoring_version": SCORING_VERSION,
        "model_identifier": {"revision": "pinned"},
    }
    write_jsonl(tmp_path / "samples.jsonl", [first])
    assert validate_resume(tmp_path, tasks, config) == [first]
    write_json(tmp_path / "completion.json", {"status": "complete"})
    with pytest.raises(ValueError, match="Completed"):
        validate_resume(tmp_path, tasks, config)
    write_json(tmp_path / "completion.json", {"status": "interrupted"})
    write_jsonl(tmp_path / "samples.jsonl", [first, first])
    with pytest.raises(ValueError, match="ordered prefix"):
        validate_resume(tmp_path, tasks, config)
    first["model_identifier"]["revision"] = "other"
    write_jsonl(tmp_path / "samples.jsonl", [first])
    with pytest.raises(ValueError, match="contract"):
        validate_resume(tmp_path, tasks, config)


def test_context_budget_is_a_refusal_not_truncation():
    from qwenlean.inference.streaming import validate_context_budget

    validate_context_budget(20000, 8192, 262144)
    with pytest.raises(ValueError, match="no truncation"):
        validate_context_budget(262144, 1, 262144)


def test_single_job_recovers_fsynced_row_without_completion_publication(tmp_path):
    from qwenlean.evaluation.single_job import evaluate_single

    task = render_task(generate('dev')[0], 'P0')
    config = {'name': 'fixture', 'revision': 'pin', 'seed': 7}
    model = {'revision': 'pin', 'weights_identifier': 'fixture-base'}
    job = tmp_path / 'one-durable-run'
    job.mkdir()
    write_json(job / 'config.json', config)
    write_json(job / 'dataset.json', {'sha256': digest([task])})
    write_json(job / 'scoring_contract.json', scoring_contract([task], config))
    write_json(job / 'model.json', model)
    row = {'sample_id': task['sample_id'], 'prompt': task['prompt'],
           'expected': task['expected'], 'replicate': 0,
           'seed': replicate_seed(7, task, 0), 'config_hash': digest(config),
           'scoring_version': SCORING_VERSION, 'model_identifier': model,
           'termination_reason': 'eos'}
    write_jsonl(job / 'samples.jsonl', [row])
    # This object has no generate method: durable recovery must not generate.
    backend = SimpleNamespace(model_info=model)
    assert evaluate_single(backend, task, config, tmp_path) == row
    row['termination_reason'] = 'stream_error'
    write_jsonl(job / 'samples.jsonl', [row])
    with pytest.raises(ValueError, match='failed'):
        evaluate_single(backend, task, config, tmp_path)


def test_single_job_rejects_multiple_attempts(tmp_path):
    from qwenlean.evaluation.single_job import evaluate_single

    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    with pytest.raises(ValueError, match='Multiple'):
        evaluate_single(SimpleNamespace(), {}, {}, tmp_path)


def test_prompt_comparison_rejects_extra_interventions_and_clusters_replicates():
    from qwenlean.evaluation.prompt_comparison import intervention_comparison

    base = {
        "task_id": "x",
        "task_content_hash": "body",
        "expected": "42",
        "model_identifier": {"revision": "pin", "weights_identifier": "base"},
        "generation_parameters": {"sampling": {"temperature": 1}, "max_output_tokens": 2048},
        "scoring_version": SCORING_VERSION,
        "correct": False,
        "format_compliant": None,
        "termination_reason": "max_output_tokens",
        "reasoning_tokens": 2048,
        "total_output_tokens": 2048,
        "latency_s": 30,
        "repetition": {"repetition_density": 0.1, "lexical_redundancy": {"density": 0.2}},
    }
    a = [
        {
            **base,
            "replicate": i,
            "seed": 100 + i,
            "prompt_policy": "P0",
            "rendered_prompt_id": "x-P0",
        }
        for i in range(3)
    ]
    b = [
        {
            **r,
            "prompt_policy": "P2",
            "rendered_prompt_id": "x-P2",
            "correct": True,
            "format_compliant": True,
        }
        for r in a
    ]
    result = intervention_comparison(a, b)
    assert result["unique_tasks"] == 1 and result["paired_trajectories"] == 3
    assert result["deltas"]["accuracy"]["mean_paired_task_delta"] == 1
    assert result["deltas"]["format_compliance"] is None
    changed = [{**r, "seed": r["seed"] + 1} for r in b]
    with pytest.raises(ValueError, match="seed"):
        intervention_comparison(a, changed)
