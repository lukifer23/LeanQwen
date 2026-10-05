import json
from fractions import Fraction

import pytest

from qwenlean.datasets.procedural import build_pools, generate
from qwenlean.datasets.provenance import check_contamination, validate_provenance
from qwenlean.inference.parsing import parse_tokens
from qwenlean.metrics.repetition import longest_repeated_span, repetition_metrics, suffix_loop
from qwenlean.scoring.exact import score
from qwenlean.utils.io import create_run, digest, read_jsonl, write_jsonl


class Tokenizer:
    eos_token_ids = {99}

    def convert_tokens_to_ids(self, text):
        return {"<think>": 97, "</think>": 98}[text]

    def decode(self, ids):
        return "".join(
            {1: "work ", 2: "FINAL: 5", 97: "<think>", 98: "</think>", 99: ""}[i] for i in ids
        )


def test_parse_prefill_and_accounting():
    p = parse_tokens([1, 1, 98, 2, 99], Tokenizer(), True)
    assert p.reasoning == "work work "
    assert p.final == "FINAL: 5"
    assert p.reasoning_tokens == 2 and p.final_tokens == 1 and p.control_tokens == 2
    assert p.total_output_tokens == p.reasoning_tokens + p.final_tokens + p.control_tokens
    assert p.parse_status == "complete"
    truncated = parse_tokens([1, 1], Tokenizer(), True)
    assert truncated.final == "" and truncated.parse_status == "unclosed_thinking"
    assert parse_tokens([2, 99], Tokenizer(), False).reasoning_tokens == 0
    assert parse_tokens([97, 1, 98, 2, 99], Tokenizer(), False).reasoning_tokens == 1


@pytest.mark.parametrize(
    "text,expected,correct",
    [
        ("FINAL: 1,234", "1234", True),
        ("FINAL: 1/2", "0.5", True),
        ("FINAL: -9", "-9", True),
        ("I think 391 but FINAL: 392", "391", False),
        ("391 is an intermediate result.", "391", False),
        ("", "391", False),
        ("FINAL: 1\nFINAL: 2", "2", False),
        ("\\boxed{42}", "42", True),
        ("FINAL: 42/0", "42", False),
        ("FINAL: 42.5", "42", False),
        ("FINAL: 42.", "42", True),
    ],
)
def test_exact_scoring(text, expected, correct):
    assert score(text, expected)["correct"] is correct


def test_all_generated_ground_truth_independently():
    for split in ["train", "dev", "test"]:
        tasks = generate(split)
        assert tasks == generate(split)
        assert len(tasks) == 60
        for task in tasks:
            f = task["facts"]
            family = task["family"]
            if family == "arithmetic":
                value = f["a"] * f["b"] - f["c"]
            elif family == "word_problem":
                value = Fraction(f["boxes"] * f["items_per_box"] * f["price_cents"]) * Fraction(
                    100 - f["discount_percent"], 100
                )
            elif family == "algebra":
                value = Fraction(f["rhs"] - f["offset"], f["coefficient"])
            elif family == "boolean_logic":
                a, b, c, d = f["values"]
                value = (
                    int(a and not b)
                    + int(b or c)
                    + int((c or d) and not (c and d))
                    + int(not a or not d)
                )
            elif family == "ordering":
                value = f["order"][f["position"] - 1]
            else:
                value = f["initial"]
                for op, operand in f["ops"]:
                    if op == "add":
                        value += operand
                    elif op == "multiply":
                        value *= operand
                    else:
                        value -= operand
                    value %= f["modulus"]
            assert Fraction(task["expected"]) == value


def test_split_and_permission_gates(tmp_path):
    pools = build_pools(tmp_path)
    check_contamination(pools)
    validate_provenance(pools["train"][0], for_training=True)
    with pytest.raises(ValueError, match="TRAIN"):
        validate_provenance(pools["test"][0], for_training=True)
    duplicate = dict(pools["train"][0], split="dev")
    with pytest.raises(ValueError, match="contamination"):
        check_contamination({"train": pools["train"], "dev": [duplicate]})
    bad = {
        **pools["train"][0],
        "provenance": {**pools["train"][0]["provenance"], "response_origin": "closed_model"},
    }
    with pytest.raises(ValueError, match="origin"):
        validate_provenance(bad, for_training=True)
    assert json.loads((tmp_path / "manifest.json").read_text())["counts"]["test"] == 60


def test_repetition_and_loop_normal_language():
    assert longest_repeated_span(list("abcdabcd")) == 4
    assert longest_repeated_span(list("abcdef")) == 0
    # Three substantial cycles trigger; two cycles do not.
    block = list(range(32))
    assert suffix_loop(block * 3)["period_tokens"] == 32
    assert suffix_loop(block * 2) is None
    assert suffix_loop(list(range(300))) is None
    text = (
        "Calculate the product carefully with exact intermediate values before the final answer. "
    )
    result = repetition_metrics(text * 3, block * 3)
    assert result["repetition_density"] > 0.4
    assert result["longest_repeated_span_words"] >= 12
    assert result["loop_evidence"] is not None
    normal = repetition_metrics("The value is 3. The value is 5. The value is 7.", list(range(30)))
    assert normal["loop_evidence"] is None
    assert normal["ngrams"]["8"]["redundant_occurrences"] == 0


def test_run_serialization(tmp_path):
    run = create_run(tmp_path, "test", {"seed": 7})
    assert run != create_run(tmp_path, "test", {"seed": 7})
    data = [{"raw_output": "\n<think>Δ</think>\n", "tokens": [1, 2], "correct": False}]
    write_jsonl(run / "samples.jsonl", data)
    assert read_jsonl(run / "samples.jsonl") == data
    assert json.loads((run / "run.json").read_text())["config_hash"] == digest({"seed": 7})


def test_process_lock_rejects_duplicate_and_recovers_stale_owner(tmp_path):
    from qwenlean.utils.process_lock import model_process_lock

    path = tmp_path / "model.lock"
    with model_process_lock(path):
        with pytest.raises(RuntimeError, match="already holds"):
            with model_process_lock(path):
                pass
    # A released lease permits the next sequential workload.
    with model_process_lock(path):
        pass
    path.write_text(json.dumps({"pid": 99999999, "create_time": 0, "active": True}))
    with model_process_lock(path):
        pass


def test_uncertainty_and_paired_comparison_are_not_false_certainty():
    from qwenlean.evaluation.summary import paired_comparison, wilson_interval

    lo, hi = wilson_interval(0, 60)
    assert lo == 0 and 0.05 < hi < 0.07
    lo, hi = wilson_interval(60, 60)
    assert 0.93 < lo < 0.95 and hi == pytest.approx(1)
    a = [
        {
            "sample_id": "x",
            "prompt": "p",
            "expected": "1",
            "seed": 42,
            "correct": False,
            "reasoning_tokens": 500,
        }
    ]
    b = [{**a[0], "correct": True, "reasoning_tokens": 300}]
    p = paired_comparison(a, b)
    assert p["accuracy_delta"] == 1 and p["gained"] == 1 and p["reasoning_median_delta"] == -200
    with pytest.raises(ValueError, match="same sample"):
        paired_comparison(a, [{**b[0], "sample_id": "other"}])


def test_loop_thresholds_are_not_limited_to_default_window():
    from qwenlean.metrics.repetition import find_loop

    block = list(range(200))
    evidence = find_loop(block * 3, min_period=200, max_period=200, repeats=3)
    assert evidence["period_tokens"] == 200 and evidence["end_token"] == 600


def test_process_lock_recovers_incomplete_dead_owner_metadata(tmp_path):
    from qwenlean.utils.process_lock import model_process_lock

    path = tmp_path / "model.lock"
    path.write_text('{"pid":')
    with model_process_lock(path):
        assert json.loads(path.read_text())["active"] is True
