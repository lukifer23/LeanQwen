"""Offline pinned-tokenizer context construction; never run the model."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from qwenlean.datasets.context_suite import QUESTIONS, build_context_task

pytestmark = pytest.mark.local_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    cfg = json.loads(Path("reports/environment.json").read_text())
    try:
        return AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"], local_files_only=True)
    except OSError:
        pytest.skip("Pinned tokenizer is not cached")


@pytest.mark.parametrize("length", [2048, 8192, 32768])
def test_actual_token_lengths_ground_truth_positions_and_evaluation_only(tokenizer, length):
    for kind, (_, expected) in QUESTIONS.items():
        task = build_context_task(tokenizer, length, 256, kind)
        assert len(task["prompt_token_ids"]) == length - 256
        assert task["expected"] == str(expected)
        assert not task["provenance"]["training_permitted"]
        text = task["prompt"]
        assert text.index("Aster-17 is 37") < len(text) * .1
        assert .3 < text.index("Birch-29 is 61") / len(text) < .7
        assert text.index("Cedar-43 is 19") > len(text) * .8
        assert task == build_context_task(tokenizer, length, 256, kind)
