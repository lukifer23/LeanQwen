"""Offline official tokenizer checks; no model instance or network request."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from qwenlean.inference.parsing import parse_tokens
from qwenlean.metrics.answer_distance import conclusion_distance


@pytest.fixture(scope="module")
def tokenizer():
    meta = json.loads(Path("reports/environment.json").read_text())
    try:
        hf = AutoTokenizer.from_pretrained(
            meta["model"], revision=meta["revision"], local_files_only=True
        )
    except OSError:
        pytest.skip("Pinned tokenizer not cached; run the environment validation first")
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    t = TokenizerWrapper(hf)
    t.add_eos_token("<|endoftext|>")
    return t


def test_real_template_and_generated_token_partition(tokenizer):
    messages = [{"role": "user", "content": "Compute 2 + 3."}]
    thinking = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    nonthinking = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert thinking.endswith("<think>\n")
    assert nonthinking.endswith("<think>\n\n</think>\n\n")
    assert tokenizer.eos_token_ids == {248044, 248046}
    raw = "2 + 3 = 5.\n</think>\nFINAL: 5<|im_end|>"
    ids = tokenizer.encode(raw, add_special_tokens=False)
    parsed = parse_tokens(ids, tokenizer, True)
    assert parsed.reasoning == "2 + 3 = 5.\n"
    assert parsed.final == "\nFINAL: 5"
    assert parsed.total_output_tokens == len(ids)
    assert len(ids) == parsed.reasoning_tokens + parsed.final_tokens + parsed.control_tokens
    metric = conclusion_distance(parsed.reasoning, parsed.reasoning_ids, "5", tokenizer)
    assert metric["evidence"] == "= 5"
    assert 0 < metric["candidate_correct_conclusion_token"] <= len(parsed.reasoning_ids)
    assert metric["tokens_after_candidate"] < len(parsed.reasoning_ids)
    assert (
        conclusion_distance(parsed.reasoning, parsed.reasoning_ids, "500", tokenizer)["evidence"]
        is None
    )


def test_eos_validation_includes_added_special_tokens(tokenizer):
    from qwenlean.inference.streaming import validate_eos_ids

    assert tokenizer.vocab_size <= min(tokenizer.eos_token_ids)
    assert validate_eos_ids(tokenizer.eos_token_ids, len(tokenizer)) == {248044, 248046}
