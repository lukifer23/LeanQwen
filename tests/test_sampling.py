"""Check the real MLX logit processor without loading model weights."""

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.mlx

from qwenlean.inference.mlx_backend import make_penalties, make_sampler  # noqa: E402


def test_presence_excludes_prompt_and_penalizes_full_history():
    processor = make_penalties(2, {"presence_penalty": 1.5, "repetition_penalty": 1.0})
    logits = mx.array([[3.0, 4.0, 5.0, 6.0]])
    assert processor(mx.array([0, 1]), logits).tolist() == [[3.0, 4.0, 5.0, 6.0]]
    assert processor(mx.array([0, 1, 2, 2, 3]), logits).tolist() == [[3.0, 4.0, 3.5, 4.5]]


def test_repetition_sign_and_no_prompt_penalty():
    processor = make_penalties(1, {"presence_penalty": 0, "repetition_penalty": 2})
    output = processor(mx.array([0, 1, 2]), mx.array([[4.0, -3.0, 6.0]]))
    assert output.tolist() == [[4.0, -6.0, 3.0]]


def test_sampler_support_and_seed():
    sampler = make_sampler({"temperature": 0.6, "top_k": 1, "top_p": 0.95})
    mx.random.seed(42)
    assert sampler(mx.array([[-2.0, 0.0, -1.0]])).item() == 1
    sampler = make_sampler({"temperature": 0, "top_k": 0, "top_p": 1})
    assert sampler(mx.array([[4.0, 5.0, 3.0]])).item() == 1
