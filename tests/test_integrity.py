"""Real small CPU tensors verify integrity failure modes without model weights."""

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.mlx

from qwenlean.training.integrity import tensor_fingerprint  # noqa: E402


def test_nonempty_fingerprint_checks_bits_shape_dtype_and_all_tensors():
    with mx.stream(mx.cpu):
        value = mx.array([1, 2, 3, 4], dtype=mx.bfloat16)
        baseline = tensor_fingerprint([("embedding", value), ("norm", mx.array([1.]))], 2)
        assert baseline["tensor_count"] == 2 and baseline["elements"] == 5
        assert baseline == tensor_fingerprint([("norm", mx.array([1.])), ("embedding", value)], 1)
        assert baseline != tensor_fingerprint([("embedding", value), ("norm", mx.array([2.]))])
        assert tensor_fingerprint([("x", value)]) != tensor_fingerprint([("x", value.reshape(2, 2))])
        assert tensor_fingerprint([("x", value)]) != tensor_fingerprint([("x", value.astype(mx.float32))])
        assert tensor_fingerprint([("x", mx.array([0.]))]) != tensor_fingerprint([("x", mx.array([-0.]))])


def test_empty_or_duplicate_integrity_sets_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        tensor_fingerprint([])
    with mx.stream(mx.cpu):
        a = mx.array([1])
        with pytest.raises(ValueError):
            tensor_fingerprint([("x", a), ("x", a)])


def test_frozen_base_includes_unadapted_layers_and_reads_replaced_arrays():
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    from qwenlean.training.integrity import frozen_base_fingerprint

    with mx.stream(mx.cpu):
        model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        model.freeze()
        model.layers[1] = LoRALinear.from_base(model.layers[1], r=2, scale=2, dropout=0)
        before = frozen_base_fingerprint(model)
        assert before['tensor_count'] == 4 and before['elements'] == 26
        model.layers[1].lora_b = model.layers[1].lora_b + 1
        assert frozen_base_fingerprint(model) == before
        model.layers[0].weight = model.layers[0].weight + 1
        assert frozen_base_fingerprint(model) != before
