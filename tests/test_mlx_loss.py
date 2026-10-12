"""Numerical loss/gradient equivalence and exact causal masking on small CPU models."""

import pytest

mx = pytest.importorskip('mlx.core')
pytestmark = pytest.mark.mlx

import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen3_5 import Model, ModelArgs, TextModelArgs  # noqa: E402

from qwenlean.training.mlx_loss import chunked_head_loss, response_loss  # noqa: E402
from qwenlean.training.mlx_sft import attach_adapters  # noqa: E402
from qwenlean.training.quality import supervision_range  # noqa: E402


@pytest.mark.parametrize('chunk', [1, 3, 32])
@pytest.mark.parametrize('dtype', [mx.float32, mx.bfloat16])
def test_checkpointed_chunks_equal_dense_loss_and_hidden_gradients(chunk, dtype):
    with mx.stream(mx.cpu):
        mx.random.seed(91)
        h = mx.random.normal((1, 7, 8)).astype(dtype)
        w = mx.random.normal((31, 8)).astype(dtype)
        labels = mx.array([[1, 7, 9, 3, 11, 0, 30]])
        def project(x):
            return x @ w.T
        def dense(x):
            return nn.losses.cross_entropy(project(x).astype(mx.float32), labels, reduction='mean')
        def bounded(x):
            return chunked_head_loss(x, labels, project, chunk)
        l0, g0 = mx.value_and_grad(dense)(h)
        l1, g1 = mx.value_and_grad(bounded)(h)
        mx.eval(l0, g0, l1, g1)
        assert mx.allclose(l0, l1, atol=1e-5).item()
        assert mx.allclose(g0, g1, atol=1e-5 if dtype == mx.float32 else 2e-3).item()


@pytest.mark.parametrize('tied', [True, False])
@pytest.mark.parametrize('materialize', [True, False])
def test_actual_qwen_decoder_lora_gradients_mask_prompt_and_include_first_token_eos(tied, materialize):
    with mx.stream(mx.cpu):
        mx.random.seed(43)
        args = TextModelArgs(model_type='qwen3_5_text', hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, vocab_size=37,
            linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=8,
            linear_value_head_dim=8, full_attention_interval=2, tie_word_embeddings=tied)
        model = Model(ModelArgs(model_type='qwen3_5', text_config=vars(args)))
        attach_adapters(model, {'num_layers': 2, 'lora_parameters': {'rank': 2, 'scale': 2, 'dropout': 0,
            'keys': ['self_attn.q_proj', 'self_attn.v_proj', 'linear_attn.in_proj_qkv']}})
        model.train()
        # Synthetic token fixture, never a teacher target. 36 is the fixture EOS.
        prompt, target = [1, 2, 3], [4, 5, 6, 7, 36]
        tokens = mx.array([prompt+target])
        begin, end = supervision_range(prompt, target)
        def dense(m):
            return nn.losses.cross_entropy(m(tokens[:, :-1])[:, begin:end].astype(mx.float32),
                    tokens[:, 1:][:, begin:end], reduction='mean')
        def bounded(m):
            return response_loss(m, tokens, begin, end, 3, materialize)
        if materialize:
            with pytest.raises(ValueError, match='disabled'):
                bounded(model)
            return
        l0, g0 = nn.value_and_grad(model, dense)(model)
        l1, g1 = nn.value_and_grad(model, bounded)(model)
        mx.eval(l0, g0, l1, g1)
        assert mx.allclose(l0, l1, atol=1e-5).item()
        flat0, flat1 = dict(tree_flatten(g0)), dict(tree_flatten(g1))
        assert flat0.keys() == flat1.keys()
        for name in flat0:
            assert mx.allclose(flat0[name], flat1[name], atol=2e-5).item(), name
        losses = nn.losses.cross_entropy(model(tokens[:, :-1]).astype(mx.float32), tokens[:, 1:])
        assert mx.allclose(l1, losses[:, 2:7].mean(), atol=1e-5).item()
        assert not mx.allclose(l1, losses[:, 3:7].mean(), atol=1e-5).item()  # First token is supervised.
        assert not mx.allclose(l1, losses[:, 2:6].mean(), atol=1e-5).item()  # EOS is supervised.
        model.language_model.model.embed_tokens.unfreeze()
        if tied:
            with pytest.raises(ValueError, match='frozen output head'):
                bounded(model)


@pytest.mark.parametrize('chunk', [0, -1, True, 1.5])
def test_invalid_chunk_size_fails_closed(chunk):
    with mx.stream(mx.cpu), pytest.raises(ValueError, match='chunk size'):
        chunked_head_loss(mx.ones((1, 2, 3)), mx.array([[0, 1]]), lambda x: x, chunk)
