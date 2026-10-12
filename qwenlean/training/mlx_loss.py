"""Complete response loss without materializing full-sequence vocabulary logits.

The causal decoder still sees every input token. Only the prompt's unsupervised
output projection is skipped; all response tokens, including EOS, keep their loss.
Checkpointed head chunks preserve full-vocabulary cross entropy and hidden-state
gradients while recomputing their large intermediates during backward.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def chunked_head_loss(hidden, labels, project, chunk_size):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Positive integer loss chunk size required')
    if hidden.ndim != 3 or labels.shape != hidden.shape[:2] or labels.size == 0:
        raise ValueError('Nonempty aligned response states and labels required')

    def chunk_sum(states, targets):
        logits = project(states).astype(mx.float32)
        # Checkpoint differentiates its inputs; integer class IDs are constants.
        return nn.losses.cross_entropy(logits, mx.stop_gradient(targets), reduction='sum')

    checkpointed = mx.checkpoint(chunk_sum)
    total = mx.array(0.0, dtype=mx.float32)
    for start in range(0, labels.shape[1], chunk_size):
        stop = min(start+chunk_size, labels.shape[1])
        total = total + checkpointed(hidden[:, start:stop], labels[:, start:stop])
    # Sum tokens, not equally weighted chunk means: the last chunk can be short.
    return total / labels.size


def response_loss(model, tokens, begin, end, chunk_size=32):
    if tokens.ndim != 2 or not 0 <= begin < end <= tokens.shape[1]-1:
        raise ValueError('Invalid causal supervision range')
    text = getattr(model, 'language_model', None)
    if text is None or text.model_type != 'qwen3_5_text':
        raise ValueError('Response-only head path requires the audited Qwen3.5 text model')
    head = text.model.embed_tokens if text.args.tie_word_embeddings else text.lm_head
    if tree_flatten(head.trainable_parameters()):
        raise ValueError('This bounded loss path requires a frozen output head')
    hidden = text.model(tokens[:, :-1])[:, begin:end, :]
    project = head.as_linear if text.args.tie_word_embeddings else head
    return chunked_head_loss(hidden, tokens[:, 1:][:, begin:end], project, chunk_size)
