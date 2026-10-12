"""Bitwise, bounded-copy verification of current frozen parameters."""

import hashlib
import json

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten


def tensor_fingerprint(leaves, chunk_elements=262144, on_host_chunk=None):
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError('Positive integer integrity chunk size required')
    leaves = sorted(leaves)
    if not leaves or len({name for name, _ in leaves}) != len(leaves):
        raise ValueError("Integrity verification requires nonempty unique tensor names")
    result = hashlib.sha256()
    elements = 0
    for name, value in leaves:
        if value.size == 0:
            raise ValueError("Empty tensor in integrity verification")
        header = json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":"))
        result.update(header.encode() + b"\0")
        flat = value.reshape(-1)
        # Raw bytes preserve dtype, signed zero and every stored bit. Host copies
        # are bounded to a chunk; no second full base checkpoint is retained.
        for start in range(0, value.size, chunk_elements):
            chunk = flat[start:start + chunk_elements].view(mx.uint8, stream=mx.cpu)
            host = np.asarray(chunk)
            if on_host_chunk is not None:
                on_host_chunk(host.nbytes)
            result.update(host.tobytes())
        elements += value.size
    return {"sha256": result.hexdigest(), "tensor_count": len(leaves), "elements": elements,
            "method": "sorted_name_shape_dtype_raw_bytes_chunked_v1"}


def frozen_base_fingerprint(model):
    trainable = {name for name, _ in tree_flatten(model.trainable_parameters())}
    parameters = tree_flatten(model.parameters())
    names = {name for name, _ in parameters}
    if len(names) != len(parameters) or not trainable.issubset(names):
        raise ValueError('Trainable parameters must belong to the unique full parameter set')
    if not trainable or any(not n.endswith(("lora_a", "lora_b")) for n in trainable):
        raise ValueError("Expected nonempty adapter-only trainable parameters")
    frozen = [(name, value) for name, value in parameters if name not in trainable]
    if any(name.endswith(("lora_a", "lora_b")) for name, _ in frozen):
        raise ValueError("An adapter tensor is unexpectedly frozen")
    # Canonicalize only the extra LoRA wrapper path. This permits an independent
    # comparison with every original loaded parameter before attachment.
    prefixes = {name.rsplit('.', 1)[0] for name in trainable}
    canonical = []
    for name, value in frozen:
        for prefix in prefixes:
            wrapped = prefix + '.linear.'
            if name.startswith(wrapped):
                name = prefix + '.' + name[len(wrapped):]
                break
        canonical.append((name, value))
    return tensor_fingerprint(canonical)
