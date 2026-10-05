"""Adapter backward compatibility probe on fixed token IDs; no optimizer or training data."""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

meta = json.loads(Path("reports/environment.json").read_text())
report = {
    "model": meta["model"],
    "revision": meta["revision"],
    "optimizer_steps": 0,
    "purpose": "Check differentiability through exact architecture, not model training",
}
try:
    path = snapshot_download(
        meta["model"],
        revision=meta["revision"],
        local_files_only=True,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "tokenizer*", "*.txt"],
    )
    model, _ = load(path)
    model.freeze()
    model.train()  # Select differentiable gated-delta path; eval kernels have no VJP.
    linear_to_lora_layers(
        model,
        2,
        {
            "rank": 4,
            "scale": 2.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv"],
        },
    )
    params = tree_flatten(model.trainable_parameters())
    report["trainable_parameters"] = sum(p.size for _, p in params)
    report["trainable_parameter_names"] = [n for n, _ in params]
    if not params:
        raise RuntimeError("No adapters attached")
    ids = mx.array([[16, 17, 18, 19, 20, 21, 22, 23] * 4])

    def probe_loss(model):
        # Scalar numerical diagnostic; no target responses or dataset selection.
        return model(ids).astype(mx.float32).square().mean()

    start = time.perf_counter()
    loss, grads = nn.value_and_grad(model, probe_loss)(model)
    mx.eval(loss, grads)
    leaves = tree_flatten(grads)
    report.update(
        loss=float(loss.item()),
        backward_s=time.perf_counter() - start,
        finite_gradients=all(bool(mx.all(mx.isfinite(g)).item()) for _, g in leaves),
        nonzero_gradient_tensors=sum(bool(mx.any(g != 0).item()) for _, g in leaves),
        mlx_peak_bytes=mx.get_peak_memory(),
        status="adapter_backward_ok",
    )
except Exception as exc:
    import traceback

    report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
Path("reports/mlx_lora_probe.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
