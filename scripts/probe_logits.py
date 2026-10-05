"""Sequential teacher-forced logit comparison at the first greedy divergence."""

import argparse
import json
from pathlib import Path

import numpy as np

from qwenlean.utils.io import write_json
from qwenlean.utils.process_lock import model_process_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["mps", "mlx"])
    args = parser.parse_args()
    meta = json.loads(Path("reports/environment.json").read_text())
    parity = json.loads(Path("reports/backend_parity.json").read_text())
    reference = json.loads(Path("reports/transformers_probe.json").read_text())
    prefix = reference["token_ids"][: parity["matching_prefix_tokens"]]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        meta["model"], revision=meta["revision"], local_files_only=True
    )
    prompt_ids = tokenizer.encode(
        parity["generation"]["formatted_prompt"], add_special_tokens=False
    )
    if args.backend == "mps":
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            meta["model"],
            revision=meta["revision"],
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        ).to("mps")
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([prompt_ids], device="mps"), use_cache=True)
            cache = out.past_key_values
            for token in prefix:
                out = model(
                    input_ids=torch.tensor([[token]], device="mps"),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
        logits = out.logits[0, -1].float().cpu().numpy()
    else:
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        path = snapshot_download(
            meta["model"],
            revision=meta["revision"],
            local_files_only=True,
            allow_patterns=["*.json", "*.jinja", "*.safetensors", "tokenizer*", "*.txt"],
        )
        model, _ = load(path)
        cache = make_prompt_cache(model)
        model(mx.array([prompt_ids[:-1]]), cache=cache)
        out = model(mx.array([[prompt_ids[-1]]]), cache=cache)
        for token in prefix:
            out = model(mx.array([[token]]), cache=cache)
            mx.eval(out)
        logits = np.array(out[0, -1].astype(mx.float32))
    directory = Path("runs/compat")
    directory.mkdir(exist_ok=True)
    np.save(directory / f"{args.backend}_logits.npy", logits)
    top = np.argsort(logits)[-10:][::-1]
    write_json(
        directory / f"{args.backend}_logits.json",
        {
            "backend": args.backend,
            "prefix_tokens": len(prefix),
            "prompt_tokens": len(prompt_ids),
            "top_tokens": [
                {"id": int(t), "text": tokenizer.decode([int(t)]), "logit": float(logits[t])}
                for t in top
            ],
        },
    )
    if all((directory / f"{b}_logits.npy").exists() for b in ["mps", "mlx"]):
        a = np.load(directory / "mps_logits.npy").astype(np.float64)
        b = np.load(directory / "mlx_logits.npy").astype(np.float64)
        report = {
            "prefix_tokens": len(prefix),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "cosine": float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)),
            "mps_top_tokens": json.loads((directory / "mps_logits.json").read_text())["top_tokens"],
            "mlx_top_tokens": json.loads((directory / "mlx_logits.json").read_text())["top_tokens"],
            "top20_overlap": len(set(np.argsort(a)[-20:]) & set(np.argsort(b)[-20:])),
            "note": "Teacher-forced cached forward at one divergence, bf16 original weights. Sequential processes, one model instance at a time.",
        }
        write_json("reports/backend_logit_comparison.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    with model_process_lock():
        main()
