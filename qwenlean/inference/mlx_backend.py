"""All Apple/MLX-specific inference stays behind this backend."""

import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import psutil
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

from qwenlean.metrics.repetition import suffix_loop
from qwenlean.utils.io import digest


def make_sampler(sampling):
    temperature = sampling["temperature"]

    def sampler(logprobs):
        if temperature == 0:
            return mx.argmax(logprobs, axis=-1)
        # Explicit ordering: penalties in logits processor, temperature, top-k, top-p, min-p.
        values = logprobs.astype(mx.float32) / temperature
        if sampling.get("top_k", 0):
            values = apply_top_k(values, sampling["top_k"])
        values = values - mx.logsumexp(values, axis=-1, keepdims=True)
        if 0 < sampling.get("top_p", 1) < 1:
            values = apply_top_p(values, sampling["top_p"])
        if sampling.get("min_p", 0):
            values = apply_min_p(values, sampling["min_p"])
        return mx.random.categorical(values)

    return sampler


def make_penalties(prompt_length, sampling):
    def penalties(tokens, logits):
        # Only generated tokens, not the prompt; full history, no implicit 20-token window.
        generated = tokens[prompt_length:]
        if generated.size == 0:
            return logits
        present = mx.zeros((logits.shape[-1],), dtype=mx.float32).at[generated].add(1) > 0
        presence = sampling.get("presence_penalty", 0)
        logits = logits - presence * present
        repetition = sampling.get("repetition_penalty", 1)
        if repetition != 1:
            penalized = mx.where(logits < 0, logits * repetition, logits / repetition)
            logits = mx.where(present, penalized, logits)
        return logits

    return penalties


class MLXBackend:
    def __init__(self, config):
        self.config = config
        path = snapshot_download(
            config["model"],
            revision=config["revision"],
            allow_patterns=["*.json", "*.jinja", "*.safetensors", "tokenizer*", "*.txt"],
        )
        self.model, self.tokenizer = load(path)
        mx.eval(self.model.parameters())
        model_config = json.loads((Path(path) / "config.json").read_text())
        eos = model_config.get("text_config", model_config).get("eos_token_id")
        for token_id in eos if isinstance(eos, list) else [eos]:
            if token_id is not None:
                self.tokenizer.add_eos_token(self.tokenizer.convert_ids_to_tokens(token_id))
        self.eos_ids = set(self.tokenizer.eos_token_ids)
        self.model_info = {
            "model": config["model"],
            "revision": config["revision"],
            "backend": "mlx",
            "eos_token_ids": sorted(self.eos_ids),
            "precision": "upstream_unquantized_bfloat16_text_weights",
            "template_sha256": hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
            "config_hash": digest(json.loads((Path(path) / "config.json").read_text())),
        }

    def generate(self, prompt, seed, config=None):
        config = config or self.config
        sampling = config["sampling"]
        thinking = config.get("enable_thinking", True)
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        prompt_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        mx.random.seed(seed)
        sampler = make_sampler(sampling)
        penalties = make_penalties(len(prompt_ids), sampling)

        mx.reset_peak_memory()
        start = time.perf_counter()
        ids, pieces = [], []
        guard = config.get("loop_guard", {})
        guard_evidence = None
        closing = self.tokenizer.convert_tokens_to_ids("</think>")
        in_reasoning = thinking
        iterator = stream_generate(
            self.model,
            self.tokenizer,
            prompt_ids,
            max_tokens=config["max_output_tokens"],
            sampler=sampler,
            logits_processors=[penalties],
        )
        rss_peak = psutil.Process().memory_info().rss
        first_token_s = None
        for response in iterator:
            if first_token_s is None:
                first_token_s = time.perf_counter() - start
            ids.append(response.token)
            pieces.append(response.text)
            if len(ids) % 64 == 0:
                rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
            if response.token == closing:
                in_reasoning = False
            if (
                guard.get("enabled", False)
                and in_reasoning
                and len(ids) % guard.get("check_every", 16) == 0
            ):
                guard_evidence = suffix_loop(
                    ids,
                    **{
                        k: v
                        for k, v in guard.items()
                        if k in {"min_period", "max_period", "repeats"}
                    },
                )
                if guard_evidence:
                    break
        iterator.close()
        mx.synchronize()
        elapsed = time.perf_counter() - start
        if guard_evidence:
            termination = "runtime_loop_guard"
        elif ids[-1] in self.eos_ids:
            termination = "eos"
        elif response.finish_reason == "length":
            termination = "max_output_tokens"
        else:
            termination = "other"
        # Decode actual tokens: stream detokenizer buffering must not lose text on guard stops.
        raw = self.tokenizer.decode(
            [t for t in ids if t not in self.eos_ids], skip_special_tokens=False
        )
        return {
            "raw_output": raw,
            "token_ids": ids,
            "prompt_tokens": len(prompt_ids),
            "formatted_prompt": formatted,
            "latency_s": elapsed,
            "first_token_latency_s": first_token_s,
            "output_tokens_per_second": len(ids) / elapsed,
            "backend_decode_tokens_per_second": response.generation_tps,
            "termination_reason": termination,
            "stop_token_id": ids[-1] if termination == "eos" else None,
            "loop_guard_evidence": guard_evidence,
            "mlx_peak_bytes": mx.get_peak_memory(),
            "rss_peak_observed_bytes": max(rss_peak, psutil.Process().memory_info().rss),
            "system_available_ram_bytes": psutil.virtual_memory().available,
            "model_identifier": self.model_info,
        }
