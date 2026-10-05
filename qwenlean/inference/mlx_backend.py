"""All Apple/MLX-specific inference stays behind this backend."""

import gc
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import psutil
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

from qwenlean.inference.streaming import consume_stream, validate_context_budget, validate_eos_ids
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
        model_eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
        if model_eos:
            validate_eos_ids(model_eos, len(self.tokenizer))
        for token_id in model_eos:
            if token_id is not None:
                self.tokenizer.add_eos_token(self.tokenizer.convert_ids_to_tokens(token_id))
        text_config = model_config.get("text_config", model_config)
        self.context_window = text_config.get("max_position_embeddings")
        self.eos_ids = validate_eos_ids(self.tokenizer.eos_token_ids, len(self.tokenizer))
        self.model_info = {
            "model": config["model"],
            "revision": config["revision"],
            "backend": "mlx",
            "eos_token_ids": sorted(self.eos_ids),
            "precision": "upstream_unquantized_bfloat16_text_weights",
            "template_sha256": hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
            "config_hash": digest(json.loads((Path(path) / "config.json").read_text())),
            "context_window_tokens": self.context_window,
            "context_config_hash": digest(
                {
                    "max_position_embeddings": self.context_window,
                    "rope_parameters": text_config.get("rope_parameters"),
                }
            ),
            "weights_identifier": f"{config['model']}@{config['revision']}:original-bf16",
            "eos_metadata_warnings": ["model_config_eos_missing"] if eos is None else [],
        }

        if config.get("adapter_path"):
            from mlx_lm.tuner.utils import load_adapters

            adapter = Path(config["adapter_path"])
            meta = json.loads((adapter / "adapter_config.json").read_text())
            if meta.get("revision") != config["revision"] or meta.get("base_model") != config["model"]:
                raise ValueError("Adapter base-model pin mismatch")
            load_adapters(self.model, adapter)
            self.model.eval()
            mx.eval(self.model.parameters())
            adapter_hash = hashlib.sha256((adapter / "adapters.safetensors").read_bytes()).hexdigest()
            self.model_info.update(
                adapter_sha256=adapter_hash,
                weights_identifier=self.model_info["weights_identifier"] + ":adapter:" + adapter_hash,
            )

    def close(self):
        """Release the model instance before the repository lease is released."""
        mx.synchronize()
        self.model = None
        self.tokenizer = None
        gc.collect()
        mx.clear_cache()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def generate(self, prompt, seed, config=None):
        config = config or self.config
        thinking = config.get("enable_thinking", True)
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        prompt_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        return self.generate_tokens(prompt_ids, seed, config, formatted_prompt=formatted)

    def generate_tokens(
        self, prompt_ids, seed, config=None, *, formatted_prompt=None, thinking=None
    ):
        """Explicit token-prefix path used only by labeled counterfactual probes."""
        config = config or self.config
        if config["max_output_tokens"] <= 0:
            raise ValueError("max_output_tokens must be positive")
        thinking = config.get("enable_thinking", True) if thinking is None else thinking
        validate_context_budget(len(prompt_ids), config["max_output_tokens"], self.context_window)
        mx.random.seed(seed)
        sampler = make_sampler(config["sampling"])
        penalties = make_penalties(len(prompt_ids), config["sampling"])
        mx.reset_peak_memory()
        start = time.perf_counter()
        rss_peak = psutil.Process().memory_info().rss
        first_token_s = None

        def observe(count):
            nonlocal rss_peak, first_token_s
            if first_token_s is None:
                first_token_s = time.perf_counter() - start
            if count % 64 == 0:
                rss_peak = max(rss_peak, psutil.Process().memory_info().rss)

        def responses():
            yield from stream_generate(
                self.model,
                self.tokenizer,
                prompt_ids,
                max_tokens=config["max_output_tokens"],
                sampler=sampler,
                logits_processors=[penalties],
            )

        iterator = responses()
        result = consume_stream(
            iterator,
            eos_ids=self.eos_ids,
            max_tokens=config["max_output_tokens"],
            thinking=thinking,
            closing=self.tokenizer.convert_tokens_to_ids("</think>"),
            opening=self.tokenizer.convert_tokens_to_ids("<think>"),
            guard=config.get("loop_guard", {}),
            on_token=observe,
        )
        response = result.pop("last_response")
        ids = result["token_ids"]
        try:
            mx.synchronize()
            raw = self.tokenizer.decode(
                [t for t in ids if t not in self.eos_ids], skip_special_tokens=False
            )
        except Exception as exc:
            raw = ""
            result["generation_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": "synchronize_decode",
            }
            result["termination_reason"] = "synchronize_decode_error"
        elapsed = time.perf_counter() - start
        return {
            **result,
            "raw_output": raw,
            "prompt_tokens": len(prompt_ids),
            "prompt_token_ids": list(prompt_ids),
            "formatted_prompt": formatted_prompt,
            "latency_s": elapsed,
            "first_token_latency_s": first_token_s,
            "output_tokens_per_second": len(ids) / elapsed if elapsed > 0 else None,
            "backend_decode_tokens_per_second": getattr(response, "generation_tps", None),
            "stop_token_id": ids[-1] if ids and result["termination_reason"] == "eos" else None,
            "mlx_peak_bytes": mx.get_peak_memory(),
            "rss_peak_observed_bytes": max(rss_peak, psutil.Process().memory_info().rss),
            "system_available_ram_bytes": psutil.virtual_memory().available,
            "model_identifier": self.model_info,
        }
