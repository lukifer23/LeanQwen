"""Reversible request ceilings; never modify checkpoint positional metadata."""

from qwenlean.inference.streaming import validate_context_budget
from qwenlean.utils.io import digest

PROFILES = {"mac-128k": 131072, "mac-100k": 102400, "native-262k": 262144}
DEFAULT_PROFILE = "mac-128k"


def context_contract(config, text_config):
    profile = config.get("context_profile", DEFAULT_PROFILE)
    if profile not in PROFILES:
        raise ValueError(f"Unknown context profile: {profile}")
    native = text_config.get("max_position_embeddings")
    if type(native) is not int or native <= 0:
        raise ValueError("Missing/invalid original model context metadata")
    limit = PROFILES[profile]
    if limit > native:
        raise ValueError("Operating context exceeds original model context")
    # Include every original setting, including architecture and all RoPE variants.
    # Hashing immutable JSON avoids making a modified model config for each profile.
    contract = {
        "original_model_context_limit": native,
        "context_profile": profile,
        "selected_operational_context_limit": limit,
        "original_text_config_hash": digest(text_config),
        "cache_policy": "full_history_unquantized_no_eviction",
        "prefill_step_size": config.get("prefill_step_size", 2048),
    }
    if type(contract["prefill_step_size"]) is not int or contract["prefill_step_size"] <= 0:
        raise ValueError("prefill_step_size must be a positive integer")
    for key in ("max_kv_size", "kv_bits", "prompt_cache", "prompt_cache_file"):
        if config.get(key) is not None:
            raise ValueError(f"{key} is unsupported in the full-history experiment")
    return {**contract, "context_configuration_fingerprint": digest(contract)}


def request_context(contract, input_tokens, reserved_output_tokens, emitted_tokens=0):
    if type(input_tokens) is not int or input_tokens < 1:
        raise ValueError("Actual input tokens must be a positive integer")
    if type(reserved_output_tokens) is not int or reserved_output_tokens < 1:
        raise ValueError("Reserved output tokens must be a positive integer")
    if type(emitted_tokens) is not int or not 0 <= emitted_tokens <= reserved_output_tokens:
        raise ValueError("Emitted tokens exceed the reserved generation allowance")
    validate_context_budget(
        input_tokens, reserved_output_tokens, contract["selected_operational_context_limit"]
    )
    return {
        **contract,
        "actual_input_tokens": input_tokens,
        "reserved_generation_allowance": reserved_output_tokens,
        "actual_emitted_tokens": emitted_tokens,
        "effective_total_sequence_length": input_tokens + emitted_tokens,
    }
