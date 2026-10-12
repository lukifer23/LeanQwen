"""Weight-free request budget and reversible profile contracts."""

import copy

import pytest

from qwenlean.inference.context import PROFILES, context_contract, request_context
from qwenlean.utils.io import digest


def test_profiles_restore_without_mutating_original_configuration():
    original = {"max_position_embeddings": 262144, "rope_parameters": {"rope_theta": 10000000},
                "layer_types": ["linear_attention", "full_attention"]}
    snapshot = copy.deepcopy(original)
    small = context_contract({}, original)
    restored = context_contract({"context_profile": "native-262k"}, original)
    assert small["selected_operational_context_limit"] == 131072
    assert restored["selected_operational_context_limit"] == 262144
    assert small["original_text_config_hash"] == restored["original_text_config_hash"]
    assert small["context_configuration_fingerprint"] != restored["context_configuration_fingerprint"]
    assert original == snapshot and digest(original) == digest(snapshot)


@pytest.mark.parametrize("profile,limit", PROFILES.items())
def test_combined_budget_boundary_and_generation_accounting(profile, limit):
    contract = context_contract({"context_profile": profile}, {"max_position_embeddings": 262144})
    record = request_context(contract, limit - 128, 128, 7)
    assert record["effective_total_sequence_length"] == limit - 121
    assert record["actual_emitted_tokens"] == 7
    with pytest.raises(ValueError, match="no truncation"):
        request_context(contract, limit - 127, 128)


@pytest.mark.parametrize("config", [{"context_profile": "unknown"}, {"prefill_step_size": 0},
                                   {"prefill_step_size": True}, {"max_kv_size": 1024},
                                   {"kv_bits": 4}, {"prompt_cache_file": "cache"}])
def test_invalid_or_lossy_configuration_rejected(config):
    with pytest.raises(ValueError):
        context_contract(config, {"max_position_embeddings": 262144})


def test_missing_native_metadata_and_oversized_emission_fail_closed():
    with pytest.raises(ValueError):
        context_contract({}, {})
    with pytest.raises(ValueError):
        context_contract({}, {"max_position_embeddings": 102400})
    contract = context_contract({}, {"max_position_embeddings": 262144})
    with pytest.raises(ValueError):
        request_context(contract, 10, 1, 2)
