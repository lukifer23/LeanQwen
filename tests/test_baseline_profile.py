"""CPU-only checks of the inference supervision profile; no admission or model."""

import json
from pathlib import Path

import pytest
import yaml

import scripts.phase4_dev as driver
from scripts.phase4_dev import GIB, INFERENCE_PROFILE, validate_resource_profile

PROTOCOL = json.loads(Path('reports/external-human-dev-protocol.json').read_text())


def profile():
    return yaml.safe_load(Path(INFERENCE_PROFILE).read_text())


def test_tracked_inference_profile_keeps_reserve_and_measured_headroom():
    checked = validate_resource_profile(profile(), PROTOCOL)
    assert checked['workload'] == 'inference_only'
    assert checked['min_available_ram_bytes'] == 4*GIB
    assert checked['training_admission_ram_bytes'] == int(6.5*GIB)
    assert not {'steps', 'learning_rate', 'lora_parameters'} & set(checked)


def test_training_smoke_admission_is_not_relaxed_by_the_inference_profile():
    smoke = yaml.safe_load(Path('configs/training/external-human-smoke-supervised.yaml').read_text())
    assert smoke['training_admission_ram_bytes'] == 8*GIB
    assert smoke['min_available_ram_bytes'] == 4*GIB
    assert smoke['max_start_swap_used_bytes'] == 2*GIB


@pytest.mark.parametrize('change', [
    {'min_available_ram_bytes': 3*GIB},
    {'training_admission_ram_bytes': 5*GIB},
    {'max_start_swap_used_bytes': 3*GIB},
    {'max_swap_used_growth_bytes': 256*1024**2},
    {'max_swap_out_delta_bytes': GIB},
    {'max_process_rss_bytes': 4*GIB},
    {'mlx_memory_limit_bytes': 4*GIB},
    {'max_training_wall_s': 600},
])
def test_weakened_profiles_fail_closed(change):
    with pytest.raises(ValueError, match='weakens'):
        validate_resource_profile({**profile(), **change}, PROTOCOL)


def test_incomplete_profile_fails_closed():
    incomplete = profile()
    incomplete.pop('max_process_rss_bytes')
    with pytest.raises(ValueError, match='every positive integer'):
        validate_resource_profile(incomplete, PROTOCOL)


def test_serial_baseline_stops_at_first_failure_without_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(driver, 'inspect_cohort', lambda *a: ({'status': 'incomplete'}, [], []))
    monkeypatch.setattr(driver.subprocess, 'call', lambda command, **k: calls.append(command) or 2)
    outcome = driver.serial('cohort', 'out', max_invocations=5)
    assert outcome == {'status': 'stopped_without_retry', 'invocations': 1, 'exit_code': 2}
    assert len(calls) == 1 and '--retry-failed' not in calls[0]


def test_serial_baseline_starts_nothing_when_cohort_is_complete(monkeypatch):
    monkeypatch.setattr(driver, 'inspect_cohort', lambda *a: ({'status': 'complete'}, [], []))
    monkeypatch.setattr(driver.subprocess, 'call', lambda *a, **k: pytest.fail('must not launch'))
    assert driver.serial('cohort', 'out', max_invocations=3) == {'status': 'complete', 'invocations': 0}
    with pytest.raises(ValueError, match='1..112'):
        driver.serial('cohort', 'out', max_invocations=500)
