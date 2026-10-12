"""Independent CPU watchdog decisions and interruption of an owned CPU fixture."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qwenlean.training.watchdog import (
    monitor_parent,
    policy,
    require_active_watchdog,
    training_watchdog,
    violation,
)


def states():
    base = {'available_ram_bytes': 12*1024**3, 'swap_used_bytes': 0, 'swap_out_bytes': 0,
            'rss_bytes': 1000, 'monotonic_s': 100}
    return base, policy({})


@pytest.mark.parametrize('changed,expected', [
    ({'available_ram_bytes': 3*1024**3}, 'available_ram_floor'),
    ({'swap_used_bytes': 200*1024**2}, 'swap_used_growth'),
    ({'swap_out_bytes': 600*1024**2}, 'swap_out_growth'),
    ({'rss_bytes': 4*1024**3}, 'process_rss_ceiling'),
    ({'monotonic_s': 221}, 'training_wall_timeout'),
])
def test_independent_pressure_checks(changed, expected):
    baseline, limits = states()
    assert violation(baseline, baseline, limits) is None
    assert violation({**baseline, **changed}, baseline, limits) == expected


def test_swap_occupancy_growth_is_checked_even_with_unchanged_swap_out_counter():
    baseline, limits = states()
    current = {**baseline, 'swap_used_bytes': 128*1024**2+1}
    assert current['swap_out_bytes'] == baseline['swap_out_bytes']
    assert violation(current, baseline, limits) == 'swap_used_growth'


def test_gradient_timeout_does_not_apply_to_completed_gradient_or_reload():
    baseline, limits = states()
    current = {**baseline, 'monotonic_s': 120}
    assert violation(current, baseline, limits, {'phase': 'gradient', 'started_monotonic_s': 100}) == 'gradient_wall_timeout'
    assert violation(current, baseline, limits, {'phase': 'gradient_completed'}) is None


def test_watchdog_cannot_monitor_or_signal_an_unrelated_pid(tmp_path):
    path = tmp_path/'admission.json'
    path.write_text(json.dumps({'pid': 0, 'create_time': 0}))
    with pytest.raises(RuntimeError, match='own parent'):
        monitor_parent(path, tmp_path)


def test_admission_refuses_before_workload_or_monitor_start(tmp_path):
    with pytest.raises(RuntimeError, match='before model loading'):
        with training_watchdog({'training_admission_ram_bytes': 10**15}, tmp_path):
            raise AssertionError('An inadmissible workload ran')
    assert json.loads((tmp_path/'resource-admission.json').read_text())['status'] == 'insufficient_starting_ram'
    assert not (tmp_path/'watchdog-ready.json').exists()


def test_missing_or_forged_watchdog_cannot_authorize_private_training(tmp_path):
    with pytest.raises(RuntimeError, match='before model loading'):
        require_active_watchdog(tmp_path)
    (tmp_path/'watchdog-ready.json').write_text(json.dumps({'pid': 0, 'parent_pid': 0, 'create_time': 0}))
    with pytest.raises(RuntimeError, match='before model loading'):
        require_active_watchdog(tmp_path)


def test_watchdog_interrupts_only_its_owned_sleeping_cpu_fixture(tmp_path):
    # No MLX import, model weights, training target or GPU work in this fixture.
    code = '''import time
from qwenlean.training.watchdog import require_active_watchdog, training_watchdog
try:
    with training_watchdog(CONFIG, OUTPUT):
        require_active_watchdog(OUTPUT)
        time.sleep(4)
except KeyboardInterrupt:
    print('owned_cpu_fixture_interrupted')
else:
    raise RuntimeError('Watchdog failed to interrupt the CPU fixture')
'''.replace('CONFIG', repr({'min_available_ram_bytes': 0, 'training_admission_ram_bytes': 0,
    'max_start_swap_used_bytes': 10**15, 'max_swap_used_growth_bytes': 10**15,
    'max_swap_out_delta_bytes': 10**15, 'max_process_rss_bytes': 10**15, 'max_training_wall_s': 1}))
    code = code.replace('OUTPUT', repr(str(tmp_path)))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            timeout=8, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    assert 'owned_cpu_fixture_interrupted' in result.stdout
    stopped = json.loads((tmp_path/'watchdog-stop.json').read_text())
    assert stopped['reason'] == 'training_wall_timeout'
    assert stopped['initial_signal'] == 'SIGINT'
