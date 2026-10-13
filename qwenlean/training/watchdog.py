"""CPU-only admission and independent supervision of this process's SFT child.

The monitor may interrupt only its own verified parent. It continues polling when
the trainer is inside native MLX evaluation; no MLX import or model allocation.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import psutil

from qwenlean.utils.io import write_json

GIB = 1024**3


def durable_json(path, value):
    """Atomic journal publication, durable before any interrupt signal."""
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    with temporary.open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def set_phase(output, phase, **details):
    durable_json(Path(output)/'active-microbatch.json', {'phase': phase,
        'started_monotonic_s': time.monotonic(), **details})


def policy(config):
    return {
        'min_available_ram_bytes': config.get('min_available_ram_bytes', 4*GIB),
        'admission_ram_bytes': config.get('training_admission_ram_bytes', 8*GIB),
        'max_start_swap_used_bytes': config.get('max_start_swap_used_bytes', 2*GIB),
        'max_swap_used_growth_bytes': config.get('max_swap_used_growth_bytes', 128*1024**2),
        'max_swap_out_delta_bytes': config.get('max_swap_out_delta_bytes', 512*1024**2),
        'max_process_rss_bytes': config.get('max_process_rss_bytes', 3*GIB),
        'max_training_wall_s': config.get('max_training_wall_s', 120),
        'max_gradient_wall_s': config.get('max_gradient_wall_s', 15),
        'poll_s': 0.1,
    }


def snapshot(process):
    swap = psutil.swap_memory()
    return {'available_ram_bytes': psutil.virtual_memory().available,
            'swap_used_bytes': swap.used, 'swap_out_bytes': swap.sout,
            'rss_bytes': process.memory_info().rss, 'monotonic_s': time.monotonic()}


def violation(current, baseline, limits, stage=None):
    if current['available_ram_bytes'] < limits['min_available_ram_bytes']:
        return 'available_ram_floor'
    if current['swap_used_bytes']-baseline['swap_used_bytes'] > limits['max_swap_used_growth_bytes']:
        return 'swap_used_growth'
    if current['swap_out_bytes']-baseline['swap_out_bytes'] > limits['max_swap_out_delta_bytes']:
        return 'swap_out_growth'
    if current['rss_bytes'] > limits['max_process_rss_bytes']:
        return 'process_rss_ceiling'
    if current['monotonic_s']-baseline['monotonic_s'] > limits['max_training_wall_s']:
        return 'training_wall_timeout'
    if (stage and stage.get('phase') == 'gradient'
        and current['monotonic_s']-stage['started_monotonic_s'] > limits['max_gradient_wall_s']):
        return 'gradient_wall_timeout'
    return None


def same_process(process, pid, created):
    return process.pid == pid and abs(process.create_time()-created) < 0.01


def require_active_watchdog(output):
    try:
        admission = json.loads((Path(output)/'resource-admission.json').read_text())
        ready = json.loads((Path(output)/'watchdog-ready.json').read_text())
        status = json.loads((Path(output)/'watchdog-status.json').read_text())
        monitor = psutil.Process(ready['pid'])
        valid = (ready['parent_pid'] == os.getpid() and monitor.ppid() == os.getpid()
                 and same_process(monitor, ready['pid'], ready['create_time'])
                 and admission['pid'] == os.getpid()
                 and same_process(psutil.Process(), admission['pid'], admission['create_time'])
                 and admission['session_id'] == ready['session_id'] == status['session_id']
                 and status['state'] == 'running'
                 and 0 <= time.monotonic()-status['monotonic_s'] < 2)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, psutil.Error):
        valid = False
    valid = valid and not any((Path(output)/name).exists() for name in
        ('watchdog-stop.json', 'watchdog-supervisor-stop.json', 'watchdog-release.json'))
    if not valid:
        raise RuntimeError('Live owned CPU watchdog required before model loading')


@contextmanager
def training_watchdog(config, output):
    output = Path(output)
    owner = psutil.Process()
    if any((output/name).exists() for name in ('resource-admission.json', 'watchdog-ready.json',
                                             'watchdog-status.json', 'watchdog-release.json')):
        raise RuntimeError('Refuse stale watchdog state; a fresh supervision destination is required')
    limits, baseline = policy(config), snapshot(owner)
    admission = {'policy': limits, 'baseline': baseline, 'pid': owner.pid,
                 'create_time': owner.create_time(), 'session_id': uuid4().hex}
    if baseline['available_ram_bytes'] < max(limits['admission_ram_bytes'], limits['min_available_ram_bytes']):
        admission['status'] = 'insufficient_starting_ram'
    elif baseline['swap_used_bytes'] > limits['max_start_swap_used_bytes']:
        admission['status'] = 'starting_swap_pressure'
    else:
        admission['status'] = 'admitted'
    durable_json(output/'resource-admission.json', admission)
    if admission['status'] != 'admitted':
        raise RuntimeError('Training resource admission refused before model loading: '+admission['status'])
    log_handle = (output/'watchdog-console.log').open('x')
    monitor = subprocess.Popen([sys.executable, '-m', 'qwenlean.training.watchdog',
        '--admission', str((output/'resource-admission.json').resolve()),
        '--output', str(output.resolve())], stdout=log_handle, stderr=log_handle)
    stopped = threading.Event()
    failure = []

    def supervise_monitor():
        while not stopped.wait(0.1):
            reason = None
            if monitor.poll() is not None:
                reason = 'monitor_exited'
            else:
                try:
                    status = json.loads((output/'watchdog-status.json').read_text())
                    if status['session_id'] != admission['session_id'] or time.monotonic()-status['monotonic_s'] > 2:
                        reason = 'monitor_heartbeat_failed'
                except (OSError, KeyError, json.JSONDecodeError):
                    reason = 'monitor_heartbeat_unreadable'
            if reason:
                # Never wait for a native workload to finish to detect monitor death.
                # This thread can signal only the process it lives in.
                failure.append(reason)
                try:
                    durable_json(output/'watchdog-supervisor-stop.json', {'reason': reason,
                        'parent_pid': owner.pid, 'session_id': admission['session_id'],
                        'monotonic_s': time.monotonic(), 'initial_signal': 'SIGINT',
                        'snapshot': snapshot(owner), 'baseline': baseline})
                finally:
                    if not stopped.is_set() and same_process(psutil.Process(), owner.pid, admission['create_time']):
                        os.kill(os.getpid(), signal.SIGINT)
                        if not stopped.wait(1):
                            durable_json(output/'watchdog-supervisor-escalation.json',
                                {'signal': 'SIGTERM', 'verified_parent_pid': owner.pid})
                            os.kill(os.getpid(), signal.SIGTERM)
                return

    guardian = None
    try:
        deadline = time.monotonic()+5
        while not (output/'watchdog-ready.json').exists():
            if monitor.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Independent watchdog did not start; model loading refused')
            time.sleep(0.01)
        require_active_watchdog(output)
        guardian = threading.Thread(target=supervise_monitor, name='watchdog-liveness', daemon=True)
        guardian.start()
        yield
        if failure or monitor.poll() is not None or (output/'watchdog-stop.json').exists():
            raise RuntimeError('Independent training watchdog stopped unexpectedly or interrupted the workload')
        reason = violation(snapshot(owner), baseline, limits)
        if reason:
            durable_json(output/'watchdog-final-stop.json', {'reason': reason,
                'snapshot': snapshot(owner), 'baseline': baseline})
            raise RuntimeError('Training resource threshold exceeded at final acceptance: '+reason)
    finally:
        stopped.set()
        if guardian:
            guardian.join(timeout=2)
        durable_json(output/'watchdog-release.json', {'session_id': admission['session_id'],
            'parent_pid': owner.pid, 'monotonic_s': time.monotonic()})
        if monitor.poll() is None:
            monitor.terminate()
        monitor.wait(timeout=5)
        log_handle.close()


def monitor_parent(admission_path, output):
    admission = json.loads(Path(admission_path).read_text())
    pid, created = admission['pid'], admission['create_time']
    if pid != os.getppid():
        raise RuntimeError('Watchdog may monitor only its own parent, never an unrelated application')
    owner = psutil.Process(pid)
    if not same_process(owner, pid, created):
        raise RuntimeError('Parent process identity differs')
    output = Path(output)
    baseline, limits = admission['baseline'], admission['policy']
    durable_json(output/'watchdog-status.json', {'state': 'running', 'session_id': admission['session_id'],
        'monotonic_s': time.monotonic()})
    durable_json(output/'watchdog-ready.json', {'pid': os.getpid(), 'parent_pid': pid,
        'create_time': psutil.Process().create_time(), 'session_id': admission['session_id']})
    while True:
        try:
            if not same_process(owner, pid, created):
                return
            current = snapshot(owner)
        except psutil.NoSuchProcess:
            return
        except psutil.Error as exc:
            durable_json(output/'watchdog-monitor-error.json', {'error': str(exc),
                'session_id': admission['session_id']})
            raise
        stage = None
        try:
            stage = json.loads((output/'active-microbatch.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        reason = violation(current, baseline, limits, stage)
        # Atomic heartbeat; only stop/ready/release records require an fsync.
        status_path = output/'watchdog-status.json'
        temporary = status_path.with_suffix('.tmp')
        write_json(temporary, {'state': 'stopping' if reason else 'running',
            'session_id': admission['session_id'], 'monotonic_s': current['monotonic_s'],
            'snapshot': current, 'stage': stage})
        os.replace(temporary, status_path)
        if reason:
            durable_json(output/'watchdog-stop.json', {'reason': reason, 'snapshot': current,
                'baseline': baseline, 'policy': limits, 'stage': stage,
                'verified_parent_pid': pid, 'initial_signal': 'SIGINT'})
            if same_process(owner, pid, created):
                owner.send_signal(signal.SIGINT)
            time.sleep(1)
            try:
                if same_process(owner, pid, created):
                    # A native call can delay Python signal handling. Preserve
                    # the independent stop record before terminating our parent.
                    durable_json(output/'watchdog-escalation.json', {'signal': 'SIGTERM', 'verified_parent_pid': pid})
                    owner.terminate()
            except psutil.NoSuchProcess:
                pass
            return
        time.sleep(limits['poll_s'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--admission', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    monitor_parent(args.admission, args.output)
