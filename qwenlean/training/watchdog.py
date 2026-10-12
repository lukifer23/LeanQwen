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
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

from qwenlean.utils.io import write_json

GIB = 1024**3


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
        ready = json.loads((Path(output)/'watchdog-ready.json').read_text())
        monitor = psutil.Process(ready['pid'])
        valid = (ready['parent_pid'] == os.getpid() and monitor.ppid() == os.getpid()
                 and same_process(monitor, ready['pid'], ready['create_time']))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, psutil.NoSuchProcess):
        valid = False
    if not valid:
        raise RuntimeError('Live owned CPU watchdog required before model loading')


@contextmanager
def training_watchdog(config, output):
    output = Path(output)
    owner = psutil.Process()
    limits, baseline = policy(config), snapshot(owner)
    admission = {'policy': limits, 'baseline': baseline, 'pid': owner.pid,
                 'create_time': owner.create_time()}
    if baseline['available_ram_bytes'] < max(limits['admission_ram_bytes'], limits['min_available_ram_bytes']):
        admission['status'] = 'insufficient_starting_ram'
    elif baseline['swap_used_bytes'] > limits['max_start_swap_used_bytes']:
        admission['status'] = 'starting_swap_pressure'
    else:
        admission['status'] = 'admitted'
    write_json(output/'resource-admission.json', admission)
    if admission['status'] != 'admitted':
        raise RuntimeError('Training resource admission refused before model loading: '+admission['status'])
    log_handle = (output/'watchdog-console.log').open('x')
    monitor = subprocess.Popen([sys.executable, '-m', 'qwenlean.training.watchdog',
        '--admission', str((output/'resource-admission.json').resolve()),
        '--output', str(output.resolve())], stdout=log_handle, stderr=log_handle)
    try:
        deadline = time.monotonic()+5
        while not (output/'watchdog-ready.json').exists():
            if monitor.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Independent watchdog did not start; model loading refused')
            time.sleep(0.01)
        yield
        if monitor.poll() is not None:
            raise RuntimeError('Independent training watchdog stopped unexpectedly or interrupted the workload')
    finally:
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
    write_json(output/'watchdog-ready.json', {'pid': os.getpid(), 'parent_pid': pid,
        'create_time': psutil.Process().create_time()})
    while True:
        try:
            if not same_process(owner, pid, created):
                return
            current = snapshot(owner)
        except psutil.NoSuchProcess:
            return
        stage = None
        try:
            stage = json.loads((output/'active-microbatch.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        reason = violation(current, baseline, limits, stage)
        if reason:
            write_json(output/'watchdog-stop.json', {'reason': reason, 'snapshot': current,
                'baseline': baseline, 'policy': limits, 'stage': stage,
                'verified_parent_pid': pid, 'initial_signal': 'SIGINT'})
            if same_process(owner, pid, created):
                owner.send_signal(signal.SIGINT)
            time.sleep(1)
            try:
                if same_process(owner, pid, created):
                    # A native call can delay Python signal handling. Preserve
                    # the independent stop record before terminating our parent.
                    write_json(output/'watchdog-escalation.json', {'signal': 'SIGTERM', 'verified_parent_pid': pid})
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
