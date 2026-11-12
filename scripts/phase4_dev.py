"""Frozen matched stock/adapter DEV cohort. One bounded workload under the lease."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import yaml

from qwenlean.datasets.prompts import render_task
from qwenlean.evaluation.single_job import evaluate_single, inspect_single
from qwenlean.evaluation.summary import summarize
from qwenlean.training.watchdog import set_phase
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl

GIB = 1024**3
INFERENCE_PROFILE = 'configs/evaluation/stock-dev-inference-supervised.yaml'
# Measured stock 2048-token inference RSS peak was 2.140 GB; require 2 GiB of
# admission headroom above the running reserve whatever profile is supplied.
INFERENCE_HEADROOM_BYTES = 2*GIB


def validate_resource_profile(profile, protocol):
    """Hard floors for any baseline supervision profile; a file cannot weaken them."""
    reserve = max(c.get('min_available_ram_bytes', 0) for c in protocol['configs'].values())
    required = ('min_available_ram_bytes', 'training_admission_ram_bytes', 'max_start_swap_used_bytes',
                'max_swap_used_growth_bytes', 'max_swap_out_delta_bytes', 'max_process_rss_bytes',
                'mlx_cache_limit_bytes', 'mlx_memory_limit_bytes', 'max_training_wall_s')
    if any(type(profile.get(k)) is not int or profile[k] <= 0 for k in required):
        raise ValueError('Baseline resource profile must state every positive integer limit')
    if (profile['min_available_ram_bytes'] < max(reserve, 4*GIB)
            or profile['training_admission_ram_bytes'] < profile['min_available_ram_bytes']+INFERENCE_HEADROOM_BYTES
            or profile['max_start_swap_used_bytes'] > 2*GIB
            or profile['max_swap_used_growth_bytes'] > 128*1024**2
            or profile['max_swap_out_delta_bytes'] > 512*1024**2
            or profile['max_process_rss_bytes'] > 3*GIB
            or profile['mlx_memory_limit_bytes'] > 3*GIB
            or profile['max_training_wall_s'] > 120):
        raise ValueError('Baseline resource profile weakens a supervised safety limit')
    return profile


def freeze(corpus, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Refuse to change frozen cohort')
    tasks = []
    for name in ('dev-id', 'dev-ood'):
        rows = read_jsonl(Path(corpus)/f'{name}.jsonl')
        for i in range(0, len(rows), 4):
            tasks.extend([render_task(rows[i], 'P2'), render_task(rows[i+3], 'P2')])
    base = yaml.safe_load(Path('configs/baseline.yaml').read_text())
    base.update(seed=20261012, max_output_tokens=2048, context_profile='mac-128k',
                prefill_step_size=512, prompt_policy='P2', replicates=1,
                min_available_ram_bytes=4*1024**3, loop_guard={'enabled': False})
    configs = {}
    for thinking in (True, False):
        name = 'thinking' if thinking else 'nonthinking'
        config = {**base, 'name': 'phase4-matched-'+name, 'enable_thinking': thinking}
        if not thinking:
            config['sampling'] = yaml.safe_load(Path('configs/nonthinking.yaml').read_text())['sampling']
        configs[name] = config
    destination.mkdir(parents=True)
    write_jsonl(destination/'tasks.jsonl', tasks)
    write_json(destination/'protocol.json', {'status': 'frozen_before_training', 'task_hash': digest(tasks),
        'task_count': len(tasks), 'configs': configs, 'configs_hash': digest(configs),
        'corpus_manifest_hash': digest(json.loads((Path(corpus)/'manifest.json').read_text())),
        'one_seed_limitation': 'Task-cluster paired comparisons; one replicate cannot estimate sampler variance.',
        'test_used': False, 'selection_rule': 'First and last structural complexity within each family/holdout; no model outcomes used'})


def inspect_cohort(cohort, output, adapter=None):
    cohort, output = Path(cohort), Path(output)
    protocol = json.loads((cohort/'protocol.json').read_text())
    tasks = read_jsonl(cohort/'tasks.jsonl')
    if protocol['task_hash'] != digest(tasks) or protocol['configs_hash'] != digest(protocol['configs']):
        raise ValueError('Frozen protocol mismatch')
    contract = {'protocol_hash': digest(protocol), 'adapter': adapter}
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text()) != contract:
        raise ValueError('Evaluation contract changed')
    model_info = json.loads((output/'model.json').read_text()) if (output/'model.json').exists() else None
    records, failures, pending = [], [], []
    for mode, frozen in protocol['configs'].items():
        config = {**frozen, **({'adapter_path': adapter} if adapter else {})}
        for i, task in enumerate(tasks):
            item = inspect_single(task, config, output/mode/f'{i:03d}', model_info)
            failures += [{'mode': mode, 'task_id': task['task_id'], **f} for f in item['failed_attempts']]
            if item['record']:
                row = {**item['record'], 'phase4_mode': mode}
                records.append(row)
                model_info = row['model_identifier'] if model_info is None else model_info
            else:
                pending.append({'mode': mode, 'index': i, 'task': task, 'config': config, 'prior_status': item['status']})
    result = {'status': 'complete' if not pending else 'incomplete', 'contract': contract,
              'completed_jobs': len(records), 'planned_jobs': len(tasks)*len(protocol['configs']),
              'infrastructure_failed_attempts': failures, 'accuracy': None}
    events = []
    for path in sorted((output/'supervision').glob('*/*.json')):
        if path.name == 'resource-admission.json':
            event = json.loads(path.read_text())
            if event['status'] != 'admitted':
                events.append({'path': str(path), 'kind': 'resource_admission_refused', 'record': event})
        elif path.name in ('watchdog-stop.json', 'watchdog-supervisor-stop.json', 'watchdog-final-stop.json'):
            events.append({'path': str(path), 'kind': 'resource_stopped_invocation',
                           'record': json.loads(path.read_text())})
    result['resource_events'] = events
    if not pending:
        result['modes'] = {m: summarize([r for r in records if r['phase4_mode'] == m]) for m in protocol['configs']}
    return result, records, pending


def run(cohort, output, adapter=None, batch_jobs=2, retry_failed=False, resource_profile=INFERENCE_PROFILE):
    from qwenlean.training.watchdog import training_watchdog
    from qwenlean.utils.process_lock import model_process_lock
    if type(batch_jobs) is not int or not 1 <= batch_jobs <= 2:
        raise ValueError('Bounded baseline invocation permits one or two new jobs')
    cohort, output = Path(cohort), Path(output)
    result, rows, pending = inspect_cohort(cohort, output, adapter)
    output.mkdir(parents=True, exist_ok=True)
    if not (output/'contract.json').exists():
        write_json(output/'contract.json', result['contract'])
    write_json(output/'progress.json', result)
    if not pending:
        return result
    if any(j['prior_status'] == 'failed' for j in pending[:batch_jobs]) and not retry_failed:
        raise ValueError('Failed job preserved; --retry-failed explicitly permits a new unique attempt')
    protocol = json.loads((cohort/'protocol.json').read_text())
    resource_config = validate_resource_profile(yaml.safe_load(Path(resource_profile).read_text()), protocol)
    safety_hash = digest(resource_config)
    safety_path = output/'resource-contract.json'
    if safety_path.exists() and json.loads(safety_path.read_text())['config_hash'] != safety_hash:
        raise ValueError('Baseline operational resource policy changed')
    if not safety_path.exists():
        write_json(safety_path, {'config_hash': safety_hash, 'config': resource_config,
                                'profile_path': str(resource_profile),
                                'frozen_protocol_unchanged': True, 'batch_jobs_max': 2})
    invocation = output/'supervision'/uuid4().hex
    invocation.mkdir(parents=True)
    with model_process_lock(), training_watchdog(resource_config, invocation):
        # Do not even import the GPU backend until fresh resource admission passes.
        import mlx.core as mx

        from qwenlean.inference.mlx_backend import MLXBackend
        old_cache = mx.set_cache_limit(resource_config['mlx_cache_limit_bytes'])
        old_memory = mx.set_memory_limit(resource_config['mlx_memory_limit_bytes'])
        try:
            initial = pending[0]['config']
            set_phase(invocation, 'model_loading')
            with MLXBackend(initial) as backend:
                saved = output/'model.json'
                if saved.exists() and json.loads(saved.read_text()) != backend.model_info:
                    raise ValueError('Loaded stock/adapter model differs from bound baseline model')
                if not saved.exists():
                    write_json(saved, backend.model_info)
                for job in pending[:batch_jobs]:
                    set_phase(invocation, 'inference', task_id=job['task']['task_id'], mode=job['mode'])
                    evaluate_single(backend, job['task'], job['config'], output/job['mode']/f"{job['index']:03d}",
                                    retry_failed=retry_failed)
                    current, records, _ = inspect_cohort(cohort, output, adapter)
                    write_jsonl(output/'records.jsonl', records)
                    write_json(output/'progress.json', current)
        finally:
            mx.clear_cache()
            mx.set_cache_limit(old_cache)
            mx.set_memory_limit(old_memory)
    result, rows, _ = inspect_cohort(cohort, output, adapter)
    write_jsonl(output/'records.jsonl', rows)
    write_json(output/'progress.json', result)
    if result['status'] == 'complete':
        write_json(output/'report.json', result)
    return result


def serial(cohort, output, adapter=None, batch_jobs=1, resource_profile=INFERENCE_PROFILE,
           max_invocations=112):
    """Serial bounded invocations, each a new process with its own fresh admission.

    This parent never imports MLX or holds the lease. The first refusal, resource
    stop or failed job ends the sequence: nothing is retried and failed jobs are
    never reattempted here.
    """
    if type(max_invocations) is not int or not 1 <= max_invocations <= 112:
        raise ValueError('Serial baseline permits 1..112 invocations')
    command = [sys.executable, str(Path(__file__).resolve()), '--cohort', str(cohort), '--output', str(output),
               '--batch-jobs', str(batch_jobs), '--resource-profile', str(resource_profile)]
    if adapter:
        command += ['--adapter', str(adapter)]
    for started in range(max_invocations):
        if inspect_cohort(cohort, output, adapter)[0]['status'] == 'complete':
            return {'status': 'complete', 'invocations': started}
        code = subprocess.call(command, stdout=subprocess.DEVNULL)
        if code != 0:
            return {'status': 'stopped_without_retry', 'invocations': started+1, 'exit_code': code}
    final = inspect_cohort(cohort, output, adapter)[0]['status']
    return {'status': final, 'invocations': max_invocations}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corpus')
    p.add_argument('--cohort', required=True)
    p.add_argument('--output')
    p.add_argument('--adapter')
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--inspect', action='store_true', help='CPU-only saved-job inventory; no model loading')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--batch-jobs', type=int, default=2)
    p.add_argument('--resource-profile', default=INFERENCE_PROFILE,
                   help='Supervision limits; validated against hard floors and bound to the output')
    p.add_argument('--serial-invocations', type=int,
                   help='Run up to N fresh single-batch processes; stop at the first refusal or failure')
    a = p.parse_args()
    if a.serial_invocations:
        if a.retry_failed or a.freeze or a.inspect:
            raise SystemExit('--serial-invocations never retries, freezes or inspects')
        outcome = serial(a.cohort, a.output, a.adapter, a.batch_jobs, a.resource_profile, a.serial_invocations)
        print(json.dumps(outcome, indent=2))
        raise SystemExit(0 if outcome['status'] == 'complete' else 2)
    if a.freeze:
        freeze(a.corpus, a.cohort)
    elif a.inspect:
        print(json.dumps(inspect_cohort(a.cohort, a.output, a.adapter)[0], indent=2))
    else:
        print(json.dumps(run(a.cohort, a.output, a.adapter, a.batch_jobs, a.retry_failed, a.resource_profile), indent=2))
