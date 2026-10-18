"""CPU-only evidence inventory. Never starts admission, inference or training.

A saved admission is historical evidence, never permission for another process.
Run from the repository root; missing local archives remain explicit errors.
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from qwenlean.training.checkpoint_audit import require_approved_adapter
from qwenlean.training.quality import load_approved_training_inputs
from qwenlean.training.watchdog import policy
from qwenlean.utils.io import digest, read_jsonl
from qwenlean.utils.process_lock import model_process_lock
from scripts.phase4_dev import inspect_cohort

CONFIG = Path('configs/training/external-human-smoke-supervised.yaml')
POOLS = {
    'smoke': Path('data/processed/external-human-smoke8-v1'),
    'pilot': Path('data/processed/external-human-pilot-v1-approved'),
}
COHORT = Path('data/generated/external-human-v1-frozen-dev')
BASELINE = Path('runs/phase4c-stock-dev')
PROTOCOL = Path('reports/external-human-dev-protocol.json')


def file_evidence(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            hasher.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hasher.hexdigest()}


def inspect_pool(root, config):
    rows, manifest = load_approved_training_inputs(config, root/'train.jsonl', root/'manifest.json')
    lengths = [len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in rows]
    return {'status': 'passed', 'records': len(rows), 'dataset_hash': digest(rows),
            'manifest_hash': digest(manifest), 'sequence_length_min': min(lengths),
            'sequence_length_max': max(lengths), 'quality_basis': manifest['quality_approval_basis'],
            'dataset': file_evidence(root/'train.jsonl'),
            'manifest': file_evidence(root/'manifest.json')}, rows


def inspect_frozen_dev(cohort, baseline, tracked_protocol, train_rows):
    protocol = json.loads((cohort/'protocol.json').read_text())
    published = json.loads(tracked_protocol.read_text())
    tasks = read_jsonl(cohort/'tasks.jsonl')
    if ({k: v for k, v in published.items() if k not in ('cohort_path', 'task_ids')} != protocol
        or published['task_ids'] != [t['task_id'] for t in tasks]
        or protocol['task_hash'] != digest(tasks)
        or protocol['configs_hash'] != digest(protocol['configs'])
        or protocol['task_count'] != len(tasks) or len(tasks) != 56
        or protocol['test_used'] is not False
        or set(protocol['configs']) != {'thinking', 'nonthinking'}
        or len({t['task_id'] for t in tasks}) != len(tasks)
        or any(t['split'] != 'dev' for t in tasks)):
        raise ValueError('Frozen DEV differs from the published 56-task protocol')
    if {r['task_id'] for r in train_rows} & {t['task_id'] for t in tasks}:
        raise ValueError('TRAIN/DEV task overlap')
    if ({r['semantic_structure_fingerprint'] for r in train_rows}
        & {t['semantic_structure_fingerprint'] for t in tasks
           if 'semantic_structure_fingerprint' in t}):
        raise ValueError('TRAIN/DEV normalized source structure overlap')
    result, _, pending = inspect_cohort(cohort, baseline)
    return {'status': 'passed', 'protocol_hash': digest(protocol),
            'protocol': file_evidence(tracked_protocol), 'inventory': result,
            'pending_by_mode': dict(Counter(j['mode'] for j in pending))}


def inspect_admission(path, config):
    record = json.loads(path.read_text())
    if record['policy'] != policy(config):
        raise ValueError('Saved admission used a different resource policy')
    observed, limits = record['baseline'], record['policy']
    failed = []
    if observed['available_ram_bytes'] < max(limits['admission_ram_bytes'], limits['min_available_ram_bytes']):
        failed.append('starting_available_ram')
    if observed['swap_used_bytes'] > limits['max_start_swap_used_bytes']:
        failed.append('starting_occupied_swap')
    expected = ('insufficient_starting_ram' if 'starting_available_ram' in failed
                else 'starting_swap_pressure' if failed else 'admitted')
    if record['status'] != expected:
        raise ValueError('Saved admission status contradicts its measurements')
    return {'status': 'recorded_refusal' if failed else 'historical_admission_only',
            'evidence': file_evidence(path), 'record': record, 'failed_gates': failed,
            'available_ram_gib': observed['available_ram_bytes']/1024**3,
            'occupied_swap_gib': observed['swap_used_bytes']/1024**3,
            'authorizes_model_loading': False}


def inspect_adapters(root, config):
    attempts = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        completion_path = path/'completion.json'
        completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
        logs = read_jsonl(path/'loss.jsonl') if (path/'loss.jsonl').exists() else []
        item = {'path': str(path), 'completion': completion,
                'logged_optimizer_updates': [r['optimizer_steps'] for r in logs],
                'quarantined': (path/'quarantine.json').exists(), 'approved': False}
        if completion and completion.get('status') == 'complete':
            item['audit'] = require_approved_adapter(path, config)
            saved = json.loads((path/'config.json').read_text())
            item['approved'] = True
            item['matches_supervised_four_step_config'] = (
                saved == config and config['steps'] == completion['optimizer_steps'] == 4
                and item['logged_optimizer_updates'] == [1, 2, 3, 4])
        attempts.append(item)
    return {'status': 'passed', 'attempts': attempts,
            'approved_adapter_count': sum(a['approved'] for a in attempts),
            'matching_four_step_smoke_count': sum(a.get('matches_supervised_four_step_config', False)
                                                 for a in attempts)}


def audit(admission_path):
    config = yaml.safe_load(CONFIG.read_text())
    result = {'audit_version': 'project-readiness-v1',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'git_worktree': subprocess.check_output(['git', 'status', '--short'], text=True).splitlines(),
              'audit_implementation': file_evidence(__file__),
              'config': file_evidence(CONFIG), 'checks': {},
              'model_execution_authorized': False, 'improvement_sft_authorized': False,
              'scope': 'CPU archives/tokenizer only; no resource admission or model workload. '
                       'Saved admissions do not establish present resource availability.'}
    checks, pools = result['checks'], {}

    def capture(name, action):
        try:
            checks[name] = action()
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
            checks[name] = {'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc)}

    def pool(name, root):
        summary, rows = inspect_pool(root, config)
        pools[name] = rows
        return summary

    # Also exclude concurrent model work while reading mutable local artifacts.
    with model_process_lock():
        for name, root in POOLS.items():
            capture(name+'_data', lambda name=name, root=root: pool(name, root))
        if 'pilot' in pools:
            capture('frozen_dev', lambda: inspect_frozen_dev(COHORT, BASELINE, PROTOCOL, pools['pilot']))
        else:
            checks['frozen_dev'] = {'status': 'unknown', 'reason': 'Pilot data validation failed'}
        capture('saved_admission', lambda: inspect_admission(Path(admission_path), config))
        capture('adapters', lambda: inspect_adapters(Path('adapters'), config))
    errors = [name for name, item in checks.items() if item['status'] in ('error', 'unknown')]
    result['status'] = 'invalid_or_missing_evidence' if errors else 'qualification_incomplete'
    result['evidence_errors'] = errors
    result['gates'] = {
        'source_and_complete_records': all(checks[n]['status'] == 'passed'
                                          for n in ('smoke_data', 'pilot_data')),
        'frozen_dev_verified': checks['frozen_dev']['status'] == 'passed',
        'stock_baseline_complete': checks['frozen_dev'].get('inventory', {}).get('status') == 'complete',
        'matching_four_step_smoke_approved': checks['adapters'].get('matching_four_step_smoke_count', 0) > 0,
        'full_pool_capacity': 'not_assessed_by_this_audit',
        'improvement_experiment': 'proposal_only',
    }
    result['next_action'] = ('Resolve evidence errors before model work' if errors else
        'After a resource change, one fresh admission and at most one supervised smoke; '
        'stop on refusal/failure. Complete serial monitored stock DEV batches only when qualified. '
        'Capacity and improvement SFT require their remaining gates and separate authorization.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', required=True, help='Existing CPU admission JSON; never rerun here')
    parser.add_argument('--output', required=True, help='New evidence file; existing files are protected')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refuse existing readiness evidence')
    result = audit(args.admission)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'gates': result['gates'],
                      'evidence_errors': result['evidence_errors'], 'output': str(output)}, indent=2))
    raise SystemExit(1 if result['evidence_errors'] else 0)


if __name__ == '__main__':
    main()
