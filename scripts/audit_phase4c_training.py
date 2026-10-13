"""Derived historical optimizer accounting; CPU file reads only, no model imports."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from qwenlean.training.checkpoint_audit import audit_lora_file
from qwenlean.utils.io import digest, read_jsonl, write_json


def audit(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Refuse existing derived training audit')
    runs = []
    for name in ('external-human-smoke-v1-attempt1', 'external-human-smoke-memory-efficient-attempt1',
                 'external-human-smoke-bounded-prefix-attempt1'):
        root = Path('adapters')/name
        logs = read_jsonl(root/'loss.jsonl')
        steps = [r['optimizer_steps'] for r in logs]
        if steps != list(range(1, len(logs)+1)) or any(r['step'] != r['optimizer_steps'] for r in logs):
            raise ValueError('Ambiguous/nonconsecutive optimizer records')
        if any(not math.isfinite(r['loss']) or r.get('finite_gradients') is not True for r in logs):
            raise ValueError('Invalid logged loss/gradient evidence')
        checkpoints = []
        for path in sorted(root.glob('step-*.safetensors')):
            checked = audit_lora_file(path)
            step = int(path.stem.split('-')[1])
            if not 1 <= step <= len(logs):
                raise ValueError('Checkpoint without its completed optimizer record')
            checked.update(step=step, matches_logged_adapter_hash=checked['tensor_fingerprint']==logs[step-1].get('adapter_hash'))
            if not checked['matches_logged_adapter_hash']:
                raise ValueError('Checkpoint tensor fingerprint differs from completed-step log')
            checkpoints.append(checked)
        active = json.loads((root/'active-microbatch.json').read_text()) if (root/'active-microbatch.json').exists() else None
        item = {'run_path': str(root), 'completed_optimizer_updates': len(logs), 'loss_history': logs,
                'active_microbatch': active, 'checkpoints': checkpoints,
                'completed_smoke': False, 'approved_adapter': False,
                'full_post_training_integrity': 'not_reached', 'save_reload_equivalence': 'not_reached',
                'fresh_adapter_inference': 'not_reached',
                'max_logged_mlx_peak_bytes': max((r['mlx_peak_bytes'] for r in logs), default=None),
                'raw_files': [{'path': str(p), 'bytes': p.stat().st_size,
                              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.iterdir()) if p.is_file()]}
        runs.append(item)
    original = Path('reports/external-human-smoke-bounded-prefix-interruption.json')
    result = {'audit_version': 'phase4c-historical-accounting-v1', 'starting_sha': 'a9f6bf1f9d66124f71175beb655e166557b18985',
              'runs': runs, 'completed_updates_across_independent_failed_runs': sum(r['completed_optimizer_updates'] for r in runs),
              'four_step_smoke_completed': False, 'gpu_used_for_audit': False,
              'original_report_path': str(original), 'original_report_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
              'correction': 'Unsafe frozen-prefix run completed updates1–3 and a matching intact step2 checkpoint; fourth microbatch was active when interrupted. Original whole-run zero-step/empty-loss accounting was erroneous. It remains preserved; this derived audit supersedes that accounting only.',
              'quarantine': 'Unsafe checkpoint remains in original ignored run directory, marked unapproved; no inference or publication of weights.',
              'counting_scope': 'Four updates across two independently initialized failed runs are not one completed four-step smoke.'}
    result['audit_hash'] = digest(result)
    write_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    r = audit(a.output)
    print(json.dumps({'completed_updates_by_run': {s['run_path']:s['completed_optimizer_updates'] for s in r['runs']},
                      'original_report_preserved': r['original_report_path'], 'audit_hash': r['audit_hash']}))
