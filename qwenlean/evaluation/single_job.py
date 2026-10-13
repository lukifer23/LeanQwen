"""Recover a one-trajectory diagnostic without duplicating a durable generation."""

import json
from pathlib import Path

from qwenlean.datasets.prompts import replicate_seed
from qwenlean.evaluation.runner import VALID_TERMINATIONS, evaluate, scoring_contract
from qwenlean.utils.io import digest, read_jsonl


def inspect_single(task, config, root, model_info=None):
    """CPU-only inventory of exact-contract durable attempts; failures stay separate."""
    if config.get('replicates', 1) != 1:
        raise ValueError('Single-job recovery requires one replicate')
    root = Path(root)
    children = sorted(p for p in root.glob('*') if p.is_dir())
    if not children:
        return {'status': 'pending', 'record': None, 'failed_attempts': [], 'attempt_count': 0}
    accepted, failed = [], []
    for run in children:
        saved_model = json.loads((run/'model.json').read_text())
        if (saved_model.get('revision') != config['revision'] or saved_model.get('model') != config.get('model')):
            raise ValueError('Durable single-job model revision changed')
        if model_info is None:
            model_info = saved_model
        if (digest(json.loads((run / 'config.json').read_text())) != digest(config)
            or json.loads((run / 'dataset.json').read_text())['sha256'] != digest([task])
            or json.loads((run / 'scoring_contract.json').read_text()) != scoring_contract([task], config)
            or digest(saved_model) != digest(model_info)):
            raise ValueError('Durable single-job contract changed')
        rows = read_jsonl(run / 'samples.jsonl') if (run / 'samples.jsonl').exists() else []
        if not rows:
            failed.append({'run': str(run), 'reason': 'no_durable_generation'})
            continue
        if len(rows) != 1:
            raise ValueError('Single-job archive has duplicate generations')
        row = rows[0]
        if (row['sample_id'] != task['sample_id'] or row['prompt'] != task['prompt']
            or row['expected'] != task['expected'] or row['replicate'] != 0
            or row['seed'] != replicate_seed(config['seed'], task, 0)
            or row['config_hash'] != digest(config)
            or row['scoring_version'] != scoring_contract([task], config)['version']
            or digest(row['model_identifier']) != digest(model_info)):
            raise ValueError('Durable generation violates recovery contract')
        if row['termination_reason'] not in VALID_TERMINATIONS:
            failed.append({'run': str(run), 'reason': row['termination_reason'],
                           'generation_error': row.get('generation_error')})
        else:
            accepted.append(row)
    if len(accepted) > 1:
        raise ValueError('Multiple successful durable generations; refuse duplicate accounting')
    return {'status': 'complete' if accepted else 'failed', 'record': accepted[0] if accepted else None,
            'failed_attempts': failed, 'attempt_count': len(children)}


def evaluate_single(backend, task, config, root, *, retry_failed=False):
    if not retry_failed and len([p for p in Path(root).glob('*') if p.is_dir()]) > 1:
        raise ValueError('Multiple durable attempts require explicit retry-aware recovery')
    inventory = inspect_single(task, config, root, backend.model_info)
    # A row is fsynced before summary/completion publication. Reuse it even if
    # the process stopped in that publication window; never append/regenerate.
    if inventory['record'] is not None:
        return inventory['record']
    if inventory['status'] == 'failed' and not retry_failed:
        raise ValueError('Failed durable attempts preserved; explicit retry_failed required for a new attempt')
    # evaluate() allocates a unique timestamp/UUID attempt. Never overwrite or
    # resume a failed attempt, and never automatically retry a new failure.
    run, _ = evaluate(backend, [task], config, root=root)
    return read_jsonl(run / 'samples.jsonl')[0]
