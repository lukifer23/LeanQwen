"""Recover a one-trajectory diagnostic without duplicating a durable generation."""

import json
from pathlib import Path

from qwenlean.datasets.prompts import replicate_seed
from qwenlean.evaluation.runner import VALID_TERMINATIONS, evaluate, scoring_contract
from qwenlean.utils.io import digest, read_jsonl


def evaluate_single(backend, task, config, root):
    if config.get('replicates', 1) != 1:
        raise ValueError('Single-job recovery requires one replicate')
    root = Path(root)
    children = sorted(p for p in root.glob('*') if p.is_dir())
    if len(children) > 1:
        raise ValueError('Multiple durable attempts exist; refuse ambiguous recovery')
    if not children:
        run, _ = evaluate(backend, [task], config, root=root)
        return read_jsonl(run / 'samples.jsonl')[0]
    run = children[0]
    if (digest(json.loads((run / 'config.json').read_text())) != digest(config)
            or json.loads((run / 'dataset.json').read_text())['sha256'] != digest([task])
            or json.loads((run / 'scoring_contract.json').read_text()) != scoring_contract([task], config)
            or digest(json.loads((run / 'model.json').read_text())) != digest(backend.model_info)):
        raise ValueError('Durable single-job contract changed')
    rows = read_jsonl(run / 'samples.jsonl') if (run / 'samples.jsonl').exists() else []
    if not rows:
        run, _ = evaluate(backend, [task], config, resume=run)
        return read_jsonl(run / 'samples.jsonl')[0]
    if len(rows) != 1:
        raise ValueError('Single-job archive has duplicate generations')
    row = rows[0]
    if (row['sample_id'] != task['sample_id'] or row['prompt'] != task['prompt']
            or row['expected'] != task['expected'] or row['replicate'] != 0
            or row['seed'] != replicate_seed(config['seed'], task, 0)
            or row['config_hash'] != digest(config)
            or row['scoring_version'] != scoring_contract([task], config)['version']
            or digest(row['model_identifier']) != digest(backend.model_info)
            or row['termination_reason'] not in VALID_TERMINATIONS):
        raise ValueError('Durable generation failed or violates recovery contract')
    # A row is fsynced before summary/completion publication. Reuse it even if
    # the process stopped in that publication window; never append/regenerate.
    return row
