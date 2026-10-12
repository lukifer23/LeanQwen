"""Approve whole independently certified examples at an explicit sequence bound."""

import argparse
import json
from collections import Counter
from pathlib import Path

from qwenlean.datasets.procedural_v2 import FAMILIES
from qwenlean.training.quality import validate_parent_sources, validate_training_pool
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl


def approve(corpus, destination, config, selection='all'):
    corpus, destination = Path(corpus), Path(destination)
    if destination.exists():
        raise ValueError('Refuse existing approved pool')
    rows = read_jsonl(corpus/'train.jsonl')
    bound = config['max_sequence_length']
    eligible = [r for r in rows if len(r['prompt_token_ids'])+len(r['target_token_ids']) <= bound]
    if selection == 'smoke':
        selected = [min([r for r in eligible if r['family'] == f], key=lambda r: abs(len(r['prompt_token_ids'])+len(r['target_token_ids'])-256)) for f in FAMILIES]
    elif selection == 'longest':
        selected = [max(eligible, key=lambda r: len(r['prompt_token_ids'])+len(r['target_token_ids']))]
    elif selection == 'all':
        selected = eligible
    else:
        raise ValueError('Unknown selection')
    if not selected:
        raise ValueError('No complete examples fit')
    manifest = json.loads((corpus/'manifest.json').read_text())
    manifest.update(dataset_hash=digest(selected), approved_solver_certificate_ids=[r['proof_certificate_id'] for r in selected],
        record_count=len(selected), distinct_train_problems=len(selected),
        source_distribution={'solver_verified_reasoning': len(selected)}, source_sampling_proportions={'solver_verified_reasoning': 1.0},
        family_distribution=dict(Counter(r['family'] for r in selected)),
        difficulty_distribution=dict(Counter(r['difficulty'] for r in selected)),
        pool_selection={'method': selection, 'max_sequence_length': bound,
                        'excluded_whole_for_length': len(rows)-len(eligible), 'selected_task_ids': [r['task_id'] for r in selected]},
        parent_corpus_manifest_hash=digest(json.loads((corpus/'manifest.json').read_text())))
    # Length distributions from the full corpus must not masquerade as pool statistics.
    for key in ('reasoning_lengths', 'response_lengths', 'complete_sequence_lengths', 'length_by_difficulty', 'sequence_fit_counts'):
        manifest.pop(key, None)
    manifest['actual_sequence_lengths'] = [len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in selected]
    validate_training_pool(selected, manifest, config)
    validate_parent_sources(selected, manifest)
    destination.mkdir(parents=True)
    write_jsonl(destination/'train.jsonl', selected)
    write_json(destination/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--selection', choices=['all', 'smoke', 'longest'], default='all')
    a = p.parse_args()
    result = approve(a.corpus, a.output, yaml.safe_load(Path(a.config).read_text()), a.selection)
    print(json.dumps({'approved': result['record_count'], 'lengths': result['actual_sequence_lengths']}))
