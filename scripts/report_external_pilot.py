"""Publish compact corpus evidence without source targets or invented training results."""

import argparse
import json
from pathlib import Path

import yaml

from qwenlean.training.quality import load_approved_training_inputs
from qwenlean.utils.io import digest, read_jsonl, write_json
from scripts.import_external_reasoning import stats


def report(candidates, approved, smoke, config, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Refuse existing report')
    candidate_manifest = json.loads((Path(candidates)/'manifest.json').read_text())
    cfg = yaml.safe_load(Path(config).read_text())
    rows, manifest = load_approved_training_inputs(cfg, Path(approved)/'train.jsonl', Path(approved)/'manifest.json')
    smoke_rows, smoke_manifest = load_approved_training_inputs(cfg, Path(smoke)/'train.jsonl', Path(smoke)/'manifest.json')
    dev = read_jsonl(Path(candidates)/'dev-questions.jsonl')
    if digest(dev) != candidate_manifest['dev_hash']:
        raise ValueError('Reserved DEV differs from frozen import')
    data = {**manifest, 'report_version': 'external-human-pilot-v1', 'starting_sha': '0a62326a9bb5ddc5cbb3af89689ac7dd5839a672',
        'candidate_archive_manifest_hash': digest(candidate_manifest), 'candidate_manifest_status_retained': candidate_manifest['status'],
        'approved_manifest_hash': digest(manifest), 'corpus_path': str(approved), 'dev_path': str(Path(candidates)/'dev-questions.jsonl'),
        'reasoning_lengths': stats([r['reasoning_tokens'] for r in rows]),
        'response_lengths': stats([len(r['target_token_ids']) for r in rows]),
        'sequence_lengths': stats([len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in rows]),
        'structural_bin_distribution': {label: sum(lo <= r['structural_complexity'] <= hi for r in rows)
            for label, lo, hi in [('few_steps', 1, 2), ('several_steps', 3, 4), ('longer_dependencies', 5, 8)]},
        'smoke': {'record_count': len(smoke_rows), 'dataset_hash': smoke_manifest['dataset_hash'], 'manifest_hash': digest(smoke_manifest),
                  'path': str(smoke), 'config_path': str(config), 'config_hash': digest(cfg),
                  'complete_sequence_lengths': [len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in smoke_rows]},
        'training_status': {'optimizer_steps': 0, 'adapter_exists': False, 'improvement_pilot_completed': False,
                            'note': 'Corpus milestone only. Actual optimizer/evaluation results require separate saved reports.'},
        'verification_scope': 'Exact calculator annotations and source final-label agreement, not independent semantic solution of every word problem.',
        'unverified_source_prose_claims_formally_certified': 0}
    write_json(output, data)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', required=True)
    p.add_argument('--approved', required=True)
    p.add_argument('--smoke', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    report(a.candidates, a.approved, a.smoke, a.config, a.output)
