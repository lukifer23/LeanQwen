"""Freeze source DEV and existing procedural retention tasks without model inference."""

import argparse
import json
from collections import Counter
from pathlib import Path

from qwenlean.datasets.prompts import render_task
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl


def freeze(corpus, retention, output, report_path):
    corpus, retention, output, report_path = map(Path, (corpus, retention, output, report_path))
    if output.exists() or report_path.exists():
        raise ValueError('Refuse existing frozen cohort/report')
    manifest = json.loads((corpus/'manifest.json').read_text())
    dev = read_jsonl(corpus/'dev-questions.jsonl')
    train = read_jsonl(corpus/'candidates.jsonl')
    if digest(dev) != manifest['dev_hash'] or digest(train) != manifest['dataset_hash']:
        raise ValueError('Imported cohort differs')
    previous = json.loads((retention/'protocol.json').read_text())
    old_tasks = read_jsonl(retention/'tasks.jsonl')
    if digest(old_tasks) != previous['task_hash']:
        raise ValueError('Historical retention cohort differs')
    selected = []
    for lo, hi in [(1, 2), (3, 4), (5, 8)]:
        bucket = sorted([t for t in dev if lo <= t['structural_complexity'] <= hi], key=lambda t: digest(['external-dev-v1', t['task_id']]))
        if len(bucket) < 8:
            raise ValueError('Insufficient source DEV structural diversity')
        selected.extend(bucket[:8])
    if ({t['task_id'] for t in selected} & {t['task_id'] for t in train}
        or {t['semantic_structure_fingerprint'] for t in selected} & {t['semantic_structure_fingerprint'] for t in train}):
        raise ValueError('Source TRAIN/DEV contamination')
    tasks = [{**render_task({**t, 'facts': {'source_task_id': t['task_id'],
        'annotation_count': t['structural_complexity']}}, 'P2'),
        'evaluation_cohort': 'gsm8k_within_source_dev'} for t in selected]
    tasks += [{**t, 'evaluation_cohort': 'procedural_retention_domain_shift'} for t in old_tasks]
    configs = {mode: {**cfg, 'name': 'external-human-v1-matched-'+mode} for mode, cfg in previous['configs'].items()}
    protocol = {'status': 'frozen_before_any_optimizer', 'task_hash': digest(tasks), 'task_count': len(tasks),
        'configs': configs, 'configs_hash': digest(configs), 'test_used': False,
        'source_dev_count': len(selected), 'retention_count': len(old_tasks),
        'source_import_manifest_hash': digest(manifest), 'retention_protocol_hash': digest(previous),
        'source_difficulty_operation_counts': dict(Counter(t['structural_complexity'] for t in selected)),
        'selection_rule': 'Eight per annotation-count bin, immutable question hash ordering. Existing 32 procedural tasks unchanged. No model outcomes used.',
        'generalization_questions': {'gsm8k_within_source_dev': 'Human word-problem source generalization across distinct normalized structures.',
            'procedural_retention_domain_shift': 'Retain prior eight-family procedural skills after human-word-problem SFT; historical solver ID/OOD labels are not current training labels.'},
        'paired_statistics': 'Cluster by task; one matched seed per task/mode, sampler variance remains unestimated.',
        'primary_budget': 2048, 'budget_frontier': 'Additional matched budgets only after primary results; not measured yet.'}
    output.mkdir(parents=True)
    write_jsonl(output/'tasks.jsonl', tasks)
    write_json(output/'protocol.json', protocol)
    write_json(report_path, {**protocol, 'cohort_path': str(output), 'task_ids': [t['task_id'] for t in tasks]})
    return protocol


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--retention', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    freeze(a.corpus, a.retention, a.output, a.report)
