"""Bounded external-source research. No model inference or paid API generation."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from qwenlean.training.solver_contract import pinned_tokenizer
from qwenlean.utils.io import write_json
from qwenlean.utils.process_lock import model_process_lock

SOURCES = {
    'open-r1/OpenR1-Math-220k': {'count': 93733, 'field': 'generations'},
    'nvidia/OpenMathInstruct-2': {'count': 13972791, 'field': 'generated_solution'},
    'zwhe99/DeepMath-103K': {'count': 103022, 'field': 'r1_solution'},
}


def fetch(url, destination, limit):
    if destination.exists():
        raise ValueError('Refuse existing source artifact')
    request = Request(url, headers={'User-Agent': 'LeanQwen-bounded-source-audit/1.0'})
    with urlopen(request, timeout=45) as response:
        data = response.read(limit+1)
        headers = dict(response.headers)
    if len(data) > limit:
        raise ValueError('Declared source download bound exceeded')
    destination.write_bytes(data)
    return {'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'headers': {k: v for k, v in headers.items() if k.lower() in {'etag', 'x-repo-commit', 'last-modified'}}}


def statistics(values):
    return {'count': len(values), 'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p90': float(np.percentile(values, 90)), 'p95': float(np.percentile(values, 95)),
            'min': min(values), 'max': max(values),
            'under': {str(b): sum(v < b for v in values) for b in (512, 1024, 2048, 4096)},
            'values': values}


def run(output, gsm_archive):
    output = Path(output)
    if output.exists():
        raise ValueError('Refuse existing audit')
    with model_process_lock():
        output.mkdir(parents=True)
        # Freeze bounded sampling policy before contacting large dataset servers.
        plan = {'rows_per_dataset': 64, 'pages_per_dataset': 4, 'rows_per_page': 16,
                'offset_policy': '0, floor(N/3), floor(2N/3), N-16',
                'max_page_bytes': 12*1024**2, 'max_total_bytes': 160*1024**2,
                'large_parquet_downloads': False, 'purpose': 'research only; no sample is training-approved'}
        write_json(output/'sampling-plan.json', plan)
        tokenizer = pinned_tokenizer()
        results, artifacts, total_bytes = {}, [], 0
        gsm = [json.loads(line) for line in (Path(gsm_archive)/'gsm8k-train.jsonl').read_text().splitlines()]
        lengths = [len(tokenizer.encode(r['answer'].rsplit('####', 1)[0], add_special_tokens=False)) for r in gsm]
        results['gsm8k-original-human-train'] = {'problem_count_measured': len(gsm),
            'reasoning_tokens': statistics(lengths), 'sample_scope': 'Entire small 4.17 MB original TRAIN file; no TEST solutions tokenized',
            'family_coverage': {'grade_school_math_word_problems': len(gsm)}}
        for dataset, spec in SOURCES.items():
            folder = output/dataset.replace('/', '--')
            folder.mkdir()
            meta_path = folder/'metadata-before.json'
            artifact = fetch('https://huggingface.co/api/datasets/'+dataset, meta_path, 2*1024**2)
            artifacts.append(artifact)
            meta = json.loads(meta_path.read_text())
            pin = meta['sha']
            artifacts.append(fetch(f'https://huggingface.co/datasets/{dataset}/raw/{pin}/README.md', folder/'README.md', 1024**2))
            problem_rows, all_lengths, families, failures = [], [], Counter(), []
            for offset in [0, spec['count']//3, 2*spec['count']//3, spec['count']-16]:
                url = 'https://datasets-server.huggingface.co/rows?'+urlencode({'dataset': dataset, 'config': 'default', 'split': 'train', 'offset': offset, 'length': 16})
                path = folder/f'rows-{offset:08d}.json'
                try:
                    artifact = fetch(url, path, plan['max_page_bytes'])
                    total_bytes += artifact['bytes']
                    if total_bytes > plan['max_total_bytes']:
                        raise ValueError('Total bounded sample budget exceeded')
                    artifacts.append(artifact)
                    rows = json.loads(path.read_text())['rows']
                    for entry in rows:
                        row = entry['row']
                        texts = row[spec['field']] if spec['field'] == 'generations' else [row[spec['field']]] if spec['field'] != 'r1_solution' else [row[f'r1_solution_{i}'] for i in (1, 2, 3)]
                        all_lengths.extend(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)
                        problem_rows.append(row)
                        families[row.get('problem_type', row.get('problem_source', row.get('topic', 'unknown')))] += 1
                except Exception as exc:
                    failures.append({'url': url, 'error_type': type(exc).__name__, 'error': str(exc)})
            after = folder/'metadata-after.json'
            artifacts.append(fetch('https://huggingface.co/api/datasets/'+dataset, after, 2*1024**2))
            after_pin = json.loads(after.read_text())['sha']
            results[dataset] = {'dataset_revision': pin, 'head_unchanged_during_audit': pin == after_pin,
                'sample_revision_binding': 'Viewer does not attest an immutable commit; before/after head checked. These sampled rows are diagnostic only, not import-approved.',
                'problems_sampled': len(problem_rows), 'problem_count_publisher': spec['count'],
                'trace_count': len(all_lengths), 'reasoning_tokens': statistics(all_lengths) if all_lengths else None,
                'family_coverage_sampled': dict(families), 'failures': failures}
            write_json(output/'progress.json', results)
        report = {'sampling_plan': plan, 'results': results, 'artifacts': artifacts,
                  'large_sample_bytes': total_bytes, 'paid_calls': 0, 'models_loaded': 0,
                  'tokenizer_model': 'Qwen/Qwen3.5-0.8B',
                  'tokenizer_revision': '2fc06364715b967f1860aea9cf38778875588b17',
                  'note': 'All source text retained intact; no language-model rewrite, generation or subjective judge.'}
        write_json(output/'report.json', report)
        return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--gsm-archive', required=True)
    a = p.parse_args()
    run(a.output, a.gsm_archive)
