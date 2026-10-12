"""Reproduce bounded diagnostic measurements from saved raw source pages."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from qwenlean.training.solver_contract import pinned_tokenizer
from qwenlean.utils.io import digest, write_json
from qwenlean.utils.process_lock import model_process_lock
from scripts.audit_external_sources import SOURCES, statistics


def summarize(archive, output):
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise ValueError('Refuse existing derived audit')
    raw = json.loads((archive/'report.json').read_text())
    result = {**raw, 'analysis_version': 'external-source-comparison-v2', 'raw_report_hash': digest(raw),
              'proxy_method': {'repeat_unit': 'Normalized newline/sentence units at least 40 characters, exact duplicate within a trace',
                               'reconsideration_regex': r'\b(?:wait|hold on|reconsider|alternatively|let me|double.check)\b',
                               'limitation': 'These proxies are not proof of waste or independent semantic review.'}}
    with model_process_lock():
        tokenizer = pinned_tokenizer()
        for source, spec in SOURCES.items():
            measured = result['results'][source]
            measured['raw_worked_response_tokens'] = measured.pop('reasoning_tokens')
            pages = sorted((archive/source.replace('/', '--')).glob('rows-*.json'))
            lengths, scope, repeats, cues, flags, breakdown = [], Counter(), 0, 0, Counter(), Counter()
            for page in pages:
                for entry in json.loads(page.read_text())['rows']:
                    row = entry['row']
                    breakdown[row.get('source', row.get('problem_source', 'unspecified'))] += 1
                    texts = row['generations'] if spec['field'] == 'generations' else [row[spec['field']]] if spec['field'] != 'r1_solution' else [row[f'r1_solution_{i}'] for i in (1, 2, 3)]
                    for text in texts:
                        match = re.search(r'<think>\s*(.*?)\s*</think>', text, re.S)
                        reasoning = match.group(1) if match else text
                        scope['explicit_thinking_segment' if match else 'whole_worked_solution_no_uniform_thinking_delimiter'] += 1
                        lengths.append(len(tokenizer.encode(reasoning, add_special_tokens=False)))
                        units = [' '.join(u.casefold().split()) for u in re.split(r'\n+|(?<=[.!?])\s+', reasoning)]
                        counts = Counter(u for u in units if len(u) >= 40)
                        repeats += any(n > 1 for n in counts.values())
                        cues += bool(re.search(result['proxy_method']['reconsideration_regex'], reasoning, re.I))
                    for key, label in [('is_reasoning_complete', 'complete_publisher_flag_true'), ('correctness_math_verify', 'final_math_verify_publisher_flag_true')]:
                        flags[label] += sum(v is True for v in row.get(key, []) or [])
            measured['reasoning_or_worked_solution_tokens'] = statistics(lengths)
            measured['measurement_scope_counts'] = dict(scope)
            measured['source_breakdown_sampled'] = dict(breakdown)
            measured['publisher_quality_flags_sampled'] = dict(flags)
            measured['deterministic_verbosity_proxies'] = {'trace_count': len(lengths),
                'traces_with_exact_repeated_sentence_or_line': repeats, 'traces_with_reconsideration_cue': cues,
                'qualification': result['proxy_method']['limitation']}
    result['permission_artifacts'] = json.loads(Path('data/raw/external-permissions-20261012/manifest.json').read_text())
    for file, url in [('data/raw/external-permissions-20261012/gsm8k-human-qa-paper.html', 'https://arxiv.org/html/2110.14168v2')]:
        data = Path(file).read_bytes()
        result['permission_artifacts'].append({'file': file, 'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summarize(args.archive, args.output)
