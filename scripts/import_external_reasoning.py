"""Bounded, source-preserving human reasoning pilot; independent review stays pending."""

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from qwenlean.datasets.external_reasoning import (
    GSM8K_LICENSE_SHA256,
    GSM8K_TEST_SHA256,
    GSM8K_TRAIN_SHA256,
    candidate_record,
)
from qwenlean.training.external_contract import (
    PERMISSION_FIELDS,
    POLICY,
    REVIEW_DIMENSIONS,
    checked_file,
    encode_external_target,
)
from qwenlean.training.solver_contract import REVISION, pinned_tokenizer, tokenizer_contract
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock


def prompt_structure(text):
    return digest(re.sub(r'[-+]?\d+(?:[.,]\d+)*', '#', ' '.join(text.casefold().split())))


def stats(values):
    return {'count': len(values), 'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p90': float(np.percentile(values, 90)), 'p95': float(np.percentile(values, 95)),
            'min': min(values), 'max': max(values),
            'cumulative_under': {str(b): sum(v < b for v in values) for b in (512, 1024, 2048, 4096)},
            'values': values}


def review_html(rows):
    cards, metadata = [], []
    for row in rows:
        task = row['task_id']
        metadata.append({'task_id': task, 'source_record_hash': row['source_record_hash']})
        checks = ''.join(f'<label><input type="checkbox" data-dimension="{key}">{key.replace("_", " ")}</label><br>' for key in REVIEW_DIMENSIONS)
        cards.append(f'<article id="{task}"><h2>{html.escape(task)}</h2><p>{html.escape(row["body"])}</p><pre>{html.escape(row["source_record"]["answer"])}</pre><p>{row["structural_complexity"]} annotated operations; {row["reasoning_tokens"]} reasoning tokens. Arithmetic checked; prose semantics need your review.</p>{checks}<label>Review note <input class="note"></label></article>')
    data = json.dumps(metadata).replace('<', '\\u003c')
    return ('<!doctype html><meta charset="utf-8"><title>External reasoning review</title>'
            '<style>body{max-width:950px;margin:30px auto;font:16px system-ui}article{padding:24px;border-bottom:1px solid #aaa}pre{white-space:pre-wrap;background:#f5f5f5;padding:15px}input.note{width:80%}</style>'
            '<h1>Independent human solution review</h1><p>These are intact original human TRAIN solutions. Inspect each problem and all its steps. No record is approved automatically. Your notes never become training targets.</p>'
            '<label>Human reviewer identity <input id="reviewer"></label> <button onclick="download()">Download reviewed records</button>'
            + ''.join(cards)+f'<script>const records={data};'
            + '''function download(){const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('Enter your human reviewer identity');return;}const output=[];for(const r of records){const card=document.getElementById(r.task_id),note=card.querySelector('.note').value.trim();const checks=[...card.querySelectorAll('[data-dimension]')];if(!checks.some(c=>c.checked)&&!note)continue;const dimensions=Object.fromEntries(checks.map(c=>[c.dataset.dimension,c.checked]));output.push({...r,...dimensions,approved_for_sft:checks.every(c=>c.checked)&&!!note,reviewer_type:'human',reviewer_id:reviewer,review_method:'individual_source_solution_inspection',review_note:note});}const blob=new Blob([output.map(r=>JSON.stringify(r)).join('\\n')+'\\n'],{type:'application/jsonl'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='external-human-reviews.jsonl';a.click();URL.revokeObjectURL(a.href);}</script>''')


def build(archive, output, count=768, dev_count=64):
    archive, output = Path(archive), Path(output)
    if not 500 <= count <= 1000 or not 16 <= dev_count <= 128:
        raise ValueError('Pilot must stay bounded: 500–1000 TRAIN, 16–128 DEV')
    if output.exists():
        raise ValueError('Refuse existing external corpus')
    with model_process_lock():
        train_path = checked_file(archive/'gsm8k-train.jsonl', GSM8K_TRAIN_SHA256)
        test_path = checked_file(archive/'gsm8k-test.jsonl', GSM8K_TEST_SHA256)
        license_path = checked_file(archive/'gsm8k-LICENSE', GSM8K_LICENSE_SHA256)
        source = read_jsonl(train_path)
        # TEST questions only: no TEST response tokenization, reasoning checks or selection.
        test_questions = [r['question'] for r in read_jsonl(test_path)]
        forbidden = {prompt_structure(q) for q in test_questions}
        for path in Path('data/splits').rglob('*.jsonl'):
            forbidden.update(prompt_structure(r.get('body', r['prompt'])) for r in read_jsonl(path))
        counts, rows, decisions, seen = Counter(), [], [], set()
        tokenizer = pinned_tokenizer()
        contract = tokenizer_contract(tokenizer)
        for index, raw in enumerate(source):
            try:
                row = candidate_record(raw, index)
                check = row['verification']
                if check['invalid_annotations']:
                    reason = 'invalid_annotated_arithmetic'
                elif check['unverified_annotations']:
                    reason = 'unrecognized_annotated_arithmetic'
                elif not check['annotation_consistent']:
                    reason = 'final_not_linked_to_last_verified_annotation'
                elif row['efficiency_screen']['flags']:
                    reason = 'efficiency_flag_requires_separate_review'
                elif row['numeric_free_prompt_hash'] in forbidden:
                    reason = 'benchmark_or_repository_prompt_overlap'
                elif row['semantic_structure_fingerprint'] in seen:
                    reason = 'normalized_task_structure_duplicate'
                else:
                    reason = 'screened_candidate_not_quality_approved'
                    encoded = encode_external_target(row, tokenizer, contract)
                    rows.append(encoded)
                    seen.add(row['semantic_structure_fingerprint'])
                counts[reason] += 1
                decisions.append({'source_index': index, 'source_record_hash': digest(raw), 'decision': reason})
            except (ValueError, SyntaxError, ZeroDivisionError) as exc:
                counts['malformed_or_unsupported_source'] += 1
                decisions.append({'source_index': index, 'source_record_hash': digest(raw),
                                  'decision': 'malformed_or_unsupported_source', 'error': str(exc)})
        groups = defaultdict(list)
        for row in rows:
            # Separate namespaces before target choice. No DEV response enters TRAIN.
            namespace = 'dev' if int(row['semantic_structure_fingerprint'][:8], 16) % 10 == 0 else 'train'
            level = 'few_steps' if row['structural_complexity'] <= 2 else 'several_steps' if row['structural_complexity'] <= 4 else 'longer_dependencies'
            groups[(namespace, level)].append(row)
        chosen = {}
        for split, quota in [('train', count), ('dev', dev_count)]:
            buckets = [sorted(rs, key=lambda r: digest(['external-pilot-seed-20261012', r['task_id']]))
                       for (s, _), rs in sorted(groups.items()) if s == split]
            selected = []
            while len(selected) < quota and any(buckets):
                for bucket in buckets:
                    if bucket and len(selected) < quota:
                        selected.append(bucket.pop(0))
            if len(selected) != quota:
                raise ValueError('Insufficient distinct independently sourced candidates; do not fill quota by duplication')
            chosen[split] = selected
        output.mkdir(parents=True)
        train, dev = chosen['train'], chosen['dev']
        write_jsonl(output/'candidates.jsonl', train)
        # DEV uses questions/expected source labels only; no worked response tokens.
        dev_tasks = []
        for row in dev:
            t = {k: row[k] for k in ('task_id', 'sample_id', 'body', 'family', 'expected', 'structural_complexity', 'semantic_structure_fingerprint')}
            t.update(split='dev', prompt=row['body'], source_split='train', dataset='external-gsm8k-dev-v1',
                     provenance={**row['provenance'], 'training_permitted': False})
            dev_tasks.append(t)
        write_jsonl(output/'dev-questions.jsonl', dev_tasks)
        write_jsonl(output/'screening-decisions.jsonl', decisions)
        templates = [{'task_id': r['task_id'], 'source_record_hash': r['source_record_hash'],
                      'approved_for_sft': False, 'reviewer_type': 'human', 'reviewer_id': '',
                      'review_method': 'individual_source_solution_inspection', 'review_note': '',
                      **{k: False for k in REVIEW_DIMENSIONS}} for r in train]
        write_jsonl(output/'review-template.jsonl', templates)
        (output/'review-bundle.html').write_text(review_html(train))
        (output/'LICENSE-MIT').write_bytes(license_path.read_bytes())
        certificate = {**PERMISSION_FIELDS, 'dataset_path': str(train_path), 'license_path': str(license_path)}
        manifest = {'status': 'independent_quality_review_pending', 'training_source_policy': POLICY,
            'dataset_hash': digest(train), 'model_revision': REVISION, 'tokenizer_contract': contract,
            'source_permissions': [certificate], 'candidate_count': len(train), 'quality_approved_count': 0,
            'distinct_problem_count': len({r['task_identity_hash'] for r in train}),
            'distinct_structure_count': len({r['semantic_structure_fingerprint'] for r in train}),
            'source_distribution': {'licensed_external_reasoning': len(train)},
            'dataset_source_distribution': {'gsm8k-original-human-train': len(train)},
            'source_sampling_proportions': {'licensed_external_reasoning': 1.0},
            'teacher_distribution': {'human_written': len(train), 'Qwen': 0, 'LFM': 0, 'solver_authored': 0},
            'source_screening_counts': dict(counts), 'reasoning_lengths': stats([r['reasoning_tokens'] for r in train]),
            'response_lengths': stats([len(r['target_token_ids']) for r in train]),
            'sequence_lengths': stats([len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in train]),
            'sequence_fit': {str(b): sum(len(r['prompt_token_ids'])+len(r['target_token_ids']) <= b for r in train) for b in (1024, 2048, 4096)},
            'operation_count_distribution': dict(Counter(r['structural_complexity'] for r in train)),
            'contamination_checks': {'passed': True, 'official_gsm8k_test_questions_hashed_only': len(test_questions),
                'excluded_all_detected_number_free_test_or_repository_prompt_matches': True,
                'train_dev_identity_overlap': 0, 'train_dev_normalized_structure_overlap': 0,
                'scope': 'Official GSM8K TEST and repository stimuli; numeric-free wording and operation fingerprints. No guarantee against semantic paraphrases or every external benchmark.'},
            'dev_count': len(dev_tasks), 'dev_hash': digest(dev_tasks), 'test_used': False, 'dev_used': False,
            'test_metadata_only': True, 'paid_calls': 0, 'student_generations_used': 0,
            'screening_scope': 'Exact annotated arithmetic, final source agreement, completion, deterministic repetition cues. Prose semantics/relevance/productive effort remain unapproved pending individual human or authorized frontier review.',
            'recipe': {'count': count, 'dev_count': dev_count, 'seed': 20261012,
                       'selection': 'Round robin few/several/longer operation dependencies after immutable hash split',
                       'no_reasoning_rewriting': True, 'no_truncation': True}}
        write_json(output/'manifest.json', manifest)
        return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=768)
    p.add_argument('--dev-count', type=int, default=64)
    a = p.parse_args()
    report = build(a.archive, a.output, a.count, a.dev_count)
    print(json.dumps({k: report[k] for k in ('candidate_count', 'quality_approved_count', 'source_screening_counts', 'sequence_fit')}))
