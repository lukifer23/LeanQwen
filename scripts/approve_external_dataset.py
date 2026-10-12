"""Approve independently reviewed intact external targets; no rewriting or teacher calls."""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from qwenlean.training.external_contract import (
    GSM8K_QA_SHA256,
    GSM8K_QA_URL,
    checked_file,
    validate_review,
)
from qwenlean.training.quality import validate_parent_sources, validate_training_pool
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock


def approve(corpus, reviews_path, output, config, limit=None, published_human_qa=None):
    corpus, output = Path(corpus), Path(output)
    if output.exists():
        raise ValueError('Refuse existing approved corpus')
    manifest = json.loads((corpus/'manifest.json').read_text())
    candidates = read_jsonl(corpus/'candidates.jsonl')
    if digest(candidates) != manifest['dataset_hash']:
        raise ValueError('Imported candidate archive differs')
    if published_human_qa:
        if reviews_path or len(candidates) > 1000:
            raise ValueError('Published human QA is a separate bounded approval basis')
        checked_file(published_human_qa, GSM8K_QA_SHA256)
        reviews = [{'task_id': r['task_id'], 'source_record_hash': r['source_record_hash'],
            'approved_for_sft': True, 'reviewer_type': 'publisher_human_qa',
            'review_method': 'gsm8k-published-human-qa-and-exact-screening-v1',
            'publication_url': GSM8K_QA_URL, 'publication_path': str(published_human_qa),
            'publication_sha256': GSM8K_QA_SHA256,
            'approval_scope': 'bounded_original_human_reference_pilot',
            'individual_project_review_performed': False, 'prose_semantics_formally_verified': False,
            'review_note': 'Original human reference solutions: published worker-answer-agreement QA plus exact arithmetic, final-label linkage, completion and repetition screening. No fresh individual inspection; unannotated prose is not formally verified.'}
            for r in candidates]
        for row, review in zip(candidates, reviews, strict=True):
            validate_review(row, review, config)
        reviews_path = output.parent/(output.name+'-publisher-qa.jsonl')
        if reviews_path.exists():
            raise ValueError('Refuse existing QA decisions')
        reviews_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(reviews_path, reviews)
        manifest['quality_approval_basis'] = 'published_human_qa_with_exact_screening'
        manifest['prose_semantics_formally_verified'] = False
        manifest['individual_project_review_performed'] = False
        manifest['screening_scope'] = reviews[0]['review_note']
    else:
        reviews = read_jsonl(reviews_path)
        manifest['quality_approval_basis'] = 'individual_independent_review'
    lookup = {r['task_id']: r for r in reviews}
    if len(lookup) != len(reviews):
        raise ValueError('Duplicate independent review identities')
    selected, excluded = [], []
    for row in candidates:
        review = lookup.get(row['task_id'])
        if not review or review.get('approved_for_sft') is not True:
            continue
        validate_review(row, review, config)
        if len(row['prompt_token_ids'])+len(row['target_token_ids']) > config['max_sequence_length']:
            excluded.append(row['task_id'])
            continue
        selected.append({**row, 'correct': True, 'termination_reason': 'eos',
                         'quality_review_id': digest(review)})
        if limit and len(selected) == limit:
            break
    if not selected:
        raise ValueError('No approved complete external targets; do not manufacture passing reviews')
    manifest.update(status='quality_approved', dataset_hash=digest(selected), record_count=len(selected),
        distinct_problem_count=len({r['task_identity_hash'] for r in selected}),
        distinct_structure_count=len({r['semantic_structure_fingerprint'] for r in selected}),
        quality_approved_count=len(selected), source_distribution={'licensed_external_reasoning': len(selected)},
        dataset_source_distribution={'gsm8k-original-human-train': len(selected)},
        source_sampling_proportions={'licensed_external_reasoning': 1.0},
        approved_source_record_hashes=[r['source_record_hash'] for r in selected],
        reviews_path=str(reviews_path), reviews_hash=digest(reviews),
        candidate_archive_hash=digest(candidates), length_excluded_whole=excluded,
        actual_sequence_lengths=[len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in selected])
    # Full-candidate statistics cannot pretend to describe an approved subset.
    for key in ('reasoning_lengths', 'response_lengths', 'sequence_lengths', 'sequence_fit', 'operation_count_distribution', 'teacher_distribution'):
        manifest.pop(key, None)
    manifest['teacher_distribution'] = {'human_written': len(selected)}
    manifest['operation_count_distribution'] = dict(Counter(r['structural_complexity'] for r in selected))
    validate_training_pool(selected, manifest, config)
    validate_parent_sources(selected, manifest, config)
    output.mkdir(parents=True)
    write_jsonl(output/'train.jsonl', selected)
    write_json(output/'manifest.json', manifest)
    (output/'LICENSE-MIT').write_bytes((corpus/'LICENSE-MIT').read_bytes())
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    basis = p.add_mutually_exclusive_group(required=True)
    basis.add_argument('--reviews')
    basis.add_argument('--published-human-qa', help='Pinned GSM8K human QA publication; explicit bounded reference-data approval, not individual prose certification')
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('Positive record limit required')
    with model_process_lock():
        result = approve(a.corpus, a.reviews, a.output, yaml.safe_load(Path(a.config).read_text()), a.limit, a.published_human_qa)
    print(json.dumps({'approved': result['record_count'], 'dataset_hash': result['dataset_hash']}))
