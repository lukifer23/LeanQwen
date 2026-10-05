"""Build exact natural SFT targets from individually inspected TRAIN candidates.

Reviewer prose stays in a separate non-training file; it never becomes a response.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from qwenlean.datasets.procedural_v2 import check_structural_contamination
from qwenlean.datasets.provenance import validate_provenance
from qwenlean.evaluation.summary import distribution
from qwenlean.generation.selection import classify
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl


def build(review_path, output, manifest_path, max_sequence_length=1024, max_examples=4):
    archive = 'reports/train_candidate_records.jsonl'
    parents = read_jsonl(archive)
    lookup = {r['generation_id']:r for r in parents}
    reviews = read_jsonl(review_path)
    if len({r['source_generation_id'] for r in reviews}) != len(reviews):
        raise ValueError('Duplicate reviewed source IDs')
    pools = {s:read_jsonl(f'data/splits/procedural-v2/{s}.jsonl') for s in ('train','dev','test')}
    contamination = check_structural_contamination(pools)
    train_ids = {r['task_id'] for r in pools['train']}
    targets, seen_tasks = [], set()
    for review in reviews:
        if review.get('approved_for_sft') is not True:
            continue
        if review.get('intermediate_reasoning_review') != 'valid' or not review.get('review_note'):
            raise ValueError('Explicit individual reasoning review required')
        parent = lookup.get(review['source_generation_id'])
        if parent is None:
            raise ValueError('Reviewed source not in measured TRAIN archive')
        validate_provenance(parent, for_training=True)
        if parent['task_id'] not in train_ids or not classify(parent)['eligible_natural_target']:
            raise ValueError('Reviewed source is not an eligible v2 TRAIN target')
        if parent['task_id'] in seen_tasks:
            raise ValueError('Multiple targets for a single task')
        if len(parent['prompt_token_ids'])+len(parent['token_ids']) > max_sequence_length:
            continue  # whole example excluded; never truncate any response/prompt
        seen_tasks.add(parent['task_id'])
        targets.append({
            'task_id':parent['task_id'], 'sample_id':parent['sample_id'], 'dataset':parent['dataset'],
            'split':'train', 'family':parent['family'], 'difficulty':parent['difficulty'],
            'prompt':parent['prompt'], 'prompt_policy':parent['prompt_policy'],
            'expected':parent['expected'], 'response':parent['raw_output'],
            'reasoning':parent['reasoning'], 'final':parent['final'], 'correct':parent['correct'],
            'termination_reason':parent['termination_reason'], 'model_revision':parent['model_identifier']['revision'],
            'parent_generation_id':parent['generation_id'],
            'prompt_token_ids':parent['prompt_token_ids'], 'target_token_ids':parent['token_ids'],
            'reasoning_tokens':parent['reasoning_tokens'],
            'provenance':{**parent['provenance'], 'source_id':parent['generation_id']+':natural-target',
                          'parent_ids':[parent['generation_id']], 'training_permitted':True}})
        if len(targets) == max_examples:
            break
    if not targets:
        raise ValueError('No individually approved complete examples fit; do not manufacture targets')
    if any(Path(p).exists() for p in (output, manifest_path)):
        raise ValueError('Refuse to overwrite an approved dataset/manifest')
    revisions = {r['model_revision'] for r in targets}
    if len(revisions) != 1:
        raise ValueError('Mixed parent revisions')
    write_jsonl(output, targets)
    manifest={'status':'quality_approved','dataset_hash':digest(targets),
              'source_records_path':archive,'source_records_hash':digest(parents),
              'approved_parent_ids':[r['parent_generation_id'] for r in targets],
              'review_file':str(review_path),'review_hash':digest(reviews),
              'model_revision':next(iter(revisions)), 'record_count':len(targets),
              'problem_count':len(targets),'family_distribution':dict(Counter(r['family'] for r in targets)),
              'difficulty_distribution':dict(Counter(r['difficulty'] for r in targets)),
              'source_distribution':{'qwen_self':len(targets)},'license_distribution':{'Apache-2.0':len(targets)},
              'correctness_rate':sum(r['correct'] for r in targets)/len(targets),'natural_targets_only':True,'compression_applied':False,
              'reasoning_length_distribution':distribution([r['reasoning_tokens'] for r in targets]),
              'sequence_length_distribution':distribution([len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in targets]),
              'duplicate_target_count':len(targets)-len({digest(r['target_token_ids']) for r in targets}),
              'contamination_checks':contamination,'test_used':False,'dev_used':False,
              'training_content_excludes_review_prose':True, 'max_sequence_length':max_sequence_length,
              'scope':'Tiny optimizer-viability pool, not a capability improvement experiment'}
    write_json(manifest_path,manifest)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reviews',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--max-sequence-length',type=int,default=1024)
    p.add_argument('--max-examples',type=int,default=4)
    a=p.parse_args()
    if a.max_examples < 1 or a.max_sequence_length < 1:
        p.error('Positive example/sequence bounds required')
    print(json.dumps(build(a.reviews,a.output,a.manifest,a.max_sequence_length,a.max_examples),indent=2))


if __name__=='__main__':
    main()
