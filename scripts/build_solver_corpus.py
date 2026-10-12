"""CPU-only reproducible solver corpus; no TEST inference or model initialization."""

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

from qwenlean.datasets.procedural_v2 import FAMILIES
from qwenlean.datasets.solver_tasks import (
    GENERATOR_VERSION,
    SEEDS,
    SOLVER_VERSION,
    VERSION,
    body_for,
    generate,
    solve,
)
from qwenlean.datasets.solver_verify import verify_proof
from qwenlean.training.solver_contract import (
    REVISION,
    pinned_tokenizer,
    source_hashes,
    tokenize_task,
    tokenizer_contract,
)
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock


def id_holdout(task, serial):
    """Same topology, fresh constants/entities. Counting shifts the whole domain."""
    row = copy.deepcopy(task)
    s, family = row['facts'], row['family']
    rng = random.Random(SEEDS['dev_id']+serial)
    if family in {'arithmetic', 'boolean_logic'}:
        def replace(node):
            if isinstance(node, int):
                return node+17
            if node[0] == 'var':
                return node
            return [node[0], *[replace(c) for c in node[1:]]]
        s['ast'] = replace(s['ast'])
        if family == 'boolean_logic':
            s['values'] = [not v for v in s['values']]
    elif family == 'state_machine':
        s['initial'] += 53
        s['modulus'] += 3
    elif family == 'sequence':
        s['input'] = [v+71 for v in s['input']]
    elif family == 'ordering':
        s['nodes'] = [n+1000 for n in s['nodes']]
        s['edges'] = [[a+1000, b+1000] for a, b in s['edges']]
    elif family == 'algebra':
        value = rng.randint(40, 60)
        for op, n in s['ops']:
            value = value+n if op == '+' else value-n if op == '-' else value*n
        s['rhs'] = value
    elif family == 'word_problem':
        s['baskets'] = [[b+11, i+13, p*2] for b, i, p in s['baskets']]
    elif family == 'counting':
        s['start'] = 101
    proof = solve(family, s)
    proof['verification'] = verify_proof(family, s, proof)
    task_id = f'dev_id-{VERSION}-{family}-{digest(s)[:20]}'
    row.update(task_id=task_id, sample_id=task_id, seed_key=task_id, split='dev',
               namespace='dev_id', facts=s, semantic_fingerprint=digest([family, s]),
               body=body_for(family, s), prompt=body_for(family, s), expected=proof['final_answer'],
               proof=proof, generator_seed=SEEDS['dev_id'],
               id_parent_task_id=task['task_id'])
    row['provenance'].update(source_id=task_id, training_permitted=False)
    return row


def distribution(values):
    ordered = sorted(values)
    return {'count': len(values), 'min': min(values), 'max': max(values),
            'mean': sum(values)/len(values), 'median': ordered[len(ordered)//2],
            'cumulative_under': {str(b): sum(v < b for v in values) for b in (512, 1024, 2048, 4096)},
            'values': values}


def build(output, per_family):
    output = Path(output)
    if output.exists():
        raise ValueError('Refuse to overwrite existing corpus')
    with model_process_lock():
        output.mkdir(parents=True)
        train = generate('train', per_family)
        # Two simple and two more dependent problems/family for each DEV namespace.
        selected = []
        for family in FAMILIES:
            tasks = sorted([t for t in train if t['family'] == family], key=lambda t: t['structural_complexity'])
            selected.extend([tasks[0], tasks[min(1, len(tasks)-1)], tasks[len(tasks)//2], tasks[-1]])
        dev_id = [id_holdout(t, i) for i, t in enumerate(selected)]
        dev_ood_all = generate('dev_ood', 16)
        dev_ood = []
        for family in FAMILIES:
            tasks = sorted([t for t in dev_ood_all if t['family'] == family], key=lambda t: t['structural_complexity'])
            dev_ood.extend([tasks[0], tasks[1], tasks[len(tasks)//2], tasks[-1]])
        # TEST generator metadata only. No proof/answer/tokenized TEST pool is emitted.
        train_structures = {r['structural_fingerprint'] for r in train}
        if len(train_structures) != len(train):
            raise ValueError('Duplicate normalized TRAIN structure')
        if len({r['response_structure_fingerprint'] for r in train}) != len(train):
            raise ValueError('Duplicate TRAIN solver response structure')
        if any(r['structural_fingerprint'] in train_structures for r in dev_ood):
            raise ValueError('Structural OOD leak')
        if any(r['structural_fingerprint'] not in train_structures for r in dev_id):
            raise ValueError('ID holdout does not share TRAIN structures')
        pools = train+dev_id+dev_ood
        if len({r['semantic_fingerprint'] for r in pools}) != len(pools) or len({r['prompt'] for r in pools}) != len(pools):
            raise ValueError('Semantic/prompt contamination')
        historical = [r for p in Path('data/splits').rglob('*.jsonl') for r in read_jsonl(p)]
        historical_prompts = {r.get('body', r['prompt']) for r in historical}
        if any(r['body'] in historical_prompts for r in pools):
            raise ValueError('Historical split prompt contamination')
        tokenizer = pinned_tokenizer()
        contract = tokenizer_contract(tokenizer)
        rows = [tokenize_task(t, tokenizer, contract) for t in train]
        write_jsonl(output/'solver-records.jsonl', train)
        write_jsonl(output/'train.jsonl', rows)
        for name, tasks in [('dev-id', dev_id), ('dev-ood', dev_ood)]:
            # Remove proof from evaluation stimuli; never pass expected derivations to Qwen.
            clean = [{k: v for k, v in t.items() if k != 'proof'} for t in tasks]
            write_jsonl(output/f'{name}.jsonl', clean)
        lengths = [len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in rows]
        checks = {'passed': True, 'train_structure_duplicates': 0, 'semantic_duplicates': 0,
                  'train_response_structure_duplicates': 0,
                  'exact_prompt_duplicates': 0, 'ood_structure_overlap': 0,
                  'id_structure_overlap_expected': len(dev_id), 'historical_prompt_overlap': 0,
                  'test_model_or_target_use': False,
                  'limitation': 'Normalized shape uniqueness does not prove conceptual diversity; word baskets and ordering transitive edges may share algorithms. ID intentionally shares topology.'}
        manifest = {'status': 'quality_approved', 'dataset_version': VERSION,
            'dataset_hash': digest(rows), 'solver_records_path': str(output/'solver-records.jsonl'),
            'solver_records_hash': digest(train), 'generator_recipe': {'namespace': 'train', 'per_family': per_family},
            'source_hashes': source_hashes(), 'generator_version': GENERATOR_VERSION,
            'solver_version': SOLVER_VERSION, 'model_revision': REVISION,
            'tokenizer_contract': contract, 'approved_solver_certificate_ids': [r['proof_certificate_id'] for r in rows],
            'approved_parent_ids': [], 'source_distribution': {'solver_verified_reasoning': len(rows)},
            'source_sampling_proportions': {'solver_verified_reasoning': 1.0},
            'license_distribution': {'Apache-2.0': len(rows)}, 'record_count': len(rows),
            'distinct_train_problems': len(rows), 'distinct_train_structures': len(train_structures),
            'family_distribution': dict(Counter(r['family'] for r in rows)),
            'difficulty_distribution': dict(Counter(r['difficulty'] for r in rows)),
            'reasoning_lengths': distribution([r['reasoning_tokens'] for r in rows]),
            'response_lengths': distribution([len(r['target_token_ids']) for r in rows]),
            'complete_sequence_lengths': distribution(lengths),
            'length_by_difficulty': {d: distribution([r['reasoning_tokens'] for r in rows if r['difficulty'] == d])
                                     for d in sorted({r['difficulty'] for r in rows})},
            'sequence_fit_counts': {str(b): sum(v <= b for v in lengths) for b in (1024, 2048, 4096)},
            'verification_failures': 0, 'contamination_checks': checks,
            'dev_hashes': {'id': digest(dev_id), 'ood': digest(dev_ood)},
            'holdout_questions': {'id': 'Fresh constants/entities on seen topology; numeric generalization',
                                 'ood': 'Hash-reserved normalized topologies; structural generalization'},
            'test_namespace': {'seed': SEEDS['test'], 'structural_hash_bucket': 4, 'sealed': True},
            'test_used': False, 'dev_used': False, 'truncation': False,
            'correction_examples': 0,
            'scope': 'Algorithmic operations rendered by fixed grammar; no assistant authored worked solutions or external corpus'}
        write_json(output/'manifest.json', manifest)
        return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--per-family', type=int, default=64)
    a = p.parse_args()
    report = build(a.output, a.per_family)
    print(json.dumps({k: report[k] for k in ('distinct_train_problems', 'family_distribution', 'difficulty_distribution', 'sequence_fit_counts')}))
