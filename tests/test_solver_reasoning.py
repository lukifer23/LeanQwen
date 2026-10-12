"""Independent routes, incomplete/forged proofs, source isolation and Qwen EOS."""

import copy
import random

import pytest

from qwenlean.datasets.procedural_v2 import FAMILIES
from qwenlean.datasets.solver_tasks import generate, solve, task_spec
from qwenlean.datasets.solver_verify import verify_proof


@pytest.mark.parametrize('family', FAMILIES)
def test_varied_structures_and_every_state(family):
    rng = random.Random(9181)
    for seed in range(24):
        spec, _, _ = task_spec(rng, family, seed)
        proof = solve(family, spec)
        result = verify_proof(family, spec, proof)
        assert result['verified'] and result['step_count'] == len(proof['steps'])
        # Every intermediate after-state must matter, not just the final answer.
        for index in range(len(proof['steps'])):
            corrupt = copy.deepcopy(proof)
            corrupt['steps'][index]['after'] = 'unrecognized/unfinished/rejected 1+1=0'
            with pytest.raises((ValueError, TypeError, IndexError)):
                verify_proof(family, spec, corrupt)
        for corrupt in [dict(proof, steps=proof['steps'][:-1]), dict(proof, final_answer='999999')]:
            with pytest.raises(ValueError):
                verify_proof(family, spec, corrupt)


def test_unique_reproducible_structures_and_reserved_namespaces():
    train = generate('train', 32)
    assert train == generate('train', 32)
    assert len({r['structural_fingerprint'] for r in train}) == 256
    assert len({r['response_structure_fingerprint'] for r in train}) == 256
    ood, test = generate('dev_ood', 4), generate('test', 4)
    assert not ({r['structural_fingerprint'] for r in train} & {r['structural_fingerprint'] for r in ood+test})
    assert all(r['split'] == 'train' and r['provenance']['response_origin'] == 'solver_verified_reasoning' for r in train)


@pytest.mark.local_tokenizer
def test_solver_official_thinking_token_roundtrip():
    from qwenlean.training.solver_contract import pinned_tokenizer, tokenize_task
    tokenizer = pinned_tokenizer()
    for task in generate('train', 1):
        row = tokenize_task(task, tokenizer)
        assert row['rendered_prompt'].endswith('<think>\n')
        assert '<think>' not in row['rendered_response']
        assert row['rendered_response'].endswith(task['expected']+'<|im_end|>')
        assert row['target_token_ids'][-1] == 248046


@pytest.mark.local_tokenizer
def test_solver_approval_requires_reproducible_archive_and_exact_tokens(tmp_path):
    import json
    from pathlib import Path

    import yaml

    from qwenlean.training.quality import validate_parent_sources, validate_training_pool
    from qwenlean.training.solver_contract import (
        REVISION,
        pinned_tokenizer,
        tokenize_task,
        tokenizer_contract,
    )
    from qwenlean.utils.io import digest, write_jsonl

    parents = generate('train', 1)
    tokenizer = pinned_tokenizer()
    contract = tokenizer_contract(tokenizer)
    rows = json.loads(json.dumps([tokenize_task(t, tokenizer, contract) for t in parents]))
    archive = tmp_path/'proofs.jsonl'
    write_jsonl(archive, parents)
    cfg = yaml.safe_load(Path('configs/training/smoke.yaml').read_text())
    cfg['max_sequence_length'] = 4096
    m = {'status': 'quality_approved', 'dataset_hash': digest(rows), 'test_used': False, 'dev_used': False,
         'model_revision': REVISION, 'approved_parent_ids': [], 'approved_solver_certificate_ids': [r['proof_certificate_id'] for r in rows],
         'source_distribution': {'solver_verified_reasoning': 8}, 'source_sampling_proportions': {'solver_verified_reasoning': 1.0},
         'contamination_checks': {'passed': True}, 'solver_records_path': str(archive), 'solver_records_hash': digest(parents),
         'generator_recipe': {'namespace': 'train', 'per_family': 1}}
    assert validate_training_pool(rows, m, cfg)
    assert validate_parent_sources(rows, m)
    rows[0]['target_token_ids'][0] += 1
    m['dataset_hash'] = digest(rows)
    with pytest.raises(ValueError, match='exact'):
        validate_parent_sources(rows, m)
    rows[0]['provenance']['response_origin'] = 'qwen_self'
    rows[0]['parent_generation_id'] = 'fake'
    m['dataset_hash'] = digest(rows)
    with pytest.raises(ValueError, match='parent'):
        validate_training_pool(rows, m, cfg)
    # The same mathematics cannot enter training under a fabricated source archive.
    parents[0]['generator_seed'] += 1
    write_jsonl(archive, parents)
    m['solver_records_hash'] = digest(parents)
    with pytest.raises(ValueError, match='reproduced'):
        validate_parent_sources([r for r in rows if r['provenance']['response_origin'] == 'solver_verified_reasoning'], m)
    assert json.loads(json.dumps(m)) == m
