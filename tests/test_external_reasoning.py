"""Source permission, independent review, exact arithmetic and no silent relabeling."""

import json
from pathlib import Path

import pytest

from qwenlean.datasets.external_reasoning import (
    GSM8K_TRAIN_SHA256,
    exact_value,
    source_parts,
    verify_annotations,
)
from qwenlean.training.external_contract import (
    POLICY,
    REVIEW_DIMENSIONS,
    authorize_training_manifest,
    checked_file,
    reject_prohibited_teacher,
    validate_frontier_certificate,
    validate_review,
)
from qwenlean.training.quality import load_approved_training_inputs, validate_training_pool


def unit_record(answer):
    # Synthetic checker fixtures, never saved as teacher data.
    return {'question': 'Unit test fixture only.', 'answer': answer}


def test_exact_decimal_and_chained_annotations():
    assert str(exact_value('0.1 + 0.2')) == '3/10'
    assert exact_value('1,000,000 / 2') == 500000
    check = verify_annotations(unit_record('Check <<0.1+0.2=0.3=3/10>>0.3, then <<0.3*10=3>>3.\n#### 3'))
    assert check['annotation_consistent']
    assert len(check['verified_annotations']) == 2
    assert check['prose_semantics_verified'] is False


@pytest.mark.parametrize('text', ['2**30', 'x+1', '1,20', '__import__("os")', '1%2', '1+'])
def test_unknown_or_unfinished_arithmetic_is_not_certified(text):
    with pytest.raises((ValueError, SyntaxError)):
        exact_value(text)


def test_final_correctness_does_not_certify_invalid_or_unparsed_steps():
    incorrect = verify_annotations(unit_record('<<6*17=108>>108, then <<1+1=2>>2.\n#### 2'))
    assert not incorrect['annotation_consistent'] and len(incorrect['invalid_annotations']) == 1
    unknown = verify_annotations(unit_record('<<x+1=2>>2.\n#### 2'))
    assert not unknown['annotation_consistent'] and len(unknown['unverified_annotations']) == 1
    rejected = verify_annotations(unit_record('The rejected hypothesis <<1+1=3>>3 is wrong. <<1+1=2>>2.\n#### 2'))
    assert not rejected['annotation_consistent']  # Does not silently ignore an invalid claim.
    with pytest.raises(ValueError, match='Unclosed'):
        verify_annotations(unit_record('<<1+1=2\n#### 2'))
    with pytest.raises(ValueError):
        source_parts(unit_record('1+1=2\n#### 2\ncontinued reconsideration'))


@pytest.mark.parametrize('identity', ['Qwen/Qwen3.5-0.8B', 'QwQ-32B', 'DeepSeek-R1-Distill-Qwen-32B',
                                      'LFM2.5', 'Liquid', 'LeanQwen', 'small_local_teacher',
                                      'GPT-5', 'Codex', 'Claude', 'Gemini'])
def test_disguised_prohibited_teacher_and_ancestry(identity):
    provenance = {'response_origin': 'licensed_external_reasoning', 'teacher_model': identity}
    with pytest.raises(ValueError, match='Prohibited'):
        reject_prohibited_teacher(provenance)
    provenance['teacher_model'] = 'unverified-renamed-model'
    provenance['upstream_model_ids'] = [identity]
    with pytest.raises(ValueError):
        reject_prohibited_teacher(provenance)


@pytest.mark.parametrize('origin', ['qwen_self', 'solver_verified_reasoning', 'algorithmic_transform', 'licensed_dataset', 'unknown'])
def test_old_or_relabeled_source_is_not_a_current_teacher(origin):
    with pytest.raises(ValueError):
        reject_prohibited_teacher({'response_origin': origin})
    with pytest.raises(ValueError, match='External-teacher-only'):
        validate_training_pool([], {'status': 'quality_approved', 'source_distribution': {origin: 1}}, {})


def test_permission_gate_precedes_dataset_read_or_mlx_import(tmp_path, monkeypatch):
    def forbidden_read(*args):
        raise AssertionError('Dataset/tensor data was read before permission')
    monkeypatch.setattr('qwenlean.training.quality.read_jsonl', forbidden_read)
    manifest = tmp_path/'historical.json'
    manifest.write_text(json.dumps({'status': 'quality_approved', 'source_distribution': {'qwen_self': 1}}))
    with pytest.raises(ValueError, match='External-teacher-only'):
        load_approved_training_inputs({}, 'must-never-be-read.jsonl', manifest)
    manifest.write_text(json.dumps({'training_source_policy': POLICY, 'status': 'independent_quality_review_pending'}))
    with pytest.raises(ValueError, match='pending'):
        load_approved_training_inputs({}, 'must-never-be-read.jsonl', manifest)


def test_source_hash_cannot_be_replaced_by_a_claimed_license(tmp_path):
    fake = tmp_path/'source.jsonl'
    fake.write_text('{"question":"a copied student prompt","answer":"a student response"}\n')
    with pytest.raises(ValueError, match='fingerprint'):
        checked_file(fake, GSM8K_TRAIN_SHA256)
    with pytest.raises(ValueError):
        authorize_training_manifest({'training_source_policy': POLICY, 'status': 'quality_approved'}, {})


def test_no_automatic_quality_approval_or_rewrite_from_reviewer():
    row = {'task_id': 'fixture', 'source_record_hash': 'abc'}
    review = {**row, 'approved_for_sft': True, 'reviewer_type': 'human', 'reviewer_id': 'unit_fixture_not_real_review',
              'review_method': 'individual_source_solution_inspection', 'review_note': 'Fixture only.',
              **{k: True for k in REVIEW_DIMENSIONS}}
    assert validate_review(row, review)
    review['replacement_solution'] = 'Forbidden authored rewrite'
    with pytest.raises(ValueError, match='rewritten'):
        validate_review(row, review)
    del review['replacement_solution']
    review['reviewer_type'] = 'coding_agent'
    with pytest.raises(ValueError, match='Only human'):
        validate_review(row, review)
    review['reviewer_type'] = 'human'
    review['mathematically_consistent'] = False
    with pytest.raises(ValueError, match='review'):
        validate_review(row, review)


def test_paid_frontier_contract_never_enables_itself():
    with pytest.raises(ValueError, match='user approval'):
        validate_frontier_certificate({'provider': 'api.deepseek.com', 'authorization_id': 'not-authorized'}, {})


def test_published_qa_keeps_prose_unverified_and_rejects_flagged_or_false_claims(monkeypatch):
    from qwenlean.datasets.external_reasoning import GSM8K_REVISION, GSM8K_SOURCE
    from qwenlean.training.external_contract import GSM8K_QA_SHA256, GSM8K_QA_URL
    # Permission artifact IO mocked only in this unit fixture; no actual approvals.
    monkeypatch.setattr('qwenlean.training.external_contract.checked_file', lambda *args: True)
    row = {'task_id': 'fixture', 'source_record_hash': 'abc',
           'provenance': {'response_origin': 'licensed_external_reasoning', 'teacher_model': None,
                          'originating_author': 'human_problem_writer', 'dataset_source': GSM8K_SOURCE,
                          'dataset_revision': GSM8K_REVISION},
           'verification': {'annotation_consistent': True}, 'efficiency_screen': {'flags': []}}
    review = {'task_id': 'fixture', 'source_record_hash': 'abc', 'approved_for_sft': True,
              'reviewer_type': 'publisher_human_qa', 'review_note': 'Published QA; fixture only.',
              'review_method': 'gsm8k-published-human-qa-and-exact-screening-v1',
              'publication_url': GSM8K_QA_URL, 'publication_sha256': GSM8K_QA_SHA256,
              'publication_path': 'fixture', 'approval_scope': 'bounded_original_human_reference_pilot',
              'individual_project_review_performed': False, 'prose_semantics_formally_verified': False}
    assert validate_review(row, review)
    for key in ('individual_project_review_performed', 'prose_semantics_formally_verified'):
        with pytest.raises(ValueError, match='fabricated'):
            validate_review(row, {**review, key: True})
    with pytest.raises(ValueError):
        validate_review(row, {**review, 'publication_sha256': 'invented'})
    row['efficiency_screen']['flags'] = ['repetition']
    with pytest.raises(ValueError):
        validate_review(row, review)


@pytest.mark.local_tokenizer
def test_real_imported_source_and_exact_template_roundtrip():
    from qwenlean.datasets.external_reasoning import candidate_record
    from qwenlean.training.external_contract import encode_external_target
    from qwenlean.training.solver_contract import pinned_tokenizer, tokenizer_contract
    path = Path('data/raw/external-audit-20261012/gsm8k-train.jsonl')
    if not path.exists():
        pytest.skip('Pinned licensed source archive not cached; never download in tests')
    raw = json.loads(path.read_text().splitlines()[0])
    tokenizer = pinned_tokenizer()
    target = encode_external_target(candidate_record(raw, 0), tokenizer, tokenizer_contract(tokenizer))
    assert target['target_transform']['reasoning_changed'] is False
    assert target['rendered_response'].startswith(source_parts(raw)[0])
    assert '<think>' not in target['rendered_response']
    assert target['target_token_ids'][-1] == 248046
    assert target['provenance']['teacher_model'] is None
    assert target['provenance']['response_origin'] == 'licensed_external_reasoning'


@pytest.mark.local_tokenizer
def test_exact_source_and_review_archive_rejects_rewriting(tmp_path):
    from qwenlean.datasets.external_reasoning import candidate_record
    from qwenlean.training.external_contract import (
        PERMISSION_FIELDS,
        encode_external_target,
        validate_external_sources,
    )
    from qwenlean.training.solver_contract import pinned_tokenizer, tokenizer_contract
    from qwenlean.utils.io import digest, write_jsonl
    archive = Path('data/raw/external-audit-20261012')
    if not archive.exists():
        pytest.skip('Pinned licensed source not cached')
    raw = json.loads((archive/'gsm8k-train.jsonl').read_text().splitlines()[0])
    tokenizer = pinned_tokenizer()
    row = encode_external_target(candidate_record(raw, 0), tokenizer, tokenizer_contract(tokenizer))
    # Isolated validation fixture; not a real human approval or corpus manifest.
    review = {'task_id': row['task_id'], 'source_record_hash': row['source_record_hash'],
              'reviewer_type': 'human', 'reviewer_id': 'unit_fixture_not_real_review',
              'review_method': 'individual_source_solution_inspection', 'approved_for_sft': True,
              'review_note': 'Unit fixture only; never written into the real corpus.',
              **{k: True for k in REVIEW_DIMENSIONS}}
    reviews = tmp_path/'fixture-reviews.jsonl'
    write_jsonl(reviews, [review])
    row['quality_review_id'] = digest(review)
    manifest = {'training_source_policy': POLICY, 'status': 'quality_approved', 'test_used': False, 'dev_used': False,
                'source_distribution': {'licensed_external_reasoning': 1},
                'reviews_path': str(reviews), 'reviews_hash': digest([review]),
                'approved_source_record_hashes': [row['source_record_hash']],
                'source_permissions': [{**PERMISSION_FIELDS, 'dataset_path': str(archive/'gsm8k-train.jsonl'),
                                         'license_path': str(archive/'gsm8k-LICENSE')}]}
    assert validate_external_sources([row], manifest, {})
    row['rendered_response'] = 'A fabricated compressed answer.'
    with pytest.raises(ValueError, match='source-preserving'):
        validate_external_sources([row], manifest, {})
    row['provenance']['teacher_model'] = 'LFM'
    with pytest.raises(ValueError, match='Prohibited'):
        validate_external_sources([row], manifest, {})
