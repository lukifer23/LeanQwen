"""External-teacher-only SFT permission, exact source and independent review gates.

Never accepts student, weak local, algorithmically authored or relabeled targets.
Frontier generation is not implemented here; its optional receipts require explicit
provider/spend authorization and audited model/output permissions.
"""

import hashlib
import json
import math
import re
from pathlib import Path

from qwenlean.datasets.external_reasoning import (
    GSM8K_LICENSE_SHA256,
    GSM8K_REVISION,
    GSM8K_SOURCE,
    GSM8K_TRAIN_SHA256,
    candidate_record,
    source_parts,
)
from qwenlean.inference.parsing import parse_tokens
from qwenlean.training.solver_contract import REVISION, pinned_tokenizer, tokenizer_contract
from qwenlean.utils.io import digest, read_jsonl

POLICY = 'external-teacher-only-v1'
R1_MODEL_REVISION = '56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad'
R1_LICENSE_SHA256 = '1c8f573e830ca9b3ebfeb7ace1823146e22b66f99ee223840e7637c9e745e1c7'
DEEPSEEK_API_TERMS_SHA256 = '2b433c53cbac75491959025eba5a27908f9f1886352dce0a05f3d5a936d87790'
GSM8K_QA_URL = 'https://arxiv.org/html/2110.14168v2'
GSM8K_QA_SHA256 = '4ea90b1486ef0cb823e899b36b9c396a9f893b8ad8bf83bdec65a8555d1274cf'
REVIEW_DIMENSIONS = ('final_answer_correct', 'mathematically_consistent', 'relevant',
                     'complete', 'productive_reasoning_without_obvious_redundancy')
PERMISSION_FIELDS = {
    'source_id': GSM8K_SOURCE, 'dataset_revision': GSM8K_REVISION,
    'publisher': 'OpenAI', 'license': 'MIT', 'originating_author': 'human_problem_writer',
    'teacher_model': None, 'source_split': 'train',
    'dataset_file_sha256': GSM8K_TRAIN_SHA256, 'license_file_sha256': GSM8K_LICENSE_SHA256,
    'source_url': f'https://github.com/openai/grade-school-math/tree/{GSM8K_REVISION}',
    'license_url': f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM8K_REVISION}/LICENSE',
    'permission_basis': 'Original human-written TRAIN responses distributed under MIT; not API outputs or Socratic/model-generated solutions',
}


def reject_prohibited_teacher(provenance):
    if provenance.get('response_origin') not in {'licensed_external_reasoning', 'authorized_frontier_reasoning'}:
        raise ValueError('Only independently sourced external reasoning is permitted; Qwen/self/solver/local targets are excluded')
    # Require explicit upstream identities; an empty/renamed teacher alone is not evidence.
    identities = [provenance.get(k) for k in ('teacher_model', 'base_model', 'originating_model', 'originating_author')]
    identities.extend(provenance.get('upstream_model_ids', []))
    for identity in identities:
        if identity and re.search(r'qwen|qwq|liquid|\blfm\w*|leanqwen|small.local|weak.local|student|codex|claude|gemini|\bgpt', str(identity), re.I):
            raise ValueError('Prohibited or not explicitly permitted teacher/ancestor identity')
    return True


def checked_file(path, expected_hash, max_bytes=8*1024**2):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError('Missing or oversized source permission artifact')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Source permission/archive file fingerprint differs')
    return path


def validate_frontier_certificate(certificate, config):
    """Optional strict schema; no provider is activated by a dataset-card license."""
    if certificate.get('authorization_id') not in config.get('user_approved_frontier_authorizations', []):
        raise ValueError('Explicit user approval of frontier provider and spend required')
    if (certificate.get('provider') != 'api.deepseek.com' or certificate.get('teacher_model') != 'deepseek-ai/DeepSeek-R1'
        or certificate.get('model_class') != 'full_671b_frontier' or certificate.get('model_license') != 'MIT'
        or certificate.get('output_distillation_permitted') is not True
        or certificate.get('teacher_model_revision') != R1_MODEL_REVISION):
        raise ValueError('Audited full frontier model identity/output permissions required')
    receipt = json.loads(checked_file(certificate['user_authorization_path'], certificate['user_authorization_sha256']).read_text())
    spend = receipt.get('max_spend_usd')
    if (receipt.get('authorized_by') != 'user' or receipt.get('provider') != 'api.deepseek.com'
        or receipt.get('authorization_id') != certificate['authorization_id']
        or not isinstance(spend, (int, float)) or not math.isfinite(spend) or spend < 0
        or certificate.get('actual_spend_usd', float('inf')) > spend):
        raise ValueError('Matching explicit user provider/spend receipt required')
    if certificate.get('provider_terms_url') != 'https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html':
        raise ValueError('Official API provider terms required, separate from model license')
    if certificate.get('provider_terms_sha256') != DEEPSEEK_API_TERMS_SHA256 or certificate.get('model_license_sha256') != R1_LICENSE_SHA256:
        raise ValueError('Unregistered frontier permission evidence; audit cannot be replaced by self-assertion')
    checked_file(certificate['provider_terms_path'], DEEPSEEK_API_TERMS_SHA256)
    checked_file(certificate['model_license_path'], R1_LICENSE_SHA256)
    if certificate.get('permission_review_status') != 'verified':
        raise ValueError('Frontier permission review is pending')
    return True


def authorize_training_manifest(manifest, config):
    """Call before reading tokenized dataset rows or creating/loading any tensors."""
    if manifest.get('training_source_policy') != POLICY:
        raise ValueError('External-teacher-only policy required; historical Qwen/solver approvals cannot authorize SFT')
    if manifest.get('status') != 'quality_approved':
        raise ValueError('Independent per-record quality approval is still pending')
    if manifest.get('quality_approval_basis') == 'published_human_qa_with_exact_screening':
        if type(manifest.get('record_count')) is not int or not 1 <= manifest['record_count'] <= 1000:
            raise ValueError('Published human QA approval is limited to a bounded reference pilot')
    counts = manifest.get('source_distribution', {})
    if (not counts or not set(counts).issubset({'licensed_external_reasoning', 'authorized_frontier_reasoning'})
        or any(type(n) is not int or n < 1 for n in counts.values())):
        raise ValueError('Prohibited/unknown/invalid training source counts')
    if not manifest.get('reviews_path') or not manifest.get('reviews_hash') or not manifest.get('approved_source_record_hashes'):
        raise ValueError('Hashed independent reviews and approved source records required')
    permissions = manifest.get('source_permissions', [])
    if not permissions:
        raise ValueError('Source-specific permission certificates required')
    for certificate in permissions:
        if certificate.get('source_id') == GSM8K_SOURCE:
            if any(certificate.get(k) != v for k, v in PERMISSION_FIELDS.items()):
                raise ValueError('Original human TRAIN dataset permission pin mismatch')
            checked_file(certificate['dataset_path'], GSM8K_TRAIN_SHA256)
            checked_file(certificate['license_path'], GSM8K_LICENSE_SHA256)
        elif certificate.get('source_type') == 'authorized_frontier_reasoning':
            validate_frontier_certificate(certificate, config)
            # No parser/verification path has been approved for frontier raw records yet.
            raise ValueError('Frontier certificate schema valid, but source-specific ingestion/quality path is not yet enabled')
        else:
            raise ValueError('Unregistered external source; licensing alone cannot enable ingestion')
    if manifest.get('test_used') is not False or manifest.get('dev_used') is not False:
        raise ValueError('DEV/TEST responses must stay outside the teacher corpus')
    return True


def encode_external_target(row, tokenizer, contract=None):
    reasoning, _, expected = source_parts(row['source_record'])
    if expected.denominator != 1:
        raise ValueError('Initial integer-scoring pilot excludes nonintegral examples whole')
    # Only replace the source's final delimiter with official chat channels. No prose rewrite.
    final = str(expected.numerator)
    prompt = row['body']+' On the final line, write only the integer answer.'
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True)
    if not rendered.endswith('<think>\n'):
        raise ValueError('Pinned official thinking prefill required')
    response = reasoning+'\n</think>\n\n'+final+'<|im_end|>'
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    target_ids = tokenizer.encode(response, add_special_tokens=False)
    parsed = parse_tokens(target_ids, tokenizer, True)
    if (tokenizer.decode(target_ids) != response or parsed.parse_status != 'complete'
        or parsed.final.strip() != final or target_ids[-1] != tokenizer.convert_tokens_to_ids('<|im_end|>')
        or any(t in tokenizer.eos_token_ids for t in target_ids[:-1])
        or target_ids.count(tokenizer.convert_tokens_to_ids('</think>')) != 1
        or tokenizer.convert_tokens_to_ids('<think>') in target_ids):
        raise ValueError('Malformed target channels, first token or EOS')
    return {**row, 'prompt': prompt, 'prompt_policy': 'P2', 'rendered_prompt': rendered,
            'rendered_response': response, 'prompt_token_ids': prompt_ids,
            'target_token_ids': target_ids, 'reasoning_tokens': parsed.reasoning_tokens,
            'model_revision': REVISION, 'tokenizer_contract': contract or tokenizer_contract(tokenizer),
            'target_transform': {'version': 'gsm8k-channel-wrap-v1', 'reasoning_changed': False,
                                 'calculator_annotations_retained': True, 'final_integer_canonicalized': True,
                                 'source_final_delimiter': '####', 'chat_eos_appended': '<|im_end|>'},
            'termination_origin': 'complete_external_solution_wrapped_not_student_natural_eos'}


def validate_review(row, review, config=None):
    if (review.get('task_id') != row['task_id'] or review.get('source_record_hash') != row['source_record_hash']
        or review.get('approved_for_sft') is not True or not review.get('review_note')):
        raise ValueError('Exact source-bound independent reasoning-quality review required')
    if any(k in review for k in ('replacement_solution', 'rewritten_solution', 'target_response')):
        raise ValueError('Reviews cannot supply rewritten training targets')
    kind = review.get('reviewer_type')
    if kind == 'publisher_human_qa':
        reject_prohibited_teacher(row['provenance'])
        # Published human reference data is a separate, explicitly scoped basis.
        # This is not a claim of fresh individual inspection or formal prose proof.
        if (review.get('review_method') != 'gsm8k-published-human-qa-and-exact-screening-v1'
            or review.get('publication_url') != GSM8K_QA_URL
            or review.get('publication_sha256') != GSM8K_QA_SHA256
            or review.get('approval_scope') != 'bounded_original_human_reference_pilot'
            or review.get('individual_project_review_performed') is not False
            or review.get('prose_semantics_formally_verified') is not False
            or any(k in review for k in REVIEW_DIMENSIONS)
            or row['provenance'].get('teacher_model') is not None
            or row['provenance'].get('dataset_source') != GSM8K_SOURCE
            or row['provenance'].get('dataset_revision') != GSM8K_REVISION
            or row['provenance'].get('originating_author') != 'human_problem_writer'
            or row['verification'].get('annotation_consistent') is not True
            or row['efficiency_screen'].get('flags') != []):
            raise ValueError('Published human QA requires intact GSM8K and exact screening, without fabricated individual review')
        checked_file(review['publication_path'], GSM8K_QA_SHA256)
        return True
    if any(review.get(k) is not True for k in REVIEW_DIMENSIONS):
        raise ValueError('Individual review must approve every reasoning-quality dimension')
    if kind == 'human':
        if not review.get('reviewer_id') or review.get('review_method') != 'individual_source_solution_inspection' or review.get('reviewer_model'):
            raise ValueError('Human reviewer identity and individual inspection required')
        if re.search(r'qwen|qwq|lfm|liquid|codex|gpt|claude|gemini|leanqwen', review['reviewer_id'], re.I):
            raise ValueError('An automated/prohibited reviewer cannot be relabeled human')
    elif kind == 'authorized_frontier':
        validate_frontier_certificate(review['permission_certificate'], config or {})
        if not review.get('raw_review_response_sha256') or not review.get('raw_review_response_path'):
            raise ValueError('Hashed raw independent frontier review required')
        checked_file(review['raw_review_response_path'], review['raw_review_response_sha256'])
    else:
        raise ValueError('Only human or explicitly authorized independent frontier quality review permitted')
    return True


def validate_external_sources(rows, manifest, config):
    authorize_training_manifest(manifest, config)
    certificate = manifest['source_permissions'][0]
    parents = read_jsonl(certificate['dataset_path'])
    reviews = read_jsonl(manifest['reviews_path'])
    if digest(reviews) != manifest['reviews_hash']:
        raise ValueError('Independent review archive differs')
    lookup = {r['task_id']: r for r in reviews}
    if len(lookup) != len(reviews):
        raise ValueError('Duplicate review identities')
    if set(manifest.get('approved_source_record_hashes', [])) != {r['source_record_hash'] for r in rows}:
        raise ValueError('Every exact source record must be approved')
    tokenizer = pinned_tokenizer()
    contract = tokenizer_contract(tokenizer)
    for row in rows:
        reject_prohibited_teacher(row['provenance'])
        index = row['source_index']
        if type(index) is not int or not 0 <= index < len(parents):
            raise ValueError('Missing original TRAIN source row')
        parent = candidate_record(parents[index], index)
        # Imported content/provenance/verification must all equal original licensed data.
        if any(row.get(k) != v for k, v in parent.items() if k != 'prompt'):
            raise ValueError('External target/provenance differs from original source; relabeling is forbidden')
        if parent['verification']['annotation_consistent'] is not True:
            raise ValueError('Incorrect/unrecognized annotated arithmetic cannot be approved')
        encoded = encode_external_target(parent, tokenizer, contract)
        for key in ('prompt', 'rendered_prompt', 'rendered_response', 'prompt_token_ids', 'target_token_ids',
                    'tokenizer_contract', 'target_transform', 'reasoning_tokens', 'model_revision'):
            if row.get(key) != encoded[key]:
                raise ValueError('External target differs from exact source-preserving template/tokenization')
        review = lookup.get(row['task_id'])
        if review is None:
            raise ValueError('Missing individual reasoning-quality review')
        if row.get('quality_review_id') != digest(review):
            raise ValueError('Target must bind its exact approval record')
        if (review.get('reviewer_type') == 'publisher_human_qa'
            and manifest.get('quality_approval_basis') != 'published_human_qa_with_exact_screening'):
            raise ValueError('Published QA must retain its explicit bounded approval basis')
        validate_review(row, review, config)
    return True
