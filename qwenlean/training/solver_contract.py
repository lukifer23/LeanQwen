"""Separate fail-closed solver-source approval, including exact template/EOS IDs."""

import hashlib
from pathlib import Path

from qwenlean.datasets.solver_tasks import (
    GENERATOR_VERSION,
    SOLVER_VERSION,
    VERSION,
    body_for,
    render_proof,
)
from qwenlean.datasets.solver_verify import VERSION as VERIFIER_VERSION
from qwenlean.datasets.solver_verify import verify_proof
from qwenlean.inference.parsing import parse_tokens
from qwenlean.utils.io import digest, read_jsonl

MODEL = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'


class ProofTokenizer:
    """CPU-only proxy; do not modify Hugging Face special-token metadata."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_ids = {tokenizer.convert_tokens_to_ids(t)
                              for t in ('<|im_end|>', '<|endoftext|>')}

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    files = ['qwenlean/datasets/solver_tasks.py', 'qwenlean/datasets/solver_verify.py',
             'qwenlean/training/solver_contract.py']
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}


def pinned_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    return ProofTokenizer(tokenizer)


def tokenizer_contract(tokenizer):
    return {'model': MODEL, 'revision': REVISION,
            'template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            'vocabulary_hash': digest(tokenizer.get_vocab()),
            'thinking_prefill': '<think>\n', 'final_eos': '<|im_end|>'}


def tokenize_task(task, tokenizer, token_contract=None):
    prompt = task['body'] + ' On the final line, write only the integer answer.'
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if not rendered.endswith('<think>\n'):
        raise ValueError('Official thinking prefill mismatch')
    reasoning = render_proof(task['proof'])
    response = reasoning+'\n</think>\n\n'+task['expected']+'<|im_end|>'
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    target_ids = tokenizer.encode(response, add_special_tokens=False)
    parsed = parse_tokens(target_ids, tokenizer, True)
    closing = tokenizer.convert_tokens_to_ids('</think>')
    opening = tokenizer.convert_tokens_to_ids('<think>')
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if (parsed.parse_status != 'complete' or parsed.final.strip() != task['expected']
        or target_ids.count(closing) != 1 or opening in target_ids or target_ids[-1] != eos
        or any(t in tokenizer.eos_token_ids for t in target_ids[:-1])
        or tokenizer.decode(target_ids) != response):
        raise ValueError('Invalid complete thinking/final/EOS contract')
    certificate_id = digest(task['proof'])
    return {**task, 'prompt': prompt, 'prompt_policy': 'P2', 'rendered_prompt': rendered,
            'rendered_response': response, 'prompt_token_ids': prompt_ids,
            'target_token_ids': target_ids, 'reasoning_tokens': parsed.reasoning_tokens,
            'model_revision': REVISION, 'tokenizer_contract': token_contract or tokenizer_contract(tokenizer),
            'proof_certificate_id': certificate_id, 'source_hashes': source_hashes(),
            'correct': True, 'termination_reason': 'eos',
            'termination_origin': 'solver_rendered_end_of_turn_not_natural_generation'}


def validate_solver_record(row):
    p = row['provenance']
    if (p.get('response_origin') != 'solver_verified_reasoning'
        or p.get('prompt_origin') != 'algorithmic' or p.get('license') != 'Apache-2.0'
        or p.get('training_permitted') is not True or row['split'] != 'train'
        or row.get('namespace') != 'train' or p.get('task_generator') != VERSION
        or p.get('generator_version') != GENERATOR_VERSION or p.get('solver_version') != SOLVER_VERSION
        or row.get('source_hashes') != source_hashes() or row.get('model_revision') != REVISION
        or row.get('body') != body_for(row['family'], row['facts'])
        or type(row.get('generator_seed')) is not int or type(row.get('generator_attempt')) is not int):
        raise ValueError('Solver-source provenance/version/permission mismatch')
    verified = verify_proof(row['family'], row['facts'], row['proof'])
    if (verified != row['proof'].get('verification')
        or verified['verifier_version'] != VERIFIER_VERSION
        or verified['independent_expected_answer'] != row['expected']
        or row.get('proof_certificate_id') != digest(row['proof'])):
        raise ValueError('Invalid independent proof certificate')
    return True


def validate_solver_sources(rows, manifest):
    parents = read_jsonl(manifest['solver_records_path'])
    if digest(parents) != manifest.get('solver_records_hash'):
        raise ValueError('Solver archive hash mismatch')
    from qwenlean.datasets.solver_tasks import generate
    recipe = manifest.get('generator_recipe', {})
    if recipe.get('namespace') != 'train' or digest(generate('train', recipe['per_family'])) != digest(parents):
        raise ValueError('Solver archive cannot be reproduced from generator seed/recipe')
    lookup = {p['task_id']: p for p in parents}
    if len(lookup) != len(parents):
        raise ValueError('Duplicate solver archive tasks')
    tokenizer = pinned_tokenizer()
    contract = tokenizer_contract(tokenizer)
    for row in rows:
        validate_solver_record(row)
        parent = lookup.get(row['task_id'])
        if parent is None or tokenize_task(parent, tokenizer, contract) != row:
            raise ValueError('Solver target differs from exact proof/archive/template rendering')
    if set(manifest.get('approved_solver_certificate_ids', [])) != {r['proof_certificate_id'] for r in rows}:
        raise ValueError('Every solver proof must be explicitly approved')
    return True
