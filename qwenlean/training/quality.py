"""Training entrypoint contracts, independent of device/framework."""

from qwenlean.datasets.provenance import validate_provenance
from qwenlean.utils.io import digest, read_jsonl


def validate_training_pool(rows, manifest, config):
    if not rows:
        raise ValueError('Empty training pool')
    if manifest.get('status') != 'quality_approved' or manifest.get('dataset_hash') != digest(rows):
        raise ValueError('Matching approved dataset-quality manifest required')
    if manifest.get('test_used') is not False or manifest.get('dev_used') is not False:
        raise ValueError('No DEV/TEST responses permitted')
    if manifest.get('model_revision') != config['revision']:
        raise ValueError('Model revision pin mismatch')
    if set(manifest.get('approved_parent_ids', [])) != {r['parent_generation_id'] for r in rows}:
        raise ValueError('Every parent trajectory must be explicitly quality approved')
    if len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Only one selected natural target per task in first SFT pool')
    for r in rows:
        validate_provenance(r, for_training=True)
        if r['provenance']['response_origin'] != 'qwen_self' or not r['provenance'].get('parent_ids'):
            raise ValueError('First pool requires natural Qwen targets with parent IDs')
        if r.get('correct') is not True or r.get('termination_reason') != 'eos':
            raise ValueError('Only correct cleanly terminated natural targets approved')
        if r.get('model_revision') != config['revision']:
            raise ValueError('Target model revision differs')
        if not r.get('prompt_token_ids') or not r.get('target_token_ids'):
            raise ValueError('Exact prompt and response token IDs required')
        if len(r['prompt_token_ids']) + len(r['target_token_ids']) > config['max_sequence_length']:
            raise ValueError('Example exceeds training sequence bound; truncation is forbidden')
    for key in ('steps', 'batch_size', 'gradient_accumulation', 'max_sequence_length', 'num_layers'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Positive {key} required')
    if config['batch_size'] != 1:
        raise ValueError('Current implementation supports batch size 1; use accumulation')
    if config.get('optimizer') != 'adamw':
        raise ValueError('Only implemented AdamW optimizer allowed')
    if not 1 <= config['num_layers'] <= 24:
        raise ValueError('Invalid adapter layer count')
    expected = ['self_attn.q_proj', 'self_attn.v_proj', 'linear_attn.in_proj_qkv']
    if config['lora_parameters'].get('keys') != expected:
        raise ValueError('Explicit validated mixed-attention target modules required')
    return True


def supervision_range(prompt_ids, target_ids):
    # Logit at prompt_length-1 predicts the first response. Last position predicts EOS.
    if not prompt_ids or not target_ids:
        raise ValueError('Both prompt and target must be nonempty')
    return len(prompt_ids)-1, len(prompt_ids)+len(target_ids)-1


def validate_parent_sources(rows, manifest):
    if not manifest.get('source_records_path') or not manifest.get('source_records_hash'):
        raise ValueError('Hashed raw parent-generation archive required')
    parents = read_jsonl(manifest['source_records_path'])
    if digest(parents) != manifest['source_records_hash']:
        raise ValueError('Parent-generation archive hash differs')
    lookup = {p['generation_id']: p for p in parents}
    if len(lookup) != len(parents):
        raise ValueError('Duplicate raw parent generation IDs')
    for row in rows:
        parent = lookup.get(row['parent_generation_id'])
        if parent is None:
            raise ValueError('Missing measured parent response')
        validate_provenance(parent, for_training=True)
        if (parent['task_id'] != row['task_id'] or parent['prompt_token_ids'] != row['prompt_token_ids']
            or parent['token_ids'] != row['target_token_ids'] or parent['prompt'] != row['prompt']
            or parent['correct'] is not True or parent['termination_reason'] != 'eos'
            or parent['model_identifier']['revision'] != row['model_revision']):
            raise ValueError('Target differs from its exact natural measured TRAIN parent')
    return True
