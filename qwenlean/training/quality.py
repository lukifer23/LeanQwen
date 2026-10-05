"""Training entrypoint contracts, independent of device/framework."""

from qwenlean.datasets.provenance import validate_provenance
from qwenlean.utils.io import digest


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
