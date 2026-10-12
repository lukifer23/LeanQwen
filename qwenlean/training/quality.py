"""Training entrypoint contracts, independent of device/framework."""

import json
import math
from collections import Counter
from pathlib import Path

from qwenlean.datasets.provenance import validate_provenance
from qwenlean.utils.io import digest, read_jsonl


def validate_historical_training_pool(rows, manifest, config):
    """Frozen natural/solver contracts for offline reproduction, never SFT admission."""
    if not rows:
        raise ValueError('Empty training pool')
    if manifest.get('status') != 'quality_approved' or manifest.get('dataset_hash') != digest(rows):
        raise ValueError('Matching approved dataset-quality manifest required')
    if manifest.get('test_used') is not False or manifest.get('dev_used') is not False:
        raise ValueError('No DEV/TEST responses permitted')
    if manifest.get('model_revision') != config['revision']:
        raise ValueError('Model revision pin mismatch')
    natural = [r for r in rows if r['provenance']['response_origin'] == 'qwen_self']
    solver = [r for r in rows if r['provenance']['response_origin'] == 'solver_verified_reasoning']
    if len(natural) + len(solver) != len(rows):
        raise ValueError('No approval path enabled for external or unknown sources')
    if set(manifest.get('approved_parent_ids', [])) != {r['parent_generation_id'] for r in natural}:
        raise ValueError('Every parent trajectory must be explicitly quality approved')
    if len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Only one selected target per distinct task')
    if solver:
        from qwenlean.training.solver_contract import validate_solver_record
        if set(manifest.get('approved_solver_certificate_ids', [])) != {r['proof_certificate_id'] for r in solver}:
            raise ValueError('Explicit solver certificate approval required')
        counts = dict(Counter(r['provenance']['response_origin'] for r in rows))
        if manifest.get('source_distribution') != counts or manifest.get('source_sampling_proportions') != {k: v/len(rows) for k, v in counts.items()}:
            raise ValueError('Exact source counts and sampling proportions required')
        if manifest.get('contamination_checks', {}).get('passed') is not True:
            raise ValueError('Solver contamination audit required')
    for r in rows:
        validate_provenance(r, for_training=True)
        if r['provenance']['response_origin'] == 'qwen_self':
            if not r['provenance'].get('parent_ids'):
                raise ValueError('Natural Qwen targets require parent IDs')
        else:
            validate_solver_record(r)
        if r.get('correct') is not True or r.get('termination_reason') != 'eos':
            raise ValueError('Only correct cleanly terminated natural targets approved')
        if r.get('model_revision') != config['revision']:
            raise ValueError('Target model revision differs')
        if not r.get('prompt_token_ids') or not r.get('target_token_ids'):
            raise ValueError('Exact prompt and response token IDs required')
        if any(type(t) is not int or t < 0 for t in r['prompt_token_ids'] + r['target_token_ids']):
            raise ValueError('Training token IDs must be nonnegative integers')
        if len(r['prompt_token_ids']) + len(r['target_token_ids']) > config['max_sequence_length']:
            raise ValueError('Example exceeds training sequence bound; truncation is forbidden')
    return validate_optimizer_config(config)


def validate_optimizer_config(config):
    if type(config.get('loss_chunk_size', 32)) is not int or config.get('loss_chunk_size', 32) < 1:
        raise ValueError('Positive integer loss chunk size required')
    if type(config.get('mlx_cache_limit_bytes', 64*1024**2)) is not int or config.get('mlx_cache_limit_bytes', 64*1024**2) < 0:
        raise ValueError('Nonnegative integer MLX cache limit required')
    if type(config.get('materialize_frozen_prefix', False)) is not bool:
        raise ValueError('Boolean frozen-prefix execution option required')
    if config.get('materialize_frozen_prefix', False):
        raise ValueError('Frozen-prefix materialization disabled after unsafe GPU pressure')
    for key in ('training_admission_ram_bytes', 'max_start_swap_used_bytes',
                'max_swap_used_growth_bytes', 'max_process_rss_bytes', 'mlx_memory_limit_bytes',
                'max_training_wall_s', 'max_gradient_wall_s'):
        if key in config and (type(config[key]) is not int or config[key] <= 0):
            raise ValueError('Positive integer training watchdog/resource limits required')
    for key in ('steps', 'batch_size', 'gradient_accumulation', 'max_sequence_length', 'num_layers'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Positive {key} required')
    for key in ('learning_rate', 'checkpoint_every'):
        value = config.get(key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'Finite positive {key} required')
    if 'checkpoint_every' in config and type(config['checkpoint_every']) is not int:
        raise ValueError('Integer checkpoint_every required')
    lora = config['lora_parameters']
    if 'rank' in lora and (type(lora['rank']) is not int or lora['rank'] < 1):
        raise ValueError('Positive integer LoRA rank required')
    if 'scale' in lora and (not math.isfinite(lora['scale']) or lora['scale'] <= 0):
        raise ValueError('Finite positive LoRA scale required')
    if 'dropout' in lora and (not math.isfinite(lora['dropout']) or not 0 <= lora['dropout'] < 1):
        raise ValueError('LoRA dropout must be in [0, 1)')
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


def validate_historical_parent_sources(rows, manifest):
    solver = [r for r in rows if r['provenance']['response_origin'] == 'solver_verified_reasoning']
    if solver:
        from qwenlean.training.solver_contract import validate_solver_sources
        validate_solver_sources(solver, manifest)
    rows = [r for r in rows if r['provenance']['response_origin'] == 'qwen_self']
    if not rows:
        return True
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


def validate_training_pool(rows, manifest, config):
    from qwenlean.training.external_contract import (
        authorize_training_manifest,
        reject_prohibited_teacher,
    )
    authorize_training_manifest(manifest, config)
    if not rows or digest(rows) != manifest.get('dataset_hash'):
        raise ValueError('Nonempty matching approved external dataset required')
    if manifest.get('record_count') != len(rows) or manifest.get('quality_approved_count') != len(rows):
        raise ValueError('Approved record counts must equal the actual pool')
    if manifest.get('model_revision') != config['revision']:
        raise ValueError('Student model revision pin mismatch')
    if len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate external task identities')
    counts = dict(Counter(r['provenance']['response_origin'] for r in rows))
    if (manifest.get('source_distribution') != counts
        or manifest.get('source_sampling_proportions') != {k: v/len(rows) for k, v in counts.items()}):
        raise ValueError('Exact external source counts and sampling proportions required')
    if manifest.get('contamination_checks', {}).get('passed') is not True:
        raise ValueError('Source/benchmark contamination checks required')
    for row in rows:
        reject_prohibited_teacher(row['provenance'])
        validate_provenance(row, for_training=True)
        if row.get('correct') is not True or row.get('termination_reason') != 'eos':
            raise ValueError('Correct source-reviewed complete terminal target required')
        if row.get('model_revision') != config['revision']:
            raise ValueError('Target student/tokenizer revision differs')
        if not row.get('prompt_token_ids') or not row.get('target_token_ids'):
            raise ValueError('Exact prompt and complete supervised response IDs required')
        tokens = row['prompt_token_ids']+row['target_token_ids']
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError('Training token IDs must be nonnegative integers')
        if len(tokens) > config['max_sequence_length']:
            raise ValueError('Whole external example exceeds sequence bound; truncation forbidden')
    return validate_optimizer_config(config)


def validate_parent_sources(rows, manifest, config=None):
    from qwenlean.training.external_contract import validate_external_sources
    return validate_external_sources(rows, manifest, config or {})


def load_approved_training_inputs(config, dataset, manifest_path):
    """Source permission first, then JSON records, then exact source/review checks."""
    from qwenlean.training.external_contract import authorize_training_manifest
    manifest = json.loads(Path(manifest_path).read_text())
    authorize_training_manifest(manifest, config)
    rows = read_jsonl(dataset)
    validate_training_pool(rows, manifest, config)
    validate_parent_sources(rows, manifest, config)
    return rows, manifest
