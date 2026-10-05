import pytest

from qwenlean.training.quality import supervision_range, validate_training_pool
from qwenlean.utils.io import digest


def contracts():
    r = {'split': 'train', 'task_id': 't', 'parent_generation_id': 'p', 'correct': True,
         'termination_reason': 'eos', 'model_revision': 'abc', 'prompt_token_ids': [1, 2],
         'target_token_ids': [3, 4, 5], 'provenance': {'source_id': 's', 'parent_ids': ['p'],
          'prompt_origin': 'algorithmic', 'response_origin': 'qwen_self', 'license': 'Apache-2.0',
          'training_permitted': True}}
    cfg = {'revision': 'abc', 'steps': 4, 'batch_size': 1, 'gradient_accumulation': 1,
           'max_sequence_length': 8, 'num_layers': 2, 'optimizer': 'adamw',
           'lora_parameters': {'keys': ['self_attn.q_proj', 'self_attn.v_proj', 'linear_attn.in_proj_qkv']}}
    m = {'status': 'quality_approved', 'dataset_hash': digest([r]), 'test_used': False,
         'dev_used': False, 'model_revision': 'abc', 'approved_parent_ids': ['p']}
    return r, cfg, m


def test_loss_mask_supervises_first_response_and_eos():
    assert supervision_range([1, 2], [3, 4, 5]) == (1, 4)
    logits_targets = [2, 3, 4, 5]
    begin, end = supervision_range([1, 2], [3, 4, 5])
    assert logits_targets[begin:end] == [3, 4, 5]


def test_provenance_manifest_context_and_revision_fail_closed():
    r, cfg, m = contracts()
    assert validate_training_pool([r], m, cfg)
    cfg['max_sequence_length'] = 4
    with pytest.raises(ValueError, match='truncation'):
        validate_training_pool([r], m, cfg)
    cfg['max_sequence_length'] = 8
    r['split'] = 'dev'
    m['dataset_hash'] = digest([r])
    with pytest.raises(ValueError, match='TRAIN'):
        validate_training_pool([r], m, cfg)
    r['split'] = 'train'
    m['dataset_hash'] = digest([r])
    m['approved_parent_ids'] = []
    with pytest.raises(ValueError, match='parent'):
        validate_training_pool([r], m, cfg)
