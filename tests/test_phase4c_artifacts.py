"""CPU-only checks against actual preserved failed-run files; never model inference."""

import json
from pathlib import Path

import pytest

from qwenlean.evaluation.single_job import inspect_single
from qwenlean.training.checkpoint_audit import audit_lora_file, require_approved_adapter
from qwenlean.utils.io import read_jsonl


def test_actual_unsafe_checkpoint_matches_completed_step_two_and_remains_unapproved():
    root = Path('adapters/external-human-smoke-bounded-prefix-attempt1')
    if not root.exists():
        pytest.skip('Historical unsafe files are local and never downloaded in CI')
    checked = audit_lora_file(root/'step-000002.safetensors')
    logs = read_jsonl(root/'loss.jsonl')
    assert [r['optimizer_steps'] for r in logs] == [1, 2, 3]
    assert checked['tensor_fingerprint'] == logs[1]['adapter_hash']
    assert checked['parameters'] == 55296 and checked['tensor_count'] == 6
    assert checked['all_parameters_finite']
    with pytest.raises(ValueError, match='Quarantined'):
        require_approved_adapter(root, {'model': 'Qwen/Qwen3.5-0.8B', 'revision': 'unused'})


def test_saved_stock_failure_is_excluded_but_complete_eos_is_recoverable():
    cohort = Path('data/generated/phase4-v1.1-frozen-dev')
    if not cohort.exists():
        pytest.skip('Historical local diagnostic files unavailable')
    tasks = read_jsonl(cohort/'tasks.jsonl')
    cfg = json.loads((cohort/'protocol.json').read_text())['configs']['nonthinking']
    root = Path('runs/phase4-v1.1-stock-dev/nonthinking')
    complete = inspect_single(tasks[0], cfg, root/'000')
    failed = inspect_single(tasks[1], cfg, root/'001')
    assert complete['status'] == 'complete' and complete['record']['termination_reason'] == 'eos'
    assert failed['status'] == 'failed' and failed['record'] is None
    assert failed['failed_attempts'][0]['reason'] == 'stream_error'
    with pytest.raises(ValueError, match='contract changed'):
        inspect_single(tasks[0], {**cfg, 'max_output_tokens': 512}, root/'000')
