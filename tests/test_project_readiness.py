"""Readiness must retain failed/unknown gates without loading model frameworks."""

import json
import subprocess
import sys

import pytest

from qwenlean.training.watchdog import policy
from qwenlean.utils.io import digest, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock
from scripts import audit_project_readiness as readiness


def test_readiness_import_is_device_free():
    result = subprocess.run([sys.executable, '-c',
        'import sys; import scripts.audit_project_readiness; '
        'assert not any(n.split(".")[0] in {"mlx", "mlx_lm", "torch", "transformers"} '
        'for n in sys.modules)'], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_saved_admission_lists_both_failures_and_never_authorizes_loading(tmp_path):
    path = tmp_path/'admission.json'
    record = {'policy': policy({}), 'status': 'insufficient_starting_ram',
              'baseline': {'available_ram_bytes': 6*1024**3, 'swap_used_bytes': 3*1024**3}}
    write_json(path, record)
    checked = readiness.inspect_admission(path, {})
    assert checked['failed_gates'] == ['starting_available_ram', 'starting_occupied_swap']
    assert checked['authorizes_model_loading'] is False
    record.update(status='admitted', baseline={'available_ram_bytes': 10*1024**3,
                                              'swap_used_bytes': 0})
    write_json(path, record)
    checked = readiness.inspect_admission(path, {})
    assert checked['status'] == 'historical_admission_only'
    assert checked['authorizes_model_loading'] is False
    record['baseline']['available_ram_bytes'] = 0
    write_json(path, record)
    with pytest.raises(ValueError, match='contradicts'):
        readiness.inspect_admission(path, {})


def test_frozen_dev_rejects_locally_rehashed_cohort_and_train_overlap(tmp_path):
    cohort = tmp_path/'cohort'
    cohort.mkdir()
    tasks = [{'task_id': f'dev-{i}', 'split': 'dev',
              'semantic_structure_fingerprint': f'structure-{i}'} for i in range(56)]
    configs = {m: {'model': 'stock', 'revision': 'pin'} for m in ('thinking', 'nonthinking')}
    protocol = {'task_hash': digest(tasks), 'configs_hash': digest(configs),
                'configs': configs, 'task_count': 56, 'test_used': False}
    published = tmp_path/'published.json'
    write_json(cohort/'protocol.json', protocol)
    write_jsonl(cohort/'tasks.jsonl', tasks)
    write_json(published, {**protocol, 'task_ids': [t['task_id'] for t in tasks]})
    checked = readiness.inspect_frozen_dev(cohort, tmp_path/'baseline', published, [])
    assert checked['inventory']['completed_jobs'] == 0
    assert checked['inventory']['accuracy'] is None
    assert checked['pending_by_mode'] == {'thinking': 56, 'nonthinking': 56}
    for row in ({'task_id': 'dev-0', 'semantic_structure_fingerprint': 'unrelated'},
                {'task_id': 'train-0', 'semantic_structure_fingerprint': 'structure-0'}):
        with pytest.raises(ValueError, match='overlap'):
            readiness.inspect_frozen_dev(cohort, tmp_path/'baseline', published, [row])
    tasks[0]['task_id'] = 'changed'
    write_jsonl(cohort/'tasks.jsonl', tasks)
    write_json(cohort/'protocol.json', {**protocol, 'task_hash': digest(tasks)})
    with pytest.raises(ValueError, match='published'):
        readiness.inspect_frozen_dev(cohort, tmp_path/'baseline', published, [])


def test_partial_and_quarantined_updates_cannot_count_as_approved_smoke(tmp_path):
    attempt = tmp_path/'unsafe'
    attempt.mkdir()
    write_json(attempt/'completion.json', {'status': 'failed'})
    write_json(attempt/'quarantine.json', {})
    write_jsonl(attempt/'loss.jsonl', [{'optimizer_steps': step} for step in (1, 2, 3)])
    checked = readiness.inspect_adapters(tmp_path, {})
    assert checked['approved_adapter_count'] == checked['matching_four_step_smoke_count'] == 0
    assert checked['attempts'][0]['logged_optimizer_updates'] == [1, 2, 3]
    assert checked['attempts'][0]['quarantined'] is True
    write_json(attempt/'completion.json', {'status': 'complete'})
    with pytest.raises(ValueError, match='Quarantined'):
        readiness.inspect_adapters(tmp_path, {})


def test_missing_source_evidence_is_unknown_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, 'model_process_lock',
                        lambda: model_process_lock(tmp_path/'model.lock'))
    monkeypatch.setattr(readiness, 'POOLS', {'smoke': tmp_path/'missing-smoke',
                                           'pilot': tmp_path/'missing-pilot'})
    result = readiness.audit(tmp_path/'missing-admission.json')
    assert result['status'] == 'invalid_or_missing_evidence'
    assert result['checks']['frozen_dev']['status'] == 'unknown'
    assert not result['gates']['source_and_complete_records']
    assert not result['gates']['stock_baseline_complete']
    assert result['model_execution_authorized'] is result['improvement_sft_authorized'] is False


def test_complete_adapter_with_fewer_updates_is_not_a_four_step_smoke(tmp_path, monkeypatch):
    attempt = tmp_path/'complete-three-step'
    attempt.mkdir()
    config = {'steps': 4}
    write_json(attempt/'config.json', config)
    write_json(attempt/'completion.json', {'status': 'complete', 'optimizer_steps': 3})
    write_jsonl(attempt/'loss.jsonl', [{'optimizer_steps': i} for i in (1, 2, 3)])
    monkeypatch.setattr(readiness, 'require_approved_adapter', lambda path, cfg: {})
    checked = readiness.inspect_adapters(tmp_path, config)
    assert checked['approved_adapter_count'] == 1
    assert checked['matching_four_step_smoke_count'] == 0


def test_existing_readiness_evidence_cannot_be_overwritten(tmp_path):
    path = tmp_path/'existing.json'
    original = json.dumps({'preserved': True})
    path.write_text(original)
    result = subprocess.run([sys.executable, 'scripts/audit_project_readiness.py',
        '--admission', str(tmp_path/'missing.json'), '--output', str(path)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and 'Refuse existing' in result.stderr
    assert path.read_text() == original
