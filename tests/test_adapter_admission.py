"""Weight-free rejection of failed, quarantined and foreign reload attempts."""

import subprocess
import sys

import pytest

from qwenlean.training.checkpoint_audit import (
    require_approved_adapter,
    require_owned_reload_checkpoint,
)


def test_pending_adapter_cannot_bypass_live_owned_watchdog(tmp_path):
    with pytest.raises(RuntimeError, match='Live owned CPU watchdog'):
        require_owned_reload_checkpoint(tmp_path, {}, 'unapproved')


def test_failed_or_partial_adapter_requires_complete_acceptance(tmp_path):
    (tmp_path/'completion.json').write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match='Complete approved'):
        require_approved_adapter(tmp_path, {})


def test_quarantine_rejected_by_internal_reload_and_offline_metadata_cli(tmp_path):
    (tmp_path/'quarantine.json').write_text('{}')
    with pytest.raises(ValueError, match='Quarantined'):
        require_owned_reload_checkpoint(tmp_path, {}, 'unapproved')
    result = subprocess.run([sys.executable, 'scripts/context_metadata.py',
        '--adapter-path', str(tmp_path), '--output', str(tmp_path/'never-published.json')],
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and 'Quarantined' in result.stderr
    assert not (tmp_path/'never-published.json').exists()
