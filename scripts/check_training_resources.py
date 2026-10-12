"""CPU-only check of SFT resource admission; never loads a model or target tensors."""

import argparse
import json
from pathlib import Path

import yaml

from qwenlean.training.quality import validate_optimizer_config
from qwenlean.training.watchdog import training_watchdog
from qwenlean.utils.process_lock import model_process_lock


def check(config, output):
    validate_optimizer_config(config)
    output = Path(output)
    if output.exists():
        raise ValueError('Refuse existing resource-check evidence')
    with model_process_lock():
        output.mkdir(parents=True)
        try:
            with training_watchdog(config, output):
                pass  # CPU watchdog readiness only; no MLX/model/teacher calls.
        except RuntimeError:
            if not (output/'resource-admission.json').exists():
                raise
            admission = json.loads((output/'resource-admission.json').read_text())
            if admission['status'] == 'admitted':
                raise
            return admission
    return json.loads((output/'resource-admission.json').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = check(yaml.safe_load(Path(a.config).read_text()), a.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'admitted' else 2)
