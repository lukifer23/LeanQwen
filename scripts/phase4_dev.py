"""Frozen matched stock/adapter DEV cohort. One bounded workload under the lease."""

import argparse
import json
from pathlib import Path

import yaml

from qwenlean.datasets.prompts import render_task
from qwenlean.evaluation.single_job import evaluate_single
from qwenlean.evaluation.summary import summarize
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl


def freeze(corpus, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Refuse to change frozen cohort')
    tasks = []
    for name in ('dev-id', 'dev-ood'):
        rows = read_jsonl(Path(corpus)/f'{name}.jsonl')
        for i in range(0, len(rows), 4):
            tasks.extend([render_task(rows[i], 'P2'), render_task(rows[i+3], 'P2')])
    base = yaml.safe_load(Path('configs/baseline.yaml').read_text())
    base.update(seed=20261012, max_output_tokens=2048, context_profile='mac-128k',
                prefill_step_size=512, prompt_policy='P2', replicates=1,
                min_available_ram_bytes=4*1024**3, loop_guard={'enabled': False})
    configs = {}
    for thinking in (True, False):
        name = 'thinking' if thinking else 'nonthinking'
        config = {**base, 'name': 'phase4-matched-'+name, 'enable_thinking': thinking}
        if not thinking:
            config['sampling'] = yaml.safe_load(Path('configs/nonthinking.yaml').read_text())['sampling']
        configs[name] = config
    destination.mkdir(parents=True)
    write_jsonl(destination/'tasks.jsonl', tasks)
    write_json(destination/'protocol.json', {'status': 'frozen_before_training', 'task_hash': digest(tasks),
        'task_count': len(tasks), 'configs': configs, 'configs_hash': digest(configs),
        'corpus_manifest_hash': digest(json.loads((Path(corpus)/'manifest.json').read_text())),
        'one_seed_limitation': 'Task-cluster paired comparisons; one replicate cannot estimate sampler variance.',
        'test_used': False, 'selection_rule': 'First and last structural complexity within each family/holdout; no model outcomes used'})


def run(cohort, output, adapter=None):
    # Import GPU libraries only for actual inference.
    from qwenlean.inference.mlx_backend import MLXBackend
    from qwenlean.utils.process_lock import model_process_lock
    cohort, output = Path(cohort), Path(output)
    protocol = json.loads((cohort/'protocol.json').read_text())
    tasks = read_jsonl(cohort/'tasks.jsonl')
    if protocol['task_hash'] != digest(tasks) or protocol['configs_hash'] != digest(protocol['configs']):
        raise ValueError('Frozen protocol mismatch')
    output.mkdir(parents=True, exist_ok=True)
    run_contract = {'protocol_hash': digest(protocol), 'adapter': adapter}
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text()) != run_contract:
        raise ValueError('Evaluation contract changed')
    write_json(output/'contract.json', run_contract)
    rows = []
    with model_process_lock():
        initial = dict(protocol['configs']['thinking'])
        if adapter:
            initial['adapter_path'] = adapter
        with MLXBackend(initial) as backend:
            for mode, cfg in protocol['configs'].items():
                config = dict(cfg)
                if adapter:
                    config['adapter_path'] = adapter
                mode_rows = []
                for i, task in enumerate(tasks):
                    row = evaluate_single(backend, task, config, output/mode/f'{i:03d}')
                    row['phase4_mode'] = mode
                    rows.append(row)
                    mode_rows.append(row)
                    write_jsonl(output/'records.jsonl', rows)
                    write_json(output/'progress.json', {'saved': len(rows), 'planned': 2*len(tasks)})
                write_json(output/f'{mode}-summary.json', summarize(mode_rows))
    report = {'status': 'complete', 'contract': run_contract, 'records_hash': digest(rows),
              'modes': {m: summarize([r for r in rows if r['phase4_mode'] == m]) for m in protocol['configs']}}
    write_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corpus')
    p.add_argument('--cohort', required=True)
    p.add_argument('--output')
    p.add_argument('--adapter')
    p.add_argument('--freeze', action='store_true')
    a = p.parse_args()
    if a.freeze:
        freeze(a.corpus, a.cohort)
    else:
        run(a.cohort, a.output, a.adapter)
