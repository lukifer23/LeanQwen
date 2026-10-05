"""Predeclared, sequential staged-cap diagnostic. Resumable per task/cap."""

import argparse
import json
from pathlib import Path

from qwenlean.datasets.prompts import render_task
from qwenlean.evaluation.runner import evaluate
from qwenlean.evaluation.tail import check_prefix, survival_bounds
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', default='runs/phase2-tail-controller')
    args = p.parse_args()
    protocol = json.loads(Path('reports/phase2_protocol.json').read_text())
    study = json.loads(Path('reports/prompt_policy_measurements.json').read_text())
    policy = study['selected_policy']
    cfg = {**study['config'], 'name': 'phase2-termination-tail', 'replicates': 1,
           'prompt_policies': [policy], 'prompt_policy': policy}
    tasks = {t['sample_id']: t for t in read_jsonl('data/splits/dev.jsonl')}
    initial = read_jsonl('reports/prompt_policy_records.jsonl')
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    contract = {'protocol_hash': digest(protocol), 'base_config_hash': digest(cfg), 'policy': policy}
    if (state / 'contract.json').exists():
        if json.loads((state / 'contract.json').read_text()) != contract:
            raise ValueError('Tail resume contract changed')
    else:
        write_json(state / 'contract.json', contract)
    records, prefixes = [], []
    from qwenlean.inference.mlx_backend import MLXBackend

    with model_process_lock():
        backend = MLXBackend(cfg)
        for case in protocol['tail']['cohort']:
            task = render_task(tasks[case['historical_sample_id']], policy)
            prior = None
            caps = protocol['tail']['caps'][:]
            if case['historical_sample_id'] in protocol['tail']['cap_16384_eligible_ids']:
                caps.append(16384)
            for cap in caps:
                cached = state / f"{task['task_id']}-{cap}.json"
                reused = [r for r in initial if cap == 2048 and r['task_id'] == task['task_id']
                          and r['prompt_policy'] == policy and r['replicate'] == 0]
                if cached.exists():
                    row = json.loads(cached.read_text())
                elif reused:
                    row = reused[0]
                    write_json(cached, row)
                else:
                    config = {**cfg, 'max_output_tokens': cap}
                    # Every generation gets its own durable harness run. If interrupted
                    # mid-token no row is saved, and that incomplete attempt remains visible.
                    run, _ = evaluate(backend, [task], config)
                    row = read_jsonl(run / 'samples.jsonl')[0]
                    write_json(cached, row)
                if row['seed'] != case['seed'] or row['prompt'] != task['prompt']:
                    raise ValueError('Tail cached seed/prompt mismatch')
                if prior:
                    check = check_prefix(prior, row)
                    prefixes.append(check)
                    if not check['exact_prefix']:
                        write_json(state / 'prefix_failure.json', check)
                        raise RuntimeError('Same-seed prefix diverged; escalation stopped')
                records.append({**row, 'diagnostic_stratum': case['stratum']})
                prior = row
                write_jsonl(state / 'records.jsonl', records)
                if row['termination_reason'] == 'eos':
                    break
    summary = {**survival_bounds(records), 'policy': policy, 'prefix_checks': prefixes,
               'selection': protocol['tail'], 'test_evaluated': False}
    write_json('reports/termination_tail_measurements.json', summary)
    write_jsonl('reports/termination_tail_records.jsonl', records)
    lines = ['# Termination tail', '', f'Seven predeclared DEV cases; policy {policy}, original weights, official thinking, guard off, matched replicate-zero seeds. Higher caps are skipped after EOS.', '',
             '| Horizon | EOS by horizon | Still generating | Unknown beyond earlier censoring | Survival bounds |',
             '|---:|---:|---:|---:|---:|']
    for h in summary['horizons']:
        lines.append(f"| {h['token_horizon']} | {h['observed_eos_by_horizon']} | {h['known_still_generating']} | {h['censored_before_horizon']} | {h['survival_lower_bound']:.1%}–{h['survival_upper_bound']:.1%} |")
    lines += ['', f"Right-censored latest observations: {summary['right_censored']}/7. Exact prefix checks: {len(prefixes)}, all passed. Observed natural reasoning lengths: {summary['natural_completed_reasoning_lengths']}.", '',
              'This selected cohort cannot estimate population nontermination. Cap lengths are censored observations, not natural lengths. Unknown outcomes at 16K remain unknown for cases whose predeclared escalation ended at 8K. Survival horizons use total generated tokens; completed reasoning lengths exclude controls/final text.', '',
              'Reproduce/resume: `uv run --frozen python scripts/termination_tail.py`. Raw records and prefix comparisons accompany this report.']
    Path('reports/termination_tail.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
