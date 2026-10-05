"""Exact own-prefix counterfactual closure, separate from natural generation."""

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from qwenlean.inference.parsing import parse_tokens
from qwenlean.scoring.exact import score
from qwenlean.utils.environment import environment_manifest
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock


def probe_plan(rows, policy, settings):
    counts, selected = Counter(), []
    for r in rows:
        if r['prompt_policy'] == policy and r['replicate'] == 0 and not counts[r['family']]:
            selected.append(r)
            counts[r['family']] += 1
    if len(selected) != 6 or any(r['split'] != 'dev' for r in selected):
        raise ValueError('Expected six balanced DEV source trajectories')
    jobs = []
    for r in selected:
        cutoffs = {n: 'fixed_token_checkpoint' for n in settings['cutoffs'] if n <= r['reasoning_tokens']}
        candidate = r['answer_distance']['candidate_correct_conclusion_token']
        if candidate is not None and candidate <= r['reasoning_tokens']:
            cutoffs[candidate] = 'candidate_correct_conclusion_heuristic'
        for cutoff, reason in sorted(cutoffs.items()):
            jobs.append((r, cutoff, reason))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', default='runs/phase2-early-exit-controller')
    args = p.parse_args()
    study = json.loads(Path('reports/prompt_policy_measurements.json').read_text())
    protocol = json.loads(Path('reports/phase2_protocol.json').read_text())
    sources = read_jsonl('reports/prompt_policy_records.jsonl')
    plan = probe_plan(sources, study['selected_policy'], protocol['early_exit'])
    cfg = {**study['config'], 'name': 'phase2-counterfactual-early-exit', 'max_output_tokens': 128,
           'enable_thinking': False, 'sampling': {'temperature': 0, 'top_p': 1, 'top_k': 0,
                                               'min_p': 0, 'presence_penalty': 0, 'repetition_penalty': 1},
           'loop_guard': {'enabled': False}}
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    contract = {'source_hash': digest(sources), 'config_hash': digest(cfg),
                'plan': [(r['generation_id'], n, reason) for r, n, reason in plan]}
    if (state / 'contract.json').exists():
        if json.loads((state / 'contract.json').read_text()) != contract:
            raise ValueError('Early-exit resume contract differs')
    else:
        write_json(state / 'contract.json', contract)
        write_json(state / 'environment.json', environment_manifest())
    saved = read_jsonl(state / 'probes.jsonl') if (state / 'probes.jsonl').exists() else []
    if [(r['source_generation_id'], r['cutoff_token'], r['cutoff_reason']) for r in saved] != [
        (r['generation_id'], n, reason) for r, n, reason in plan[:len(saved)]
    ]:
        raise ValueError('Probes must be exact unique ordered prefix')
    from qwenlean.inference.mlx_backend import MLXBackend

    with model_process_lock():
        backend = MLXBackend(cfg)
        close = backend.tokenizer.encode('</think>\n\n', add_special_tokens=False)
        if not close or close[0] != backend.tokenizer.convert_tokens_to_ids('</think>'):
            raise ValueError('Counterfactual close is not an atomic reasoning control token')
        with (state / 'probes.jsonl').open('a') as handle:
            for r, n, reason in plan[len(saved):]:
                parsed = parse_tokens(r['token_ids'], backend.tokenizer, True)
                # Reuse prompt and emitted IDs verbatim: never retokenize the prefix.
                prefix = r['prompt_token_ids'] + parsed.reasoning_ids[:n] + close
                g = backend.generate_tokens(prefix, r['seed'], cfg, thinking=False)
                p = parse_tokens(g['token_ids'], backend.tokenizer, False)
                fields = asdict(p)
                fields.pop('reasoning_ids')
                row = {**g, **fields, **score(p.final, r['expected'], r['prompt_policy']),
                       'probe_id': f"{r['generation_id']}:counterfactual-{n}",
                       'source_generation_id': r['generation_id'], 'task_id': r['task_id'],
                       'cutoff_token': n, 'cutoff_reason': reason, 'expected': r['expected'],
                       'source_correct': r['correct'], 'source_eos': r['termination_reason'] == 'eos',
                       'source_reasoning_tokens': r['reasoning_tokens'],
                       'observed_generated_tokens_saved': r['total_output_tokens'] - n - p.total_output_tokens,
                       'natural_completion_savings_known': r['termination_reason'] == 'eos',
                       'generation_parameters': cfg, 'counterfactual': True,
                       'provenance': {'source_id': f"{r['generation_id']}:probe-{n}",
                                      'response_origin': 'qwen_self', 'prompt_origin': 'algorithmic_transform',
                                      'parent_ids': [r['generation_id']], 'license': 'Apache-2.0',
                                      'training_permitted': False}, 'split': 'dev'}
                handle.write(json.dumps(row) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                saved.append(row)
                print(f"Early-exit {len(saved)}/{len(plan)} {r['family']} at {n}: correct={row['correct']} stop={g['termination_reason']}", flush=True)
                if g['termination_reason'] not in {'eos', 'max_output_tokens'}:
                    raise RuntimeError('Explicit probe generation failure retained')
    minima = {}
    for row in saved:
        if row['correct']:
            minima[row['source_generation_id']] = min(row['cutoff_token'], minima.get(row['source_generation_id'], row['cutoff_token']))
    result = {'probes': len(saved), 'source_trajectories': 6, 'correct_probes': sum(r['correct'] for r in saved),
              'minimum_sufficient_observed_prefix': minima, 'protocol': protocol['early_exit'],
              'policy': study['selected_policy'], 'test_evaluated': False,
              'interpretation': 'Conditional forced-close recovery, not natural stopping, certainty, or population accuracy. Censored source savings are relative to observed cap only.'}
    write_json('reports/early_exit_measurements.json', result)
    write_jsonl('reports/early_exit_records.jsonl', saved)
    lines = ['# Counterfactual early-exit probes', '', result['interpretation'], '',
             f"{result['correct_probes']}/{len(saved)} probes recovered the correct integer across six DEV sources. {len(minima)}/6 sources had at least one successful tested prefix.", '',
             '| Source task | Earliest successful tested prefix | Original correct | Original EOS |',
             '|---|---:|---:|---:|']
    for r, _, _ in plan:
        if any(r['task_id'] in line for line in lines):
            continue
        lines.append(f"| {r['task_id']} | {minima.get(r['generation_id'], 'none')} | {r['correct']} | {r['termination_reason'] == 'eos'} |")
    lines += ['', 'Uses actual saved chat-template prompt IDs plus generated reasoning IDs, then explicitly forces `</think>` and two newlines. Greedy final generation has 128 tokens and no repetition/presence penalties. Original baseline records are untouched. Counts are nested within six tasks, not independent benchmark problems.', '',
              'Reproduce/resume: `uv run --frozen python scripts/early_exit_probe.py`.']
    Path('reports/early_exit_probe.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
