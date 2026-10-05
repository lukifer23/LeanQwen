"""Small natural N=4 TRAIN pool; refuse generation before DEV diagnostic gates."""

import argparse
import json
from collections import Counter
from pathlib import Path

from qwenlean.datasets.prompts import render_task
from qwenlean.datasets.provenance import validate_provenance
from qwenlean.evaluation.runner import evaluate, validate_resume
from qwenlean.generation.selection import select_natural
from qwenlean.utils.io import read_jsonl, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock


def selected_tasks():
    rows = read_jsonl('data/splits/procedural-v2/train.jsonl')
    # Predeclared bounded pilot: one task per family from trivial and medium bins.
    chosen, seen = [], set()
    for r in rows:
        key = (r['family'], r['difficulty'])
        if r['difficulty'] in {'trivial', 'medium'} and key not in seen:
            validate_provenance(r, for_training=True)
            chosen.append(r)
            seen.add(key)
    if len(chosen) != 16:
        raise ValueError('Expected sixteen TRAIN tasks, eight families × two difficulty bins')
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume')
    p.add_argument('--publish-run')
    args = p.parse_args()
    for file in ('prompt_policy_ablation.md', 'termination_tail.md', 'early_exit_probe.md',
                 'metric_calibration.md', 'runtime_guard_calibration.md'):
        if not (Path('reports') / file).exists():
            raise ValueError(f'Diagnostic gate pending: {file}')
    study = json.loads(Path('reports/prompt_policy_measurements.json').read_text())
    cfg = {**study['config'], 'name': 'phase2-train-natural-n4', 'replicates': 4,
           'prompt_policy': study['selected_policy'], 'prompt_policies': [study['selected_policy']],
           'max_output_tokens': 4096}
    tasks = [render_task(t, study['selected_policy']) for t in selected_tasks()]
    if args.publish_run:
        run = Path(args.publish_run)
        if json.loads((run / 'completion.json').read_text())['status'] != 'complete':
            raise ValueError('Incomplete candidate pool')
    else:
        if args.resume:
            validate_resume(args.resume, tasks, cfg)
        from qwenlean.inference.mlx_backend import MLXBackend

        with model_process_lock():
            backend = MLXBackend(cfg)
            run, _ = evaluate(backend, tasks, cfg, resume=args.resume)
    rows = read_jsonl(run / 'samples.jsonl')
    enriched, selected, report = select_natural(rows)
    report.update({'source_run': str(run), 'replicates': 4, 'output_cap': 4096,
                   'policy': study['selected_policy'], 'family_counts': dict(Counter(r['family'] for r in selected)),
                   'selected_parent_ids': [r['generation_id'] for r in selected], 'test_used': False,
                   'dataset_manifest': 'data/splits/procedural-v2/manifest.json'})
    write_jsonl('data/generated/train-candidates.jsonl', enriched)
    write_jsonl('data/processed/natural-selected-review.jsonl', selected)
    write_json('reports/train_candidate_quality.json', report)
    # Public raw TRAIN evidence contains Qwen text only; code-authored summaries
    # and reviewer prose must never become response training targets.
    write_jsonl('reports/train_candidate_records.jsonl', enriched)
    lines = ['# Natural best-of-N TRAIN pilot', '',
             f"{len(rows)} original-Qwen trajectories, sixteen TRAIN tasks, N=4, policy {study['selected_policy']}, 4096 total-output cap, guard off. No compression, DEV/TEST responses, or proprietary teacher responses.", '',
             f"Tasks with any correct final: {report['tasks_with_correct_candidate']}/16. Clean eligible natural target coverage: {len(selected)}/16. Tier counts: {report['tier_counts']}.", '',
             f"Selected reasoning lengths: {report['selected_reasoning_tokens']}. Per-task median candidate minus selected tokens: {report['median_candidate_minus_selected_tokens']}.", '',
             'Quality precedes length. Eligibility requires correct final, clean EOS/partition, no exact loop and no detected invalid numeric claim. Recognized intermediate claims are narrowly checked; absence of a detected error is not proof of reasoning validity. Selected targets remain a review pool until individual examples are approved in a hashed quality manifest.', '',
             'Reproduce: `uv run --frozen python scripts/train_candidates.py`. Resume with `--resume runs/<incomplete-run>`.']
    Path('reports/train_candidate_quality.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
