"""Small natural N=4 TRAIN pool; refuse generation before DEV diagnostic gates."""

import argparse
import json
from collections import Counter
from pathlib import Path

from qwenlean.datasets.provenance import validate_provenance
from qwenlean.generation.selection import select_natural
from qwenlean.utils.io import read_jsonl, write_json, write_jsonl


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
    p.add_argument('--output', help='New directory for historical CPU reanalysis only')
    args = p.parse_args()
    if args.resume or not args.publish_run or not args.output:
        raise ValueError('New Qwen teacher generation/resume is prohibited. Historical reanalysis requires --publish-run and a new --output directory.')
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('Historical evidence cannot be overwritten')
    for file in ('prompt_policy_ablation.md', 'termination_tail.md', 'early_exit_probe.md',
                 'metric_calibration.md', 'runtime_guard_calibration.md'):
        if not (Path('reports') / file).exists():
            raise ValueError(f'Diagnostic gate pending: {file}')
    study = json.loads(Path('reports/prompt_policy_measurements.json').read_text())
    tail = json.loads(Path('reports/termination_tail_measurements.json').read_text())
    early = json.loads(Path('reports/early_exit_measurements.json').read_text())
    calibration = json.loads(Path('reports/metric_calibration.json').read_text())
    guard = json.loads(Path('reports/runtime_guard_replay.json').read_text())
    if guard.get('full_diagnostic_gate_complete') is not True:
        raise ValueError('Full prompt-and-tail guard replay required; partial prompt-only replay is not a gate')
    if study['trajectories'] != 144 or tail['selected_trajectories'] != 7 or early['source_trajectories'] != 6:
        raise ValueError('Incomplete predeclared DEV diagnostic design')
    if any(not c['exact_prefix'] for c in tail['prefix_checks']) or calibration['calibration_pair_count'] < 24:
        raise ValueError('Prefix/metric calibration gate failed')
    if any(t.get('manual_review') == 'pending' for g in guard['configurations'] for t in g['triggers']):
        raise ValueError('Every offline guard trigger needs individual review')
    cfg = {**study['config'], 'name': 'phase2-train-natural-n4', 'replicates': 4,
           'prompt_policy': study['selected_policy'], 'prompt_policies': [study['selected_policy']],
           'max_output_tokens': 4096}
    run = Path(args.publish_run)
    if json.loads((run / 'completion.json').read_text())['status'] != 'complete':
        raise ValueError('Incomplete historical candidate archive')
    rows = read_jsonl(run / 'samples.jsonl')
    enriched, selected, report = select_natural(rows)
    report.update({'source_run': str(run), 'replicates': 4, 'output_cap': 4096,
                   'historical_generation_config': cfg,
                   'policy': study['selected_policy'], 'family_counts': dict(Counter(r['family'] for r in selected)),
                   'selected_parent_ids': [r['generation_id'] for r in selected], 'test_used': False,
                   'dataset_manifest': 'data/splits/procedural-v2/manifest.json'})
    report.update(training_permitted=False, historical_reproduction_only=True)
    destination.mkdir(parents=True)
    write_jsonl(destination/'train-candidates.jsonl', enriched)
    write_jsonl(destination/'natural-selected-review.jsonl', selected)
    write_json(destination/'quality.json', report)
    # Public raw TRAIN evidence contains Qwen text only; code-authored summaries
    # and reviewer prose must never become response training targets.
    write_jsonl(destination/'records.jsonl', enriched)
    lines = ['# Natural best-of-N TRAIN pilot', '',
             f"{len(rows)} original-Qwen trajectories, sixteen TRAIN tasks, N=4, policy {study['selected_policy']}, 4096 total-output cap, guard off. No compression, DEV/TEST responses, or proprietary teacher responses.", '',
             f"Tasks with any correct final: {report['tasks_with_correct_candidate']}/16. Clean eligible natural target coverage: {len(selected)}/16. Tier counts: {report['tier_counts']}.", '',
             f"Selected reasoning lengths: {report['selected_reasoning_tokens']}. Per-task median candidate minus selected tokens: {report['median_candidate_minus_selected_tokens']}.", '',
             'Quality precedes length. Eligibility requires correct final, clean EOS/partition, no exact loop and no detected invalid numeric claim. Recognized intermediate claims are narrowly checked; absence of a detected error is not proof of reasoning validity. Selected targets remain a review pool until individual examples are approved in a hashed quality manifest.', '',
             'Historical CPU reanalysis only. Outputs cannot authorize current SFT; new teacher generation/resume is prohibited.']
    (destination/'quality.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
