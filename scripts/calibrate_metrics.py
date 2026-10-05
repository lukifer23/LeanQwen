"""Offline DEV metric analysis with a local model under the same exclusive lease."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from qwenlean.inference.parsing import parse_tokens
from qwenlean.metrics.conclusions import conclusion_states
from qwenlean.metrics.guard_replay import replay
from qwenlean.metrics.repetition import STOP_WORDS, words
from qwenlean.metrics.semantic import LocalSemanticMetric
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock


def confusion(labels, predictions):
    c = Counter(('tp' if p else 'fn') if y else ('fp' if p else 'tn')
                for y, p in zip(labels, predictions))
    return {**{k: c[k] for k in ('tp', 'fp', 'tn', 'fn')},
            'precision': c['tp']/max(1, c['tp']+c['fp']),
            'recall': c['tp']/max(1, c['tp']+c['fn'])}


def lexical_pair(a, b):
    va, vb = [Counter(w for w in words(t) if w not in STOP_WORDS) for t in (a,b)]
    d = math.sqrt(sum(v*v for v in va.values()) * sum(v*v for v in vb.values()))
    return sum(v*vb[k] for k,v in va.items())/d if d else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--guard-only', action='store_true')
    args = p.parse_args()
    study = json.loads(Path('reports/prompt_policy_measurements.json').read_text())
    rows = read_jsonl('reports/prompt_policy_records.jsonl') + read_jsonl('reports/termination_tail_records.jsonl')
    unique = {r['generation_id']: r for r in rows}
    rows = list(unique.values())
    pairs = read_jsonl('reports/redundancy_calibration_pairs.jsonl')
    annotations = read_jsonl('reports/conclusion_calibration_annotations.jsonl')
    sources = {r['generation_id']:r for r in rows}
    if any(a['source_generation_id'] not in sources for a in annotations+pairs):
        raise ValueError('Annotation references unknown generation')
    derived = [{ 'source_generation_id':r['generation_id'], 'task_id':r['task_id'], 'policy':r['prompt_policy'],
                 'conclusion_states': conclusion_states(r['reasoning'], r['expected'], r['correct'], r['termination_reason'])}
               for r in rows]
    write_jsonl('reports/conclusion_state_measurements.jsonl', derived)
    meta_labels = [a['format_meta_present'] for a in annotations]
    meta_pred = [sources[a['source_generation_id']]['format_meta']['flagged_lines'] > 0 for a in annotations]
    fmt = confusion(meta_labels, meta_pred)
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from transformers import AutoTokenizer

    tokenizer = TokenizerWrapper(AutoTokenizer.from_pretrained(study['config']['model'],
                                     revision=study['config']['revision'], local_files_only=True))
    tokenizer.add_eos_token('<|endoftext|>')
    configs = [{'min_period':24, 'max_period':128, 'repeats':3},
               {'min_period':16, 'max_period':256, 'repeats':3},
               {'min_period':48, 'max_period':512, 'repeats':2}]
    guard_results = []
    for config in configs:
        triggers = []
        for r in rows:
            ids = parse_tokens(r['token_ids'], tokenizer, True).reasoning_ids
            evidence = replay(ids, config)
            if evidence:
                n = evidence['trigger_reasoning_token']
                triggers.append({'source_generation_id':r['generation_id'], **evidence,
                                 'original_correct':r['correct'], 'original_eos':r['termination_reason']=='eos',
                                 'trigger_context':tokenizer.decode(ids[max(0,n-3*evidence['period_tokens']):n]),
                                 'continuation_after_trigger':tokenizer.decode(ids[n:]),
                                 'manual_review': 'pending'})
        guard_results.append({'thresholds':config, 'check_every':16,'replayed_trajectories':len(rows),
                              'triggers':triggers,'trigger_count':len(triggers),
                              'observed_saved_reasoning_tokens':sum(t['observed_reasoning_tokens_saved'] for t in triggers)})
    write_json('reports/runtime_guard_replay.json', {'configurations':guard_results,
               'source_hash':digest(rows), 'live_pilot_run':False, 'training_permitted':False})
    if not any(g['triggers'] for g in guard_results):
        Path('reports/runtime_guard_calibration.md').write_text(
            '# Runtime exact-cycle guard replay\n\n'
            f"Replayed {len(rows)} saved DEV trajectories at three threshold settings and the actual 16-token polling cadence. No configuration triggered. Estimated observed token savings: zero. Precision/recall are unidentified because there were no triggers and these traces are not independently cycle-labeled.\n\n"
            'No live guard pilot was justified. Guard stays off. Exact-cycle failure to detect paraphrased reconsideration is an expected limitation, not evidence of healthy reasoning. Details: runtime_guard_replay.json. Reproduce: `uv run --frozen python scripts/calibrate_metrics.py --guard-only`.\n')
    else:
        print('Guard triggers need individual review before report gate', flush=True)
    if args.guard_only:
        return
    with model_process_lock():
        metric = LocalSemanticMetric()
        alltexts = [t for pair in pairs for t in (pair['text_a'], pair['text_b'])]
        if metric.bound_chunks(alltexts) != alltexts:
            raise ValueError('Calibration units subdivide; recreate aligned pair annotations')
        vectors = metric.embeddings(alltexts)
        measured_pairs = []
        for i, pair in enumerate(pairs):
            measured_pairs.append({**pair, 'semantic_cosine':float(vectors[2*i]@vectors[2*i+1]),
                                  'lexical_cosine':lexical_pair(pair['text_a'],pair['text_b'])})
        labels = [p['repeated_claim_or_operation'] for p in measured_pairs]
        thresholds = (.80,.85,.90,.95)
        calibration = {str(t):confusion(labels,[p['semantic_cosine']>=t for p in measured_pairs]) for t in thresholds}
        lexical = confusion(labels,[p['lexical_cosine']>=.90 for p in measured_pairs])
        semantic_rows = []
        for i,r in enumerate(rows):
            semantic_rows.append({'source_generation_id':r['generation_id'],'task_id':r['task_id'],
                                  'policy':r['prompt_policy'], 'semantic':metric.measure(r['reasoning'])})
            if (i+1)%12 == 0:
                print(f'Semantic measurement {i+1}/{len(rows)}',flush=True)
    write_jsonl('reports/semantic_redundancy_records.jsonl',semantic_rows)
    semantic_lookup = {r['source_generation_id']: r['semantic'] for r in semantic_rows}
    prompt_rows = read_jsonl('reports/prompt_policy_records.jsonl')
    augmented = [{**r, 'semantic': semantic_lookup[r['generation_id']]} for r in prompt_rows]
    from qwenlean.evaluation.prompt_comparison import intervention_comparison

    enriched_pairs = {p: intervention_comparison(
        [r for r in augmented if r['prompt_policy'] == 'P3'],
        [r for r in augmented if r['prompt_policy'] == p]) for p in ('P0', 'P1', 'P2')}
    write_json('reports/prompt_policy_enriched_comparison.json', enriched_pairs)
    write_jsonl('reports/redundancy_calibration_measured_pairs.jsonl',measured_pairs)
    result={'calibration_pair_count':len(pairs),'independent_annotated_tasks':len({sources[p['source_generation_id']]['task_id'] for p in pairs}),
            'semantic_by_threshold':calibration,'lexical_threshold_0.90':lexical,
            'format_meta_presence_calibration':fmt,'manual_annotation_count':len(annotations),
            'by_policy_semantic_density': {p:float(np.mean([r['semantic']['density'] for r in semantic_rows if r['policy']==p])) for p in ('P0','P1','P2','P3')},
            'default_threshold':.90,'threshold_selection':'retain conservative preregistered 0.90; sweep characterizes sensitivity, no optimality claim',
            'model':metric.model.model_card_data.model_id if hasattr(metric.model, 'model_card_data') else 'all-MiniLM-L6-v2',
            'limitations':'Curated DEV pairs, mainly one algebra task, one assistant reviewer, no independent human labels or held-out calibration. Similarity does not establish uselessness. Not a training filter.'}
    # Avoid library-owned metadata object in serialized schema.
    result['model']='sentence-transformers/all-MiniLM-L6-v2'
    write_json('reports/metric_calibration.json',result)
    lines=['# Local redundancy and conclusion calibration','',result['limitations'],'',
           f"{len(pairs)} manually inspected pairs; default semantic threshold 0.90. Lexical @0.90: {lexical}. Format-meta presence: {fmt} over {len(annotations)} inspected traces.", '',
           '| Semantic threshold | TP | FP | TN | FN | Precision | Recall |','|---:|---:|---:|---:|---:|---:|---:|']
    for t,c in calibration.items():
        lines.append(f"| {t} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} | {c['precision']:.2f} | {c['recall']:.2f} |")
    lines += ['', 'Observed conclusion categories and source-linked calibration annotations are saved separately. Correct-number appearances, conflicting cue totals and forced-close recovery remain distinct concepts. Annotation text is DEV-only and forbidden for training.', '',
              'MiniLM Apache-2.0 revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, CPU float32, one thread; vectors cached locally. Encoder chunks are subdivided to fit 256 wordpieces, never shortening the Qwen prompt/context. Raw generations are unchanged.', '',
              'Reproduce: `uv sync --frozen --extra semantic` then `uv run --frozen --extra semantic python scripts/calibrate_metrics.py`. Run only between Qwen workloads.']
    Path('reports/metric_calibration.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
