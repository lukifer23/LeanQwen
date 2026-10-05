"""Derived local semantic measurements after the TRAIN generator fully exits."""

from qwenlean.metrics.semantic import LocalSemanticMetric
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock


def main():
    rows = read_jsonl('reports/train_candidate_records.jsonl')
    if any(r['split'] != 'train' or r['provenance']['response_origin'] != 'qwen_self' for r in rows):
        raise ValueError('Measured Qwen TRAIN archive required')
    results = []
    with model_process_lock(), LocalSemanticMetric() as metric:
        for i, r in enumerate(rows):
            results.append({'source_generation_id':r['generation_id'], 'task_id':r['task_id'],
                            'semantic':metric.measure(r['reasoning']), 'training_permitted':False,
                            'scope':'Measurement only; no embedding vectors or reviewer text used as supervised response'})
            if (i+1)%8 == 0:
                print(f'TRAIN semantic measurement {i+1}/{len(rows)}',flush=True)
    write_jsonl('reports/train_candidate_semantics.jsonl',results)
    write_json('reports/train_candidate_semantic_manifest.json', {
        'source_records_hash':digest(rows),'measurements_hash':digest(results),
        'source_count':len(rows),'ranking_use':'Inspect alongside natural target review; calibrated metric is too weak for an automatic training filter.'})


if __name__=='__main__':
    main()
