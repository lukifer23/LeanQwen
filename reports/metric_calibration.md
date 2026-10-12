# Local redundancy and conclusion calibration

Curated DEV pairs across five tasks, including four matched algebra policies; one assistant reviewer, no independent human labels or held-out calibration. Pair selection includes high lexical similarity and is not representative. Similarity does not establish uselessness. Not a training filter.

36 manually inspected pairs; default semantic threshold 0.90. Lexical @0.90: {'tp': 9, 'fp': 2, 'tn': 15, 'fn': 10, 'precision': 0.8181818181818182, 'recall': 0.47368421052631576}. Format-meta presence v1: {'tp': 7, 'fp': 4, 'tn': 1, 'fn': 0, 'precision': 0.6363636363636364, 'recall': 1.0}; derived anchored v2: {'tp': 7, 'fp': 0, 'tn': 5, 'fn': 0, 'precision': 1.0, 'recall': 1.0} over 12 inspected traces.

| Semantic threshold | TP | FP | TN | FN | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 15 | 5 | 12 | 4 | 0.75 | 0.79 |
| 0.85 | 14 | 4 | 13 | 5 | 0.78 | 0.74 |
| 0.9 | 11 | 1 | 16 | 8 | 0.92 | 0.58 |
| 0.95 | 6 | 1 | 16 | 13 | 0.86 | 0.32 |

Observed conclusion categories and source-linked calibration annotations are saved separately. Correct-number appearances, conflicting cue totals and forced-close recovery remain distinct concepts. Annotation text is DEV-only and forbidden for training.

MiniLM Apache-2.0 revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, CPU float32, one thread; vectors cached locally. Encoder chunks are subdivided to fit 256 wordpieces, never shortening the Qwen prompt/context. Raw generations are unchanged.

Reproduce: `uv sync --frozen --extra semantic` then `uv run --frozen --extra semantic python scripts/calibrate_metrics.py`. Run only between Qwen workloads.
