# Phase 3 stock context and prefill measurements

Five deterministic questions with repeated distractor filler, one greedy decode each. Shared facts repeated across lengths/conditions are not independent tasks. Preliminary interactive timings, sequential chunk variants, no thermal control or statistical resource-saving estimate.

Original pinned bf16 Qwen3.5-0.8B, mac-128k profile, greedy non-thinking, full history, no KV quantization or eviction, 256 output tokens reserved. PonderScope exited naturally before these runs. No adapter or training data involved.

| Combined envelope | Actual input | Prefill chunk | Correct / 5 | Retrieval / 3 | EOS / 5 | MLX peak GB | Mean prefill s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 7936 | 512 | 4 | 3 | 4 | 2.473 | 2.772 |
| 32768 | 32512 | 512 | 4 | 3 | 5 | 2.960 | 16.381 |
| 2048 | 1792 | 2048 | 3 | 3 | 4 | 2.742 | 0.832 |
| 8192 | 7936 | 2048 | 4 | 3 | 4 | 3.016 | 2.705 |
| 32768 | 32512 | 2048 | 4 | 3 | 5 | 5.016 | 16.274 |

At 8K, 512 chunks reduced observed peak allocation by 18.0%; 4/5 emitted-ID sequences matched. The changed inventory trace was incorrect in both conditions. At 32K, observed peak allocation fell 41.0%; all five sequences matched, with mean prefill 16.274 versus 16.381 seconds. These are actual chunk-size measurements, not savings from a lower context ceiling.

All beginning/middle/end facts were retrieved at every tested length/condition. The distant sum failed at 2K (listed three values without adding), and succeeded at 8K/32K. The inventory calculation failed in every condition: 2K/8K traces misinterpreted returns and hit the cap; 32K returned 19 instead of 79 with EOS. Clean termination alone therefore does not establish capability.

KV cache reached 24/96/384 MiB in 2K/8K/32K envelopes; recurrent state stayed about 18.6 MiB. Raw chunk/decode memory observations, swap counters, full output IDs/text and archive hashes are in phase3_context_stock_records.jsonl. Exact large prompts/input IDs remain durably in the referenced ignored per-job archives and are reconstructible with the versioned suite.

Use 512 for paired long-context comparisons to limit prefill peak on this Mac. Keep historical prompt/tail/TRAIN sampler and prefill contracts unchanged. 100K, 128K and 262K-length behavior remains untested; short live profile restoration is separately verified in phase3_live_profile_restoration.json.

Reproduce: `uv run --frozen --no-sync python scripts/context_benchmark.py --lengths 2048 8192 32768 --state runs/context-stock-2048-prefill` and `uv run --frozen --no-sync python scripts/context_benchmark.py --lengths 8192 32768 --prefill-step-size 512 --state runs/context-stock-512-prefill`. Completed per-job records are reused, never regenerated.
