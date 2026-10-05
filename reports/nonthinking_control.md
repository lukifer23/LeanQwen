# Official default-mode control

| Variant | Accuracy | Median Think Tokens | P95 Think Tokens | Exact Output Loop Rate | Max-Out Rate | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| A thinking | 0.0% | 2048 | 2048 | 0.0% | 100.0% | 33.90 s |
| A0 non-thinking | 31.7% | 0 | 0 | 0.0% | 28.3% | 12.39 s |

Non-thinking uses the official non-thinking sampler. This is a mode plus sampler control. A zero reasoning partition does not mean zero compute: median total output tokens are 2048 for A and 382 for A0. This control does not establish that harder reasoning tasks can dispense with thinking.

Task accuracy 31.7%; strict extraction accuracy 25.0%; format compliance 45.0%. Scorer: terminal_answers_v4.
