# Local training updates and system-pressure stop

One real AdamW update completed on an intact external human target: finite loss
1.066536665, finite gradients,1.215982 seconds,171 supervised/281 complete tokens.
The remaining three steps did not execute; the next backward hit the unchanged
RAM floor. No approved adapter, full base-integrity/reload result or SFT improvement
exists. See [measured attempt](external-human-smoke-memory-efficient-attempt1.json).

Response-only checkpointed vocabulary loss and bounded reusable cache are
implemented. Float32/bf16 head-loss and hidden-gradient comparisons plus actual
tiny Qwen LoRA-gradient/masking tests pass. Prompt tokens remain masked; first
response token, all complete reasoning/final tokens and EOS remain supervised.
No weights, native context, RoPE, tokenizer or source targets changed.

An additional frozen-prefix evaluation experiment was unsafe on the GPU despite
CPU equivalence tests passing. macOS sample reported **26.9G peak physical
footprint** (its display units, not an MLX allocator byte count). It stalled inside
MLX evaluation before any optimizer update. Only the verified owned PID35811
received SIGINT; it exited with saved KeyboardInterrupt failure. Subsequent SIGTERM
found it already gone. Raw console/sample and exact source/config/base metadata are
preserved in the [interruption report](external-human-smoke-bounded-prefix-interruption.json).
The frozen-prefix path has been removed and its option rejected before model loading.

The RAM check after backward was insufficient for pressure developing inside
native evaluation. Occupied swap also increased sharply while the swap-out counter
showed a much smaller change. Following the user's hardware constraint, no further
GPU model job was launched. Added an independent **CPU-only watchdog** that monitors
only its verified parent every100 ms, including during native calls. Default smoke
limits:8 GiB starting available RAM, at most2 GiB existing swap occupancy,4 GiB
running reserve,128 MiB swap-occupancy growth,512 MiB swap-out delta,3 GiB process
RSS,15-second gradient and120-second total timeout. It records evidence, sends
SIGINT to its own parent and uses SIGTERM after a one-second grace period if needed.
Startup must be confirmed before model loading. No unrelated process is targeted.

MLX's3 GiB memory setting is a guideline rather than a hard allocation guarantee;
the independent monitor is essential. Its timeout/ownership/admission behavior is
verified using CPU fixtures, not another GPU stress run. Current CPU-only admission
refused before any model loading at8,074,756,096 bytes available versus the8 GiB
startup requirement, with4,583,915,520 occupied swap bytes. Exact snapshot and hashes
are in [the safety summary](local-training-resource-safety.json).

All earlier raw runs, failure reports and the first update evidence remain intact.
Future technical smoke requires a fresh safe admission and the supervised profile;
no unsafe-prefix rerun or SFT improvement pilot is authorized by these CPU checks.
