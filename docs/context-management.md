# Reversible context management

Qwen3.5-0.8B's pinned original checkpoint has `max_position_embeddings=262144`.
LeanQwen never edits its weights, tokenizer, chat template, architecture or RoPE.
The runtime's combined input/output ceilings are independently selectable:

| Profile | Maximum combined tokens | Purpose |
|---|---:|---|
| `mac-128k` | 131072 | Default Mac request ceiling |
| `mac-100k` | 102400 | Explicitly authorized fallback |
| `native-262k` | 262144 | Original native request ceiling |

An operating profile configures admission, not demonstrated capability or an
allocation reservation. A short request does not save memory merely by selecting
a lower ceiling. No 100K, 128K or 262K inference feasibility is established yet.

## Request accounting

The official chat template is rendered first. The actual input token count includes
the complete conversation/prefix and template controls. Admission requires:

`actual_input_tokens + max_output_tokens <= selected_operational_context_limit`.

`max_output_tokens` is the reserved generation allowance, including reasoning,
final answer and sampled EOS. No prompt truncation is performed. Every new backend
generation saves a `context` record containing the native maximum, selected
maximum/profile, input length, reserved allowance, emitted length, effective total
sequence length and context configuration fingerprint. Positional metadata and
prefill/cache policy are included in that fingerprint.

Existing configuration files remain unchanged to preserve historical experiment
hashes. An omitted profile now selects `mac-128k`; historical requests were far
below every ceiling. Historical records remain immutable and carry their original
native-context metadata. Token-prefix consistency must still be checked empirically
when resuming diagnostics; the runtime profile alone does not establish it.

```bash
uv run --frozen qwenlean generate --config configs/baseline.yaml --context-profile mac-128k --prompt 'Compute 17 * 23.' --output runs/context-128k-short.json
uv run --frozen qwenlean generate --config configs/baseline.yaml --context-profile mac-100k --prompt 'Compute 17 * 23.' --output runs/context-100k-short.json
```

Both `eval` and `generate` accept `--context-profile` and `--prefill-step-size`.
YAML configurations can set the same keys; training uses its YAML profile.
Do not reuse an existing output file or change a resumable run's explicit options.

## Full 262K restoration

1. Ensure all other model work has finished. Every LeanQwen model command takes
   `model_process_lock`; that repository lease cannot detect unrelated applications.
2. Keep the pinned original `Qwen/Qwen3.5-0.8B` revision
   `2fc06364715b967f1860aea9cf38778875588b17` in the YAML configuration.
3. Select `native-262k`. If applying an adapter, set `adapter_path` in a separate
   YAML config, or use `context_benchmark.py --adapter-path`. Adapter metadata must
   match the original base model and revision. Switching profiles requires no
   retraining or checkpoint conversion.
4. Run the offline file/profile audit. It checks configuration, template, tokenizer
   and original weight-file SHA256 before and after selecting all three profiles.
   For an existing adapter, also supply `--adapter-path` to verify its files.

```bash
uv run --frozen --no-sync python scripts/context_metadata.py --verify-weight-files --output runs/native-restoration-file-audit.json
uv run --frozen --no-sync python scripts/profile_restore_probe.py --output runs/native-restoration-live-audit.json
uv run --frozen qwenlean generate --config configs/baseline.yaml --context-profile native-262k --prompt 'Compute 17 * 23.' --output runs/native-restoration-short.json
uv run --frozen --no-sync python scripts/context_benchmark.py --context-profile native-262k --lengths 2048 8192 32768 --state runs/native-progressive-small
```

5. Inspect generation records: full-history caches, no rotating/sliding eviction,
   no cache quantization, actual input and reserved output within the ceiling.
   The wrapper rejects `max_kv_size`, `kv_bits` and externally supplied caches.
6. On hardware justified by the smaller probes, continue progressively:

```bash
uv run --frozen --no-sync python scripts/context_benchmark.py --context-profile native-262k --lengths 65536 --state runs/native-64k
uv run --frozen --no-sync python scripts/context_benchmark.py --context-profile native-262k --lengths 102400 --state runs/native-100k
uv run --frozen --no-sync python scripts/context_benchmark.py --context-profile native-262k --lengths 131072 --state runs/native-128k
uv run --frozen --no-sync python scripts/context_benchmark.py --context-profile native-262k --lengths 262144 --state runs/native-262k
```

These are implemented MLX commands, not claims they ran. The suite's lengths are
combined sequence budgets: the default 256-token generation reservation leaves
1792/7936/32512 input tokens for 2K/8K/32K and 261888 for native 262K. Each actual
tokenized input is checked exactly. Report successful **input** length separately
from the configured maximum. A future CUDA implementation needs its own backend
validation; this repository currently implements MLX only.

## Small paired capability suite and prefill profiling

For a new GPU session, run the small primitive/backend gates before a context suite:

```bash
uv run --frozen --no-sync python scripts/preflight_metal_integrity.py --output runs/<new-metal-preflight>.json
uv run --frozen --no-sync python scripts/preflight_backend.py --output runs/<new-backend-preflight>.json
uv run --frozen --no-sync python scripts/context_benchmark.py --lengths 2048 --questions beginning --state runs/<new-context-2k-preflight>
uv run --frozen --no-sync python scripts/context_benchmark.py --lengths 8192 --questions beginning --state runs/<new-context-8k-preflight>
```

Then expand to complete 2K/8K tests; qualify 32K only if those runs and memory
headroom warrant it. The first Phase 3 complete suite preceded the audit addendum;
its original stimuli and measured records remain unchanged. New `--questions`
selection is optional; omitted selection preserves the original full-suite resume
contract. See the [pre-run checkpoint](../reports/phase3-preflight-checkpoint.md).

Five deterministic questions per length cover beginning/middle/end retrieval,
similar distractors, distant addition and a simple inventory calculation. Ground
truth is independently computed by Python. The repeated archive filler is a
bounded diagnostic, not a representative natural-document benchmark. These DEV
tasks are forbidden as training responses. Greedy non-thinking generation isolates
context retrieval from the known thinking-continuation confound.

```bash
uv run --frozen --no-sync python scripts/context_benchmark.py --lengths 2048 8192 32768 --state runs/context-stock-2048-prefill
uv run --frozen --no-sync python scripts/context_benchmark.py --lengths 8192 32768 --prefill-step-size 512 --state runs/context-stock-512-prefill
```

Jobs save full prompts, actual IDs, answers, outputs, seeds, model/adapter identity,
environment and timings durably. Repeating the exact command reuses complete
per-job results; changed contracts are rejected. A 4 GiB available-memory floor
defers escalation, without changing model state or evicting history. The same floor
is checked after prefill chunks and sampled decode boundaries; violations are
retained as explicit generation failures. Sampling cannot guarantee capture of
an intra-chunk memory peak; inspect pressure before launch
and do not escalate on the 18 GiB Mac without evidence from smaller probes.
Run the identical suite with `--adapter-path adapters/<actual-checkpoint>` after SFT.

Installed MLX-LM 0.32.0 forwards `stream_generate` keyword arguments to
`generate_step`, whose prefill default is 2048. Prefill evaluates cache state after
each chunk and clears unused allocator buffers. Qwen's 18 linear-attention blocks
keep convolution and recurrent DeltaNet state; six full-attention blocks retain
growing `KVCache` history. Clearing unused allocation buffers does not erase those
states. No cross-request prompt cache is used. MLX-LM temporarily sets its wired
allocation limit to the device-recommended working set, then restores it; LeanQwen
does not increase system Metal limits or install a larger override.

Generation records contain backend prefill/decode duration, wall/first-token latency,
MLX peak allocation and end allocation/cache, sampled RSS, available system memory
and start/end swap counters. Prefill-chunk and decode snapshots include actual
KV-cache and linear-state bytes. Each request owns fresh, ordinary `KVCache` and
`ArraysCache` objects; unexpected cache types are refused. The wrapper retains
those objects through the final memory snapshot, then releases them on return.
End allocation includes retained request state, while allocator cache bytes are
unused/reusable buffers; these are distinct measurements. Sampling can miss peaks;
swap counters may be unavailable or zero on some platforms. Never sum overlapping
MLX/RSS measurements or infer savings without matched measured runs.

The actual cached-file restoration audit is
[`phase3_context_configuration.json`](../reports/phase3_context_configuration.json).
It verifies unchanged original files and positional metadata without loading the
model. [`phase3_live_profile_restoration.json`](../reports/phase3_live_profile_restoration.json)
also verifies all live loaded tensors and model arguments across short requests
in 128K/262K profiles, with identical emitted IDs. Supply `--adapter-path` to
`profile_restore_probe.py` when an actual adapter exists. Neither audit validates
a full 262K-length request. The [stock context measurements](../reports/phase3-context-stock.md)
cover actual inputs through 32512 tokens; adapter restoration/retention is pending.

### Later varied context suite (design only)

Keep `context_document_v1` frozen for paired adapter evaluation. A future
`context_document_v2` should derive identifiers/values from a pinned deterministic
seed, control fact locations after tokenization, vary distractors (near identifiers,
irrelevant facts, conflicting superseded entries), and use distinct document
forms such as inventory ledgers, dated memos and interleaved tables. Include
missing-fact controls with a separately versioned typed abstention scorer, and
multi-evidence questions whose exact answers are computed independently. Record
separate sample/task IDs and hashes; forbid evaluation responses from training.
This richer suite is not implemented or validated, and does not block the current
reasoning/data/optimizer work. Current five fixed-fact questions establish only
the narrow measured retrieval/calculation behavior, not general context capability.

## Training is a separate budget

The optimizer smoke may use complete examples up to 1024 tokens. Real pilot
examples may use 2048/4096 (8192 only if measured feasible). Examples exceeding a
training bound are excluded whole, never truncated. Short SFT sequences do not
change the native positional configuration or mechanically shorten inference
context. They also do not prove context retention: repeat the paired context suite
at every feasible stock-tested length after applying the adapter.
