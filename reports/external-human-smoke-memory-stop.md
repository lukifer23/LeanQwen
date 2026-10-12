# External human-source technical smoke — resource stop

The four-step MLX LoRA attempt used eight intact licensed human GSM8K solutions.
It computed its first forward/backward pass, then the existing memory gate stopped
it **before optimizer update 1**. No saved adapter, finite-gradient/loss approval,
save/reload equivalence, fresh adapter inference or improvement result exists.
This is a resource-admission failure, not a negative SFT outcome.

Available RAM at stop: **2,773,237,760 bytes** (2.583 GiB),
below the unchanged **4,294,967,296 byte** (4 GiB) floor.
MLX active allocation: 1,520,987,980 bytes;
MLX allocator peak: **2,896,505,106 bytes** (2.897 GB).
Observed RSS at stop: 2,076,753,920 bytes.
Swap-out delta from training start: 4,161,536 bytes,
below the separate 512 MiB swap-out limit. Allocation/RSS are overlapping measures.
No completed step time or loss was recorded; neither is inferred from timestamps.

Adapter attachment preserved the full loaded-base fingerprint before optimization.
Post-training integrity/reload checks were never reached. Native context/RoPE and
tokenizer changes were not made. Configuration: rank4/scale2, last two mixed-attention
blocks, AdamW1e-4, batch1/accumulation1, four requested steps, max complete sequence1024.
All eight actual examples are complete and shorter than512; no truncation. The
first seeded example's length is reproducibly reconstructed in the JSON, not
misrepresented as a runtime token-length measurement.

Raw console output, completion, allocation snapshot, original/attached base
fingerprints and empty loss log are preserved under
`adapters/external-human-smoke-v1-attempt1`; exact hashes and configuration are in
[the measured report](external-human-smoke-memory-stop.json). Dataset/source
fingerprints match the approved human corpus. Qwen was released and the lease
reacquired for a read-only check; no unrelated process was terminated.

Do not weaken the floor or launch an improvement pilot. Next action is the same
bounded smoke when memory supports its unchanged gate, followed by integrity/reload,
complete stock DEV baseline and capacity checks. No new Qwen/LFM teacher generation,
paid API calls, source expansion or outside review is needed for that technical retry.
