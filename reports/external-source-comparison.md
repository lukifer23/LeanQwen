# External reasoning source decision — 2026-10-12

Selected: original human-written GSM8K TRAIN, pinned and MIT licensed. The first
pilot uses 768 intact solutions and reserves 64 distinct DEV questions. No Qwen,
LFM, coding-agent solutions, paid API calls or synthetic-source imports.
This is external-data progress; no optimizer, adapter or improvement result exists.

Starting repository SHA: `0a62326a9bb5ddc5cbb3af89689ac7dd5839a672` on `main`.
The earlier 64-generation negative pilot and solver corpus are preserved unchanged.
See [machine-readable measurements](external-source-comparison-v2.json) and
[pilot audit](external-human-pilot.json).

## Source and permission comparison

| Candidate / publisher / official source | Revision | Available data | Permission decision |
|---|---|---|---|
| [Original GSM8K](https://github.com/openai/grade-school-math), OpenAI / human writers | `3101c7d5072418e28b9008a6636bde82a006892c` | 7,473 TRAIN problems with complete human solutions; 1,319 reserved official TEST | MIT, original human source; enabled bounded ingestion. Separate Socratic/model solutions excluded. |
| [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), Hugging Face Open R1 | `e4e141ec9dea9f8326f4d347be56105859b2bd68` | Default 93,733; extended 131,396; combined 225,129 problem rows, multiple traces | Apache-2.0 card and permissive full DeepSeek-R1 outputs; underlying NuminaMath problem rights/benchmark origins still need per-source audit. Disabled for TRAIN. |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2), NVIDIA | `469216e3f46f4dacf476b382e192485ea51a143e` | 13,972,791 solution pairs; distinct problems not established | CC-BY-4.0 dataset, Llama-3.1-405B-Instruct outputs. Upstream license includes obligations for output-trained models; disabled pending compatibility decision. |
| [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K), DeepMath authors | `5cf055d1fe3d7a2eb19719ac020211469736ae44` | 103,022 problems, three full DeepSeek-R1 solutions each | MIT card/permissive teacher; underlying mixed problem rights and contamination unresolved. Disabled for TRAIN. |

OpenMathInstruct-2's dataset license does not remove the upstream requirement that
an output-trained model distributed or made available have a name beginning with
Llama. Other attribution and use conditions also apply. This is a compatibility
hold, not a claim that distillation is categorically prohibited.
[Exact Llama 3.1 license](https://raw.githubusercontent.com/meta-llama/llama-models/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_1/LICENSE).

OpenR1's NuminaMath ancestry includes scraped contest/forum materials and mixed
upstream sources. Card licenses and a correct final-answer flag are insufficient
for automatic import. Its sampled rows are research-only: the viewer API does not
attest an immutable revision, although repository HEAD matched before/after sampling.
All card revisions, raw pages, permission snapshots and hashes are retained locally.

## Actual token measurements

All counts use the pinned Qwen tokenizer, not a teacher tokenizer. Synthetic
sources were sampled in four predeclared windows of 16 problems each: first,
one-third, two-thirds and last. This is a bounded diagnostic, not an unbiased
population estimate. Total large-source response download: 7,554,902 bytes;
no giant parquet download. GSM8K's small original TRAIN file was measured in full.

| Measurement | Problems / traces | Mean | Median | p90 | p95 | Range |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K reasoning before final delimiter | 7,473 / 7,473 | 119.45 | 108 | 193 | 226 | 25–446 |
| OpenR1 reasoning segment when explicit, otherwise whole solution | 64 / 134 | 5,903.67 | 4,686 | 12,340.1 | 13,493.25 | 1,046–17,725 |
| OpenR1 complete raw response including final/control text | 64 / 134 | 6,428.51 | 5,140.5 | 13,213.2 | 14,109.6 | 1,548–17,778 |
| OpenMathInstruct-2 whole worked solution, no uniform thinking delimiter | 64 / 64 | 312.59 | 248.5 | 576.1 | 748.65 | 54–867 |
| DeepMath whole worked solution, no uniform thinking delimiter | 64 / 192 | 6,124.53 | 4,553 | 12,415.3 | 14,841.5 | 902–23,959 |

Explicit thinking segments are whitespace-trimmed for token measurement only;
source training text is never rewritten. OpenR1 reasoning rows comprise 126 explicit thinking segments and eight whole
solutions with no uniform delimiter. Its publisher flags identify 126 complete
and 94 Math-Verify-correct sampled traces; these are not independent intermediate
proof checks. More than one trace per problem is possible; do not treat every
trace as correct because a problem has a successful candidate.

The reproducible deterministic proxy finds exact repeated sentence/line units
(minimum 40 normalized characters) in 58/134 OpenR1 traces,
0/64 OpenMathInstruct traces and 89/192 DeepMath traces.
Its broad reconsideration cues occur in 134/134,
0/64 and 192/192 respectively. The exact regex and
method are stored in the JSON and `scripts/summarize_external_sources.py`.
These cues are not proof of waste, semantic judgments, or grounds to delete useful
checks. No subjective language-model judge or rewriting was used.

## Quality, difficulty and suitability

GSM8K has human natural-language solutions and elementary multi-step word problems.
Published QA uses worker answer agreement. The selected subset adds exact annotation
checks and conservative completion/repetition screening. Prose interpretation is
not formally verified; the bounded approval explicitly retains that uncertainty.
[Published human QA](https://arxiv.org/html/2110.14168v2).
It is the clearest permitted, complete, varied and manageable starting source for
an 0.8B student. It covers word-problem arithmetic rather than all reasoning domains.

OpenR1 spans algebra, number theory, geometry, combinatorics and other competition
math. Final answers use Math-Verify and, for some cases, a Llama judge; neither
certifies all reasoning. Its natural traces are predominantly too long for the
initial complete-example smoke. Source rights and efficient, complete per-trace
selection need work before import; do not compress/crop them to fit.

OpenMathInstruct-2 offers more manageable worked solutions over GSM8K/MATH TRAIN
and augmented problems. Augmented labels use teacher majority voting, not an
independent mathematical proof. It may be useful later, after upstream obligations
and per-example quality/benchmark audit are resolved. No sampled exact-repeat cue
is evidence only within this small diagnostic, not a corpus-wide quality guarantee.

DeepMath contains challenging multi-domain math, mostly difficulty levels 5–9.
It could provide productive long dependencies later, but sampled traces are mostly
far beyond initial bounds. Mixed origins and hard-task correctness need audit;
longness alone does not make the solutions defective. Not selected for this pilot.

## Storage and import costs

| Source | Published/source storage | Bounded action |
|---|---|---|
| GSM8K | TRAIN 4.17 MB; TEST metadata source 0.75 MB; MIT notice 1,062 bytes | Downloaded pinned originals; 768 TRAIN + 64 DEV selected without rewriting |
| OpenR1 | Default ~2.15 GB compressed / 4.96 GB decoded; combined ~4.22 / 9.73 GB | 64 problems only; no training import |
| OpenMathInstruct-2 | Main TRAIN ~15.55 GB decoded; card totals over overlapping configurations ~20.2 GB download / 26.2 GB decoded | 64 rows only; no training import |
| DeepMath | ~2.14 GB compressed / 4.96 GB decoded | 64 problems only; no training import |

Large raw archives, generated datasets and weights stay outside Git. Compact
manifests, revisions, file hashes, recipes and measurements are tracked.

## Frontier decision and first experiment

A paid teacher is unnecessary for the first human-reference pilot. Full
[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) explicitly permits
distillation, and audited [official API terms](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html)
permit output use for training. However, [current API models](https://api-docs.deepseek.com/quick_start/pricing/)
are V4 aliases; an R1 model license does not attest a current R1 API endpoint.
No provider or spend is approved, no paid calls were made, and frontier ingestion
remains disabled. Further expansion may use an independently audited source or a
specifically approved provider, with raw responses and exact permissions archived.

First execute the four-step technical MLX smoke on eight intact human solutions:
rank 4, scale 2, final two mixed-attention blocks, AdamW 1e-4, batch/accumulation 1,
1024 maximum sequence bound, 512-token fresh-inference allowance. Check finite
loss/gradients, nonzero updates, full frozen-base integrity and exact reload logits.
Actual complete sequences are shorter than 512, so this does not test 1024 capacity.

The frozen matched DEV cohort contains 24 within-source GSM8K questions plus the
existing 32 procedural retention tasks, both modes, 2048-output-token bounds.
See [frozen protocol](external-human-dev-protocol.json).

After a complete matched stock DEV baseline and measured hardware gates, the
proposed first improvement schedule is 128 updates, accumulation 4, complete-example
bound 512, checkpoints every 32 updates, with periodic matched DEV evaluation.
Final configuration remains conditional on the smoke, capacity and retention
results. No improvement, formatting-only effect, or before/after token savings
can yet be concluded. Outside review is valuable for source expansion and results;
no additional user choice is required to execute this bounded human-data route.
