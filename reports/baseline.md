# Baseline A — original Qwen weights, official thinking sampler

Measured on 60 DEV problems, ten each across six deterministic families. TEST was not evaluated.
Pinned original bf16 text weights, MLX on M3 Pro. Total-output cap: 2048 tokens. No runtime guard.

| Variant | Accuracy | Median Think Tokens | P95 Think Tokens | Exact Loop Rate | Max-Out Rate | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| A: official thinking | 0.0% | 2048 | 2048 | 0.0% | 100.0% | 33.90 s |

Accuracy 95% Wilson interval: [0.0, 0.06017185214208986].
Reasoning mean/p75/p90/p99: 2044.1 / 2048.0 / 2048.0 / 2048.0.
Aggregate output throughput (including prefill): 60.42 tokens/s.
MLX allocator peak: 1.684 GB. Sampled process RSS peak: 2.140 GB.
Memory measures overlap and are not total-machine usage.
Mean repeated-content density: 7.76%. Longest repeated span: 33 words.
Candidate correct-conclusion found in 40/60 traces; this is a heuristic, not proof of solution.

## By task family

| Family | Correct / N | Accuracy | Median / P95 Reasoning |
|---|---:|---:|---:|
| algebra | 0 / 10 | 0.0% | 2048 / 2048 |
| arithmetic | 0 / 10 | 0.0% | 2048 / 2048 |
| boolean_logic | 0 / 10 | 0.0% | 2048 / 2048 |
| ordering | 0 / 10 | 0.0% | 2048 / 2048 |
| state_machine | 0 / 10 | 0.0% | 2048 / 2048 |
| word_problem | 0 / 10 | 0.0% | 2048 / 2048 |

## Inspection candidates

The following are actual trace excerpts. Excerpts can suggest failure modes; attribution requires manual inspection. Full outputs and token IDs remain in the local run directory.

### capped: `dev-arithmetic-0e6e42cd6282`

Calculate (49 * 57) - 58. End your response with FINAL: followed by just the integer answer.

Ground truth: 2735; final score: False; reasoning tokens: 2048; termination: max_output_tokens.

Reasoning beginning:

```text
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Calculate (49 * 57) - 58.
    *   Constraint: End with "FINAL:".
    *   Constraint: Followed by just the integer answer.

2.  **Perform the Calculation:**
    *   Step 1: Multiply 49 by 57.
        *   49 * 57 = 49 * (60 - 3) OR 49 * (50 + 7) OR simply calculate digit by digit.
        *   49 * 57
        *   49 * 5 = 245
        *   49 * 7 = 343
        *   245 + 343 = 588
        *   Alternatively: (50-1)(50+7) = 2500 + 350 - 50 - 7? No.
        *   Using standard multiplication:
            57 * 49
            -----
             57
            343 (Wait, 7*4=28 carry 0. Actually easier to do (50+7)*57 is not quite right. Stand
```

Reasoning ending:

```text
 = 2280.
    *   2280 + 450 = 2730.
    *   2730 + 63 = 2793.
    *   Correct.
    *   Now, 2793 - 58.
    *   2793 - 50 = 2743.
    *   2743 - 8 = 2735.
    *   Correct.
    *   Result: 2735.

3.  **Check Constraints:**
    *   Calculate (49 * 57) - 58.
    *   End response with "FINAL:".
    *   Followed by just the integer answer.
    *   Format: "Final:" is required at the end, meaning literally the text "FINAL:", but wait, does it mean a colon or period? "End your response with FINAL: followed by just the integer answer."
    *   It says "End your response with FINAL: followed by just the integer answer."
    *   This implies the last characters of the output should be `FINAL:` and then
```

Final answer:

```text

```

### largest_literal_repeat: `dev-state_machine-2a23df54f658`

Start with state 46, reduce it modulo 64. Apply in order: subtract 8, add 3, add 4, add 8, multiply 6, add 2. After EVERY operation reduce modulo 64 to a nonnegative remainder. What is the final state? End your response with FINAL: followed by just the integer answer.

Ground truth: 0; final score: False; reasoning tokens: 2048; termination: max_output_tokens.

Reasoning beginning:

```text
The user wants me to simulate a series of modular arithmetic operations starting from the number 46.
I need to apply these instructions sequentially.
The sequence is: `46 -> 46 - 8` -> `... -> ... -> Final state`.

Wait, let's re-read the constraints and logic carefully.
1. Start with 46.
2. Reduce modulo 64 after every operation (reduce remainder to non-negative).
3. Apply in order: Sub 8, Add 3, Add 4, Add 8, Mult 6, Add 2.
   Actually, looking at the prompt "after EVERY operation reduce modulo 64", this means I must perform the math operation on the *current* value and then do another reduction mod 64. Or does it mean "apply each instruction"?
   Let's look at the wording: "Start with sta
```

Reasoning ending:

```text
f I had 500, I'd do $500 \pmod{64}$.
   But wait, does the instruction say "Start with 46, reduce it modulo 64. Then apply...?"
   
   Okay, let's re-read carefully: "Start with state 46, reduce it modulo 64. Apply in order: subtract 8, add 3, add 4, add 8, multiply 6, add 2. After EVERY operation reduce modulo 64 to a nonnegative remainder."
   
   Is it possible "reduce it modulo 64" was meant to be followed by the whole sequence?
   The word "after every operation" modifies the last part.
   However, usually in these prompts, the modulo applies to the result of every step.
   
   Let's parse step-by-step.
   $S_0 = 46$.
   Op: Subtract 8 $\rightarrow S_1 = 38$.
   Op: Add 3 $\rightarrow S
```

Final answer:

```text

```

### candidate_conclusion_then_failure: `dev-arithmetic-c61a2e941069`

Calculate (50 * 41) - 44. End your response with FINAL: followed by just the integer answer.

Ground truth: 2006; final score: False; reasoning tokens: 2048; termination: max_output_tokens.

Reasoning beginning:

```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: Calculate (50 * 41) - 44.
    *   Constraint: End the response with "FINAL:".
    *   Content: Just the integer answer.

2.  **Perform the Calculation:**
    *   Step 1: Multiply 50 and 41.
        *   $50 \times 40 = 2000$
        *   $50 \times 1 = 50$
        *   $2000 + 50 = 2050$
    *   Step 2: Subtract 44 from the result.
        *   $2050 - 44$
        *   $2050 - 40 = 2010$
        *   $2010 - 4 = 2006$

3.  **Format the Output:**
    *   The user explicitly requested the final line to be "FINAL:" followed by just the integer answer.
    *   Wait, looking closely at the instruction: "End your response with FINAL: followe
```

Reasoning ending:

```text
ure that the response ends with the specific string.
    *   Okay, I will generate the answer with the number and then close with "FINAL: 2006".
    *   Wait, does "FINAL:" need to be the start of the last line or the last line includes the answer?
    *   Let's simplify. I will show the calculation and then finish with the exact requirement.
    *   Wait, what if it's a trick and I shouldn't even write the integer in the last line unless it's mandatory?
    *   Let's stick to the explicit request.
    *   Final decision: Output the steps (optional but helpful) and then put "FINAL:" and the number. Or maybe "Answer: 2006\nFINAL:"?
    *   Best interpretation: The response should end with the
```

Final answer:

```text

```

## Limits

This is a small procedural DEV suite, not a general reasoning benchmark. The cap censors long traces. Exact cycle detection misses paraphrased/reconsideration loops. Lexical cosine is only a semantic redundancy proxy. One generation seed per problem does not measure sampler seed variance. The FINAL instruction may itself influence format deliberation.

## Reproduction

Raw run: `runs/20261005T170152-A-official-thinking-0d62e9fa`.
```bash
uv sync --frozen --extra compat
uv run --frozen --extra compat qwenlean eval --config configs/baseline.yaml
```
