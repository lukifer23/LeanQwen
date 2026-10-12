"""Small deterministic, evaluation-only document suite with exact token budgets."""

from qwenlean.utils.io import digest

VERSION = "context_document_v1"
QUESTIONS = {
    "beginning": ("What is the recorded value for Aster-17?", 37),
    "middle": ("What is the recorded value for Birch-29?", 61),
    "end": ("What is the recorded value for Cedar-43?", 19),
    "distant_sum": ("Add the recorded values for Aster-17, Birch-29, and Cedar-43.", 37 + 61 + 19),
    "document_reasoning": (
        "An order uses Aster-17 units from warehouse one and Birch-29 units from warehouse two. "
        "Cedar-43 units are returned. How many units remain?", 37 + 61 - 19),
}


def build_context_task(tokenizer, sequence_budget, reserve, kind):
    """Construct padding, never truncate a supplied request or source document."""
    input_length = sequence_budget - reserve
    if kind not in QUESTIONS or input_length < 512 or reserve < 1:
        raise ValueError("Invalid context task budget/type")
    question, answer = QUESTIONS[kind]
    filler = ("Archive note: Aster-18 is 38. Birch-28 is 62. Cedar-44 is 20. "
              "These are separate entries from the requested identifiers.\n")
    filler_tokens = len(tokenizer.encode(filler, add_special_tokens=False))
    repetitions, padding = 0, 4
    for iteration in range(8):
        half = repetitions // 2
        body = (
            "Read this archive. Identifiers are exact; similar identifiers are different entries.\n"
            "Aster-17 is 37.\n" + filler * half + " pad" * (padding // 2) +
            "\nBirch-29 is 61.\n" + filler * (repetitions - half) +
            " pad" * (padding - padding // 2) + "\nCedar-43 is 19.\n" +
            question + " End your response with a final line containing only the integer answer."
        )
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer.encode(formatted, add_special_tokens=False)
        remaining = input_length - len(ids)
        if remaining == 0:
            break
        if iteration == 0:
            repetitions = max(0, remaining // filler_tokens - 1)
        else:
            padding += remaining
            if padding < 4:
                repetitions -= 1
                padding += filler_tokens
    else:
        raise ValueError("Could not construct the exact tokenized context budget")
    tid = f"{VERSION}:{sequence_budget}:{kind}"
    return {
        "task_id": tid, "sample_id": tid, "family": "long_context", "difficulty": kind,
        "split": "dev", "prompt": body, "prompt_token_ids": ids,
        "expected": str(answer), "sequence_budget": sequence_budget,
        "actual_input_tokens": len(ids), "reserved_output_tokens": reserve,
        "context_suite_version": VERSION, "document_hash": digest(body),
        "prompt_policy": "P2", "prompt_policy_version": "context_integer_line_v1",
        "provenance": {"source_id": tid, "prompt_origin": "algorithmic",
                       "response_origin": "algorithmic", "license": "Apache-2.0",
                       "training_permitted": False},
    }
