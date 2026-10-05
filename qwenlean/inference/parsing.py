"""Partition generated token IDs, including incomplete prefilling-based thinking."""

from dataclasses import dataclass


@dataclass
class ParsedOutput:
    reasoning: str
    final: str
    reasoning_tokens: int
    final_tokens: int
    control_tokens: int
    total_output_tokens: int
    parse_status: str
    reasoning_ids: list[int]


def parse_tokens(ids, tokenizer, thinking):
    opening = tokenizer.convert_tokens_to_ids("<think>")
    closing = tokenizer.convert_tokens_to_ids("</think>")
    eos = set(tokenizer.eos_token_ids)
    reasoning, final = [], []
    controls = 0
    in_think = thinking
    closed = not thinking
    for token in ids:
        if token in eos:
            controls += 1
        elif token == opening:
            controls += 1
            in_think = True
            closed = False
        elif token == closing:
            controls += 1
            in_think = False
            closed = True
        elif in_think:
            reasoning.append(token)
        else:
            final.append(token)
    status = "complete" if closed else "unclosed_thinking"
    return ParsedOutput(
        tokenizer.decode(reasoning),
        tokenizer.decode(final),
        len(reasoning),
        len(final),
        controls,
        len(ids),
        status,
        reasoning,
    )
