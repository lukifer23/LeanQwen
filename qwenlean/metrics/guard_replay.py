"""Replay exact runtime polling on full saved output IDs, including controls."""

from qwenlean.metrics.repetition import suffix_loop


def replay(ids, thresholds, check_every=16, *, opening=None, closing=None, eos_ids=(), thinking=True):
    if check_every < 1:
        raise ValueError('Polling cadence must be positive')
    in_reasoning, reasoning_positions = thinking, []
    for index, token in enumerate(ids, start=1):
        if token in eos_ids:
            break
        if token == closing:
            in_reasoning = False
        elif token == opening:
            in_reasoning = True
        elif in_reasoning:
            reasoning_positions.append(index)
        if in_reasoning and index % check_every == 0:
            evidence = suffix_loop(ids[:index], **thresholds)
            if evidence:
                # Count remaining reasoning without treating final/control tokens as reasoning.
                after_reasoning, state = 0, in_reasoning
                for t in ids[index:]:
                    if t in eos_ids:
                        break
                    if t == closing:
                        state = False
                    elif t == opening:
                        state = True
                    elif state:
                        after_reasoning += 1
                return {**evidence, 'trigger_output_token': index,
                        'trigger_reasoning_token': len(reasoning_positions),
                        'observed_output_tokens_saved': len(ids)-index,
                        'observed_reasoning_tokens_saved': after_reasoning}
    return None
