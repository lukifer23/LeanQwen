"""Replay the runtime guard's exact polling cadence on saved reasoning IDs."""

from qwenlean.metrics.repetition import suffix_loop


def replay(ids, thresholds, check_every=16):
    if check_every < 1:
        raise ValueError('Polling cadence must be positive')
    for n in range(check_every, len(ids)+1, check_every):
        evidence = suffix_loop(ids[:n], **thresholds)
        if evidence:
            return {**evidence, 'trigger_reasoning_token': n,
                    'observed_reasoning_tokens_saved': len(ids)-n}
    return None
