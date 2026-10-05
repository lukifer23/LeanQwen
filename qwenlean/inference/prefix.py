"""Exact saved-token prefix selection for explicitly labeled interventions."""


def reasoning_prefix(ids, cutoff, *, opening, closing, eos_ids=()):
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError('Nonnegative reasoning cutoff required')
    if cutoff == 0:
        return []
    count, in_reasoning = 0, True
    for i, token in enumerate(ids):
        if token in eos_ids:
            break
        if token == opening:
            in_reasoning = True
        elif token == closing:
            in_reasoning = False
        elif in_reasoning:
            count += 1
        if count == cutoff:
            return list(ids[:i+1])
    raise ValueError('Cutoff exceeds observed reasoning; no fabricated prefix')
