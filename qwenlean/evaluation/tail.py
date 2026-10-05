"""Censor-aware summaries and exact same-seed prefix checks."""


def check_prefix(short, long):
    for field in ('task_id', 'seed', 'prompt', 'model_identifier'):
        if short[field] != long[field]:
            raise ValueError(f'Prefix comparison invariant differs: {field}')
    a, b = short['generation_parameters'], long['generation_parameters']
    for field in ('sampling', 'enable_thinking', 'loop_guard'):
        if a.get(field) != b.get(field):
            raise ValueError(f'Prefix sampler mismatch: {field}')
    ids = short['token_ids']
    equal = long['token_ids'][:len(ids)] == ids
    return {'short_generation_id': short['generation_id'], 'long_generation_id': long['generation_id'],
            'short_tokens': len(ids), 'exact_prefix': equal,
            'first_mismatch': next((i for i, (x, y) in enumerate(zip(ids, long['token_ids']))
                                    if x != y), None)}


def survival_bounds(rows, horizons=(2048, 4096, 8192, 16384)):
    latest = {}
    for r in rows:
        key = (r['task_id'], r['seed'])
        if key not in latest or r['generation_parameters']['max_output_tokens'] > latest[key]['generation_parameters']['max_output_tokens']:
            latest[key] = r
    observations = list(latest.values())
    result = []
    for h in horizons:
        ended = sum(r['termination_reason'] == 'eos' and r['total_output_tokens'] <= h for r in observations)
        ongoing = sum(r['total_output_tokens'] >= h and not
                      (r['termination_reason'] == 'eos' and r['total_output_tokens'] <= h)
                      for r in observations)
        unknown = len(observations) - ended - ongoing
        result.append({'token_horizon': h, 'observed_eos_by_horizon': ended,
                       'known_still_generating': ongoing, 'censored_before_horizon': unknown,
                       'survival_lower_bound': ongoing / len(observations),
                       'survival_upper_bound': (ongoing + unknown) / len(observations)})
    return {'selected_trajectories': len(observations), 'horizons': result,
            'natural_completed_reasoning_lengths': [r['reasoning_tokens'] for r in observations
                                                    if r['termination_reason'] == 'eos'],
            'right_censored': sum(r['termination_reason'] != 'eos' for r in observations),
            'population_estimate': False}
