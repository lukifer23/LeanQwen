"""Versioned deterministic stimuli and operation-producing solvers; no teacher prose.

Verification lives in solver_verify.py and uses different algorithms. All text in
targets is a fixed rendering of computed operations, not supplied worked answers.
"""

import random
from fractions import Fraction

from qwenlean.datasets.procedural_v2 import FAMILIES, render_ast, structure
from qwenlean.utils.io import digest

VERSION = 'solver-reasoning-v1.1'
GENERATOR_VERSION = 'solver-tasks-1.0.1'
SOLVER_VERSION = 'operation-solver-1.0.0'
SEEDS = {'train': 410031, 'dev_id': 410061, 'dev_ood': 410091, 'test': 410121}


def numeric(value):
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else str(value)


def make_tree(rng, count, boolean=False):
    if count == 0:
        return ['var', rng.randrange(6)] if boolean else rng.randint(-9, 12)
    left = rng.randrange(count)
    if boolean and rng.random() < .2:
        return ['NOT', make_tree(rng, count-1, True)]
    op = rng.choice(['AND', 'OR', 'XOR'] if boolean else ['+', '-', '*'])
    return [op, make_tree(rng, left, boolean), make_tree(rng, count-1-left, boolean)]


def task_spec(rng, family, index):
    # Vary topology as well as numbers. Rejection below enforces structure uniqueness.
    count = rng.choice([1, 2, 3, 4, 5, 7, 9, 12, 16, 24])
    if family in {'arithmetic', 'boolean_logic'}:
        node = make_tree(rng, count, family == 'boolean_logic')
        spec = {'ast': node}
        if family == 'boolean_logic':
            spec['values'] = [rng.choice([False, True]) for _ in range(6)]
        return spec, [family, structure(node)], count
    if family == 'state_machine':
        ops = [[rng.choice(['add', 'subtract', 'multiply']), rng.randint(1, 9)]
               for _ in range(count)]
        return {'initial': rng.randint(-20, 50), 'modulus': rng.randint(3, 43), 'ops': ops}, [family, [o for o, _ in ops]], count
    if family == 'sequence':
        size = rng.randint(3, 12)
        ops = [rng.choice(['reverse', 'sort', 'rotate_left', 'negate']) for _ in range(count)]
        pos = rng.randrange(size)
        return {'input': [rng.randint(-30, 30) for _ in range(size)], 'ops': ops, 'position': pos}, [family, size, ops, pos], count
    if family == 'ordering':
        size = rng.randint(3, 16)
        names = rng.sample(range(100, 999), size)
        edges = [(i, i+1) for i in range(size-1)]
        extra = [(i, j) for i in range(size) for j in range(i+2, size)]
        edges += rng.sample(extra, rng.randrange(min(len(extra), 12)+1))
        pos = rng.randrange(size)
        sig = [family, size, sorted(edges), pos]
        concrete = [[names[i], names[j]] for i, j in edges]
        rng.shuffle(concrete)
        rng.shuffle(names)
        return {'nodes': names, 'edges': concrete, 'position': pos}, sig, size-1
    if family == 'algebra':
        x = rng.randint(-12, 12)
        ops = [[rng.choice(['+', '-', '*']), rng.randint(2, 7)] for _ in range(count)]
        node, rhs = 'x', x
        for op, operand in ops:
            node = [op, node, operand]
            rhs = {'+': lambda: rhs+operand, '-': lambda: rhs-operand,
                   '*': lambda: rhs*operand}[op]()
        return {'ops': ops, 'rhs': rhs}, [family, [o for o, _ in ops]], count
    if family == 'word_problem':
        # Divisible by lcm(10**4, 4**4): up to four exact-cent discounts.
        baskets = [[rng.randint(1, 8), rng.randint(1, 9), rng.randint(1, 20)*160000]
                   for _ in range(rng.randint(1, 5))]
        rates = [rng.choice([10, 20, 25, 50]) for _ in range(rng.randint(0, 4))]
        fees = [rng.randint(1, 99) for _ in range(rng.randint(0, 4))]
        return {'baskets': baskets, 'rates': rates, 'fees': fees}, [family, len(baskets), len(rates), len(fees)], len(baskets)*3+len(rates)+len(fees)
    if family == 'counting':
        n, k, m = rng.randint(4, 20), rng.randint(1, 4), rng.randint(2, 5)
        residue = rng.randrange(m)
        return {'n': n, 'k': k, 'modulus': m, 'residue': residue, 'start': 1}, [family, n, k, m], n
    raise ValueError('Unknown family')


def body_for(family, s):
    if family == 'arithmetic':
        return f'Calculate {render_ast(s["ast"])}.'
    if family == 'boolean_logic':
        vals = ', '.join(f'{chr(65+i)}={v}' for i, v in enumerate(s['values']))
        return f'{vals}. Evaluate {render_ast(s["ast"])}. Give 1 for true or 0 for false.'
    if family == 'state_machine':
        return (f'Start at {s["initial"]} modulo {s["modulus"]}. Apply in order: '
                + ', '.join(f'{op} {n}' for op, n in s['ops'])
                + '. Reduce to a nonnegative remainder after each operation. What is the final state?')
    if family == 'sequence':
        definitions = {'reverse': 'reverse reverses the list', 'sort': 'sort is ascending',
                       'rotate_left': 'rotate_left moves the first item to the end',
                       'negate': 'negate changes every sign'}
        glossary = '; '.join(definitions[o] for o in dict.fromkeys(s['ops']))
        return (f'List {s["input"]}. Apply in order: {", ".join(s["ops"])}. {glossary}. '
                f'What integer is in position {s["position"]+1}, counting from 1?')
    if family == 'ordering':
        return ('Runner numbers are identifiers. All finish in distinct positions. '
                + '; '.join(f'{a} precedes {b}' for a, b in s['edges'])
                + f'. Which runner is in position {s["position"]+1}, counting from 1?')
    if family == 'algebra':
        expr = 'x'
        for op, n in s['ops']:
            expr = f'({expr} {op} {n})'
        return f'Solve for integer x: {expr} = {s["rhs"]}.'
    if family == 'word_problem':
        text = '; '.join(f'{b} boxes of {i} items at {p} cents each' for b, i, p in s['baskets'])
        if s['rates']:
            text += '. Apply successive discounts of ' + ', then '.join(f'{r}%' for r in s['rates'])
        if s['fees']:
            text += '. Finally add fees of ' + ', '.join(f'{f} cents' for f in s['fees'])
        return text + '. What is the total cost in exact cents?'
    if family == 'counting':
        return (f'How many unordered size-{s["k"]} subsets of {s["start"]} through {s["start"]+s["n"]-1} have '
                f'sum congruent to {s["residue"]} modulo {s["modulus"]}? No repeated elements.')
    raise ValueError('Unknown family')


def solve(family, s):
    steps = []

    def emit(op, before, after, context=None):
        steps.append({'operation': op, 'before': before, 'after': after, 'context': context})

    def tree(node):
        if isinstance(node, int):
            return Fraction(node)
        if node[0] == 'var':
            return s['values'][node[1]]
        op, *args = node
        values = [tree(a) for a in args]
        result = {'+': lambda: values[0]+values[1], '-': lambda: values[0]-values[1],
                  '*': lambda: values[0]*values[1], 'NOT': lambda: not values[0],
                  'AND': lambda: values[0] and values[1], 'OR': lambda: values[0] or values[1],
                  'XOR': lambda: bool(values[0]) != bool(values[1])}[op]()
        emit(op, [numeric(v) for v in values], numeric(result), node)
        return result

    if family in {'arithmetic', 'boolean_logic'}:
        answer = tree(s['ast'])
    elif family == 'state_machine':
        value = s['initial'] % s['modulus']
        emit('reduce', s['initial'], value, s['modulus'])
        for op, n in s['ops']:
            raw = {'add': lambda: value+n, 'subtract': lambda: value-n,
                   'multiply': lambda: value*n}[op]()
            new = raw % s['modulus']
            emit(op, value, new, {'operand': n, 'raw': raw, 'modulus': s['modulus']})
            value = new
        answer = value
    elif family == 'sequence':
        value = s['input'][:]
        for op in s['ops']:
            new = {'reverse': lambda: value[::-1], 'sort': lambda: sorted(value),
                   'rotate_left': lambda: value[1:]+value[:1],
                   'negate': lambda: [-v for v in value]}[op]()
            emit(op, value, new)
            value = new
        emit('select', value, value[s['position']], s['position'])
        answer = value[s['position']]
    elif family == 'ordering':
        remaining, result = sorted(s['nodes']), []
        while remaining:
            ready = [n for n in remaining if not any(b == n and a in remaining for a, b in s['edges'])]
            if len(ready) != 1:
                raise ValueError('Ordering must have exactly one available runner')
            node = ready[0]
            new = [n for n in remaining if n != node]
            emit('place', remaining, new, {'runner': node, 'position': len(result)+1})
            result.append(node)
            remaining = new
        emit('select', result, result[s['position']], s['position'])
        answer = result[s['position']]
    elif family == 'algebra':
        value = Fraction(s['rhs'])
        for i in range(len(s['ops'])-1, -1, -1):
            op, n = s['ops'][i]
            new = {'+': lambda: value-n, '-': lambda: value+n, '*': lambda: value/n}[op]()
            emit('undo_'+op, numeric(value), numeric(new), {'operand': n, 'remaining_ops': s['ops'][:i]})
            value = new
        emit('substitute', numeric(value), str(s['rhs']), s['ops'])
        answer = value
    elif family == 'word_problem':
        total = 0
        for b, i, p in s['baskets']:
            items, cost = b*i, b*i*p
            emit('*', [str(b), str(i)], str(items))
            emit('*', [str(items), str(p)], str(cost))
            emit('+', [str(total), str(cost)], str(total+cost))
            total += cost
        for rate in s['rates']:
            new = Fraction(total)*(100-rate)/100
            emit('discount', numeric(total), numeric(new), rate)
            total = new
        for fee in s['fees']:
            emit('+', [numeric(total), str(fee)], numeric(total+fee))
            total += fee
        answer = total
    elif family == 'counting':
        m, k = s['modulus'], s['k']
        table = [[0]*m for _ in range(k+1)]
        table[0][0] = 1
        for value in range(s['start'], s['start']+s['n']):
            new = [row[:] for row in table]
            for size in range(1, k+1):
                for rem in range(m):
                    new[size][(rem+value) % m] += table[size-1][rem]
            emit('include_or_exclude', table, new, value)
            table = new
        emit('select_count', table, table[k][s['residue']], {'size': k, 'residue': s['residue']})
        answer = table[k][s['residue']]
    else:
        raise ValueError('Unknown family')
    if Fraction(answer).denominator != 1:
        raise ValueError('Final answer must be an exact integer')
    return {'initial_state': s, 'steps': steps, 'final_answer': numeric(answer)}


def render_proof(proof):
    lines = []
    for step in proof['steps']:
        op, before, after, ctx = (step[k] for k in ('operation', 'before', 'after', 'context'))
        if op in {'+', '-', '*', 'NOT', 'AND', 'OR', 'XOR'}:
            expression = f'NOT {before[0]}' if op == 'NOT' else f'{before[0]} {op} {before[1]}'
            lines.append(f'{expression} = {after}.')
        elif op == 'reduce':
            lines.append(f'{before} mod {ctx} = {after}.')
        elif op in {'add', 'subtract', 'multiply'}:
            symbol = {'add': '+', 'subtract': '-', 'multiply': '*'}[op]
            lines.append(f'({before} {symbol} {ctx["operand"]}) mod {ctx["modulus"]} = {after}.')
        elif op.startswith('undo_'):
            symbol = {'undo_+': '-', 'undo_-': '+', 'undo_*': '/'}[op]
            lines.append(f'Undo {op[5:]} {ctx["operand"]}: the inner expression equals {before} {symbol} {ctx["operand"]} = {after}.')
        elif op == 'substitute':
            lines.append(f'Substituting x = {before} into the original expression gives {after}, matching the right side.')
        elif op == 'discount':
            lines.append(f'After {ctx}% off: {before} * {100-ctx}/100 = {after} cents.')
        elif op == 'place':
            lines.append(f'Among remaining runners {before}, only {ctx["runner"]} has no remaining predecessor; place it at {ctx["position"]}.')
        elif op == 'include_or_exclude':
            if not lines:
                lines.append('Use counts by subset size (rows) and sum remainder (columns), starting with the empty subset. For each value, exclude it or add it to a smaller subset; use the previous table so no value repeats.')
            lines.append(f'After including value {ctx}, counts are {after}.')
        elif op in {'select', 'select_count'}:
            lines.append(f'Select the requested entry: {after}.')
        else:
            lines.append(f'{op}: {before} -> {after}.')
    return '\n'.join(lines)


def response_structure(family, spec, signature):
    # Ordering's solver produces the same removal schedule for different redundant
    # transitive edges. Deduplicate that response schedule by size/query position.
    if family == 'ordering':
        return [family, len(spec['nodes']), spec['position']]
    return signature


def generate(namespace, per_family=64):
    if namespace not in SEEDS or not 1 <= per_family <= 128:
        raise ValueError('Unsupported namespace/count')
    rng, rows, seen, response_seen = random.Random(SEEDS[namespace]), [], set(), set()
    for family in FAMILIES:
        accepted = 0
        for attempt in range(200000):
            spec, sig, complexity = task_spec(rng, family, attempt)
            fingerprint = digest(sig)
            response_fingerprint = digest(response_structure(family, spec, sig))
            bucket = int(fingerprint[:8], 16) % 5
            wanted = bucket < 3 if namespace in {'train', 'dev_id'} else bucket == (3 if namespace == 'dev_ood' else 4)
            if not wanted or fingerprint in seen or response_fingerprint in response_seen:
                continue
            body = body_for(family, spec)
            proof = solve(family, spec)
            from qwenlean.datasets.solver_verify import verify_proof
            verification = verify_proof(family, spec, proof)
            if not verification['verified']:
                raise ValueError('Independent verification failed')
            task_id = f'{namespace}-{VERSION}-{family}-{digest(spec)[:20]}'
            split = 'dev' if namespace.startswith('dev') else namespace
            difficulty = ('trivial' if complexity <= 2 else 'easy' if complexity <= 5
                          else 'medium' if complexity <= 12 else 'harder')
            rows.append({'task_id': task_id, 'sample_id': task_id, 'seed_key': task_id,
                         'split': split, 'namespace': namespace, 'dataset': VERSION,
                         'family': family, 'difficulty': difficulty, 'structural_complexity': complexity,
                         'structural_signature': sig, 'structural_fingerprint': fingerprint,
                         'response_structure_fingerprint': response_fingerprint,
                         'semantic_fingerprint': digest([family, spec]), 'facts': spec,
                         'body': body, 'prompt': body, 'expected': proof['final_answer'],
                         'generator_seed': SEEDS[namespace], 'generator_attempt': attempt,
                         'content_version': GENERATOR_VERSION,
                         'proof': {**proof, 'verification': verification},
                         'provenance': {'source_id': task_id, 'prompt_origin': 'algorithmic',
                                        'response_origin': 'solver_verified_reasoning',
                                        'license': 'Apache-2.0', 'training_permitted': split == 'train',
                                        'task_generator': VERSION, 'generator_version': GENERATOR_VERSION,
                                        'solver_version': SOLVER_VERSION}})
            seen.add(fingerprint)
            response_seen.add(response_fingerprint)
            accepted += 1
            if accepted == per_family:
                break
        else:
            raise ValueError(f'Inadequate structural diversity for {namespace}/{family}')
    return rows
