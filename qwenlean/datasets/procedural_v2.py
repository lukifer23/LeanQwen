"""Exact integer tasks with explicit structural split namespaces and difficulty.

No output-format policy is embedded. Structural holdouts are deliberately harder
than random prompt-hash holdouts; they do not guarantee conceptual independence.
"""

import itertools
import random
from fractions import Fraction
from pathlib import Path

from qwenlean.datasets.provenance import check_contamination
from qwenlean.utils.io import digest, write_json, write_jsonl

VERSION = 'procedural-v2'
SPLITS = ('train', 'dev', 'test')
SEEDS = {'train': 711031, 'dev': 722061, 'test': 733091}
DIFFICULTIES = ('trivial', 'easy', 'medium', 'harder')
FAMILIES = ('arithmetic', 'word_problem', 'algebra', 'boolean_logic', 'ordering',
            'state_machine', 'counting', 'sequence')


def evaluate_ast(node, values=None):
    if isinstance(node, int):
        return Fraction(node)
    op, *args = node
    if op == 'var':
        return bool(values[args[0]])
    v = [evaluate_ast(a, values) for a in args]
    if op == '+':
        return v[0] + v[1]
    if op == '-':
        return v[0] - v[1]
    if op == '*':
        return v[0] * v[1]
    if op == '/':
        return v[0] / v[1]
    if op == 'NOT':
        return not v[0]
    if op == 'AND':
        return bool(v[0] and v[1])
    if op == 'OR':
        return bool(v[0] or v[1])
    if op == 'XOR':
        return bool(v[0]) != bool(v[1])
    raise ValueError(f'Unknown AST operator {op}')


def structure(node):
    if isinstance(node, int):
        return '#'
    if node[0] == 'var':
        return 'variable'  # Excludes variable renaming and truth assignments.
    return [node[0], *[structure(a) for a in node[1:]]]


def render_ast(node):
    if isinstance(node, int):
        return str(node)
    op, *args = node
    if op == 'var':
        return chr(65 + args[0])
    if op == 'NOT':
        return f'NOT ({render_ast(args[0])})'
    return f'({render_ast(args[0])} {op} {render_ast(args[1])})'


def owner(signature):
    return SPLITS[int(digest(signature)[:8], 16) % 3]


def random_logic(rng, depth):
    if depth == 0:
        return ['var', rng.randrange(6)]
    if rng.random() < .25:
        return ['NOT', random_logic(rng, depth - 1)]
    return [rng.choice(['AND', 'OR', 'XOR']), random_logic(rng, depth - 1),
            random_logic(rng, depth - 1)]


def make_task(rng, split, family, level):
    namespace = SPLITS.index(split)
    lo = 3 + namespace * 20
    a, b, c = [rng.randint(lo, lo + 16) for _ in range(3)]
    facts = {}
    if family == 'arithmetic':
        # Trivial operators are explicitly held out; deeper trees use hash-owned shapes.
        node = rng.randint(-b, a)
        ops = []
        for i in range(level + 1):
            op = ('+', '-', '*')[namespace] if level == 0 else rng.choice(['+', '-', '*', '/'])
            value = rng.randint(2, 8 + 3 * level)
            if op == '/':
                node = ['/', ['*', node, value], value]  # exact integral fraction
            else:
                leaf = rng.randint(-value, value)
                node = [op, node, leaf] if i % 2 == 0 else [op, leaf, node]
            ops.append(op)
        expected = evaluate_ast(node)
        sig = [family, structure(node)]
        facts = {'ast': node, 'operation_count': len(ops)}
        body = f'Calculate {render_ast(node)}.'
    elif family == 'boolean_logic':
        node = random_logic(rng, level + 2)
        vals = [rng.choice([False, True]) for _ in range(6)]
        expected = int(evaluate_ast(node, vals))
        sig = [family, structure(node)]
        facts = {'ast': node, 'values': vals, 'depth': level + 2}
        body = (', '.join(f'{chr(65+i)}={v}' for i, v in enumerate(vals))
                + f'. Evaluate {render_ast(node)}. What is its truth value as 1 for true or 0 for false?')
    elif family == 'state_machine':
        modulus = rng.randint(7 + 20 * namespace, 23 + 20 * namespace)
        state = a % modulus
        history = [state]
        ops = [(rng.choice(['add', 'subtract', 'multiply']), rng.randint(2, 9))
               for _ in range(2 + 2 * level)]
        for op, n in ops:
            state = {'add': lambda: state + n, 'subtract': lambda: state - n,
                     'multiply': lambda: state * n}[op]() % modulus
            history.append(state)
        expected = state
        sig = [family, [op for op, _ in ops]]
        facts = {'initial': a, 'modulus': modulus, 'ops': ops, 'history': history}
        body = (f'Start at {a} modulo {modulus}. Apply in order: '
                + ', '.join(f'{op} {n}' for op, n in ops)
                + f'. Reduce after every operation to a nonnegative remainder modulo {modulus}. What is the final state?')
    elif family == 'sequence':
        nums = [rng.randint(-b, a) for _ in range(3 + 2 * level)]
        ops = [rng.choice(['reverse', 'sort', 'rotate_left', 'negate'])
               for _ in range(2 + level)]
        result = nums[:]
        for op in ops:
            if op == 'reverse':
                result.reverse()
            elif op == 'sort':
                result.sort()
            elif op == 'rotate_left':
                result = result[1:] + result[:1]
            else:
                result = [-x for x in result]
        pos = rng.randrange(len(nums))
        expected = result[pos]
        sig = [family, len(nums), ops]
        facts = {'input': nums, 'ops': ops, 'result': result, 'position': pos + 1}
        body = (f'List {nums}. Apply in order: {", ".join(ops)}. sort means ascending; '
                'rotate_left moves the first item to the end; negate changes every sign. '
                f'What is the integer in position {pos+1}, counting from 1?')
    elif family == 'ordering':
        # Entity counts are structural holdouts; distractor edges do not alter the unique order.
        size = 3 + 3 * level + namespace
        order = rng.sample(range(lo * 10, (lo + 16) * 10), size)
        constraints = [(i, i + 1) for i in range(size - 1)]
        extra = [(i, j) for i in range(size) for j in range(i + 2, size)]
        constraints += rng.sample(extra, min(level, len(extra)))
        rng.shuffle(constraints)
        pos = rng.randrange(size)
        expected = order[pos]
        sig = [family, size, 'chain_with_transitive_distractors']
        facts = {'order': order, 'position': pos + 1, 'edges': constraints}
        body = (f'{size} numbered runners finish in distinct positions. '
                + '; '.join(f'{order[i]} precedes {order[j]}' for i, j in constraints)
                + f'. Which runner is in position {pos+1}, counting from 1?')
    elif family == 'algebra':
        x = rng.randint(-b, b)
        if namespace == 0:
            rhs = a * (x + b) - c
            equation = f'{a} * (x + {b}) - {c} = {rhs}'
            sig = [family, 'shift_scale_subtract']
        elif namespace == 1:
            if a == b:
                b += 1
            rhs = a * x - b * (x - c)
            equation = f'{a} * x - {b} * (x - {c}) = {rhs}'
            sig = [family, 'difference_of_variable_terms']
        else:
            rhs = (a + b) * x + c * x + level
            equation = f'({a} + {b}) * x + {c} * x + {level} = {rhs}'
            sig = [family, 'sum_of_variable_terms']
        base_rhs = rhs
        for step in range(level):
            shift = step + 1
            lhs = equation.split(' = ')[0]
            rhs = 2 * (rhs + shift)
            equation = f'2 * ({lhs} + {shift}) = {rhs}'
        sig.append(level)
        expected = x
        facts = {'a': a, 'b': b, 'c': c, 'rhs': rhs, 'solution': x, 'template': sig[1], 'base_rhs': base_rhs, 'wrappers': level}
        body = f'Solve for the integer x: {equation}.'
    elif family == 'counting':
        n = 4 + level * 3 + namespace + rng.randrange(5)
        k = 1 + level
        # Split-specific restrictions and exact enumeration, bounded to C(19,4).
        values = list(range(1, n + 1))
        restrictions = ('sum is even', 'sum is divisible by 3', 'sum is odd')
        subsets = itertools.combinations(values, k)
        pred = (lambda v: sum(v) % 2 == 0, lambda v: sum(v) % 3 == 0,
                lambda v: sum(v) % 2 == 1)[namespace]
        expected = sum(pred(v) for v in subsets)
        sig = [family, n, k, restrictions[namespace]]
        facts = {'n': n, 'k': k, 'restriction': restrictions[namespace]}
        body = (f'How many unordered subsets of size {k} from integers 1 through {n} '
                f'have a {restrictions[namespace]}? Each integer may appear at most once.')
    else:
        boxes, items, price = a, b, c * 100
        discount = (10, 20, 25, 50)[level]
        if namespace == 0:
            expected = boxes * items * price * (100 - discount) // 100
            body = (f'{boxes} boxes contain {items} items each at {price} cents per item. '
                    f'All items receive a {discount}% discount. What is the total price in cents?')
            sig = [family, 'bulk_discount']
        elif namespace == 1:
            expected = boxes * items * price * (100 - discount) // 100 + c
            body = (f'Buy {boxes} packs of {items} items at {price} cents each with '
                    f'{discount}% off the items and a {c}-cent delivery fee. What is the total cost in cents?')
            sig = [family, 'bulk_discount_plus_fee']
        else:
            expected = boxes * items * price - boxes * price * (100 - discount) // 100
            body = (f'{boxes} boxes of {items} items cost {price} cents per item. Return '
                    f'one item from every box for a refund of {100-discount}% of its price. '
                    'What is the net cost in cents?')
            sig = [family, 'bulk_minus_partial_refund']
        fees = [rng.randint(2, 20) for _ in range(level)]
        expected += sum(fees)
        if fees:
            body = body.rsplit(' What ', 1)[0] + f' Additional fees are {fees} cents. What is the final net cost in cents?'
        sig.append(level)
        facts = {'boxes': boxes, 'items': items, 'price': price, 'discount': discount, 'fees': fees}
    # Owner rejection applies only to combinatorial random templates.
    if family in {'boolean_logic', 'state_machine', 'sequence'} or (family == 'arithmetic' and level):
        if owner(sig) != split:
            return None
    if Fraction(expected).denominator != 1:
        raise ValueError('Non-integral ground truth')
    semantic = digest({'family': family, 'facts': facts})
    task_id = f'{split}-{family}-v2-{semantic[:16]}'
    return {'sample_id': task_id, 'task_id': task_id, 'seed_key': task_id,
            'dataset': VERSION, 'content_version': VERSION, 'split': split, 'family': family,
            'body': body, 'prompt': body, 'expected': str(int(expected)), 'facts': facts,
            'difficulty': DIFFICULTIES[level], 'generator_seed': SEEDS[split],
            'structural_signature': sig, 'semantic_fingerprint': semantic,
            'provenance': {'source_id': task_id, 'prompt_origin': 'algorithmic',
                           'response_origin': 'algorithmic', 'license': 'Apache-2.0',
                           'training_permitted': split == 'train', 'generator_version': VERSION}}


def generate(split, per_bin=2):
    if split not in SPLITS or per_bin < 1:
        raise ValueError('Valid split and positive per-bin count required')
    rng, rows, seen = random.Random(SEEDS[split]), [], set()
    for family in FAMILIES:
        for level in range(len(DIFFICULTIES)):
            count = 0
            for _ in range(10000):
                row = make_task(rng, split, family, level)
                if row is None or row['semantic_fingerprint'] in seen:
                    continue
                rows.append(row)
                seen.add(row['semantic_fingerprint'])
                count += 1
                if count == per_bin:
                    break
            else:
                raise RuntimeError(f'Insufficient task space: {split}/{family}/{level}')
    rng.shuffle(rows)
    return rows


def check_structural_contamination(pools):
    check_contamination(pools)
    owners, fingerprints = {}, {}
    for split, rows in pools.items():
        for row in rows:
            key = digest(row['structural_signature'])
            if key in owners and owners[key] != split:
                raise ValueError('Cross-split structural template collision')
            owners[key] = split
            fp = row['semantic_fingerprint']
            if fp in fingerprints:
                raise ValueError('Repeated semantic task')
            fingerprints[fp] = split
    return {'cross_split_structure_collisions': 0, 'semantic_duplicates': 0,
            'unique_structures': {s: len({digest(r['structural_signature']) for r in rows})
                                  for s, rows in pools.items()},
            'limitation': 'Signatures exclude numerical values/variable aliases where applicable, but do not prove conceptual independence. Algebra/word templates and ordering sizes are intentionally held out; difficulty is structural, not observed accuracy.'}


def build_pools(destination, per_bin=2):
    path = Path(destination)
    if path.exists() and any(path.iterdir()):
        raise ValueError('Refuse to overwrite an existing dataset')
    path.mkdir(parents=True, exist_ok=True)
    pools = {s: generate(s, per_bin) for s in SPLITS}
    checks = check_structural_contamination(pools)
    for split, rows in pools.items():
        write_jsonl(path / f'{split}.jsonl', rows)
    manifest = {'version': VERSION, 'seeds': SEEDS, 'per_bin': per_bin,
                'counts': {s: len(rows) for s, rows in pools.items()},
                'hashes': {s: digest(rows) for s, rows in pools.items()}, 'checks': checks,
                'test_policy': 'No model evaluation or trajectory selection on TEST.'}
    write_json(path / 'manifest.json', manifest)
    return pools
