"""Deterministic exact-ground-truth problems; independently seeded split pools."""

import random
from pathlib import Path

from qwenlean.datasets.provenance import check_contamination
from qwenlean.utils.io import digest, write_json, write_jsonl

SEEDS = {"train": 11031, "dev": 22061, "test": 33091}
RANGES = {"train": (11, 40), "dev": (41, 70), "test": (71, 100)}
FAMILIES = ["arithmetic", "word_problem", "algebra", "boolean_logic", "ordering", "state_machine"]
SUFFIX = " End your response with FINAL: followed by just the integer answer."


def generate(split, per_family=10):
    rng = random.Random(SEEDS[split])
    lo, hi = RANGES[split]
    records, prompts = [], set()
    for family in FAMILIES:
        count = 0
        while count < per_family:
            a, b, c = [rng.randint(lo, hi) for _ in range(3)]
            if family == "arithmetic":
                prompt = f"Calculate ({a} * {b}) - {c}."
                expected = a * b - c
                facts = {"a": a, "b": b, "c": c}
            elif family == "word_problem":
                p = rng.choice([10, 20, 25, 50])
                price = a * 100
                prompt = (
                    f"A shop has {b} boxes with {c} items each. Each item costs {price} cents. "
                    f"A buyer purchases all items with a {p}% discount. "
                    "What is the total price in cents after the discount?"
                )
                expected = b * c * price * (100 - p) // 100
                facts = {
                    "boxes": b,
                    "items_per_box": c,
                    "price_cents": price,
                    "discount_percent": p,
                }
            elif family == "algebra":
                expected = b
                prompt = f"Solve for the integer x: {a} * x + {c} = {a * b + c}."
                facts = {"coefficient": a, "offset": c, "rhs": a * b + c}
            elif family == "boolean_logic":
                values = [rng.choice([True, False]) for _ in range(4)]
                names = ["A", "B", "C", "D"]
                true_count = sum(
                    [
                        values[0] and not values[1],
                        values[1] or values[2],
                        values[2] != values[3],
                        not (values[0] and values[3]),
                    ]
                )
                salt = a * 100 + b
                prompt = (
                    f"Logic instance {salt}: "
                    + ", ".join(f"{n}={v}" for n, v in zip(names, values))
                    + ". How many of these four expressions are true: "
                    "(A AND NOT B), (B OR C), (C XOR D), NOT(A AND D)?"
                )
                expected = true_count
                facts = {"values": values, "instance": salt}
            elif family == "ordering":
                labels = rng.sample(range(lo * 10, hi * 10), 5)
                order = rng.sample(labels, len(labels))
                constraints = [f"{order[i]} precedes {order[i + 1]}" for i in range(4)]
                rng.shuffle(constraints)
                position = rng.randint(1, 5)
                prompt = (
                    "Five numbered runners finish in distinct positions. "
                    + "; ".join(constraints)
                    + f". Which runner finishes in position {position} (counting from 1)?"
                )
                expected = order[position - 1]
                facts = {"order": order, "position": position}
            else:
                initial, modulus = a, rng.randint(lo, hi)
                ops = [
                    (rng.choice(["add", "multiply", "subtract"]), rng.randint(2, 9))
                    for _ in range(6)
                ]
                state = initial % modulus
                for op, value in ops:
                    if op == "add":
                        state += value
                    elif op == "multiply":
                        state *= value
                    else:
                        state -= value
                    state %= modulus
                prompt = (
                    f"Start with state {initial}, reduce it modulo {modulus}. Apply in order: "
                    + ", ".join(f"{op} {value}" for op, value in ops)
                    + f". After EVERY operation reduce modulo {modulus} to a nonnegative remainder. "
                    "What is the final state?"
                )
                expected = state
                facts = {"initial": initial, "modulus": modulus, "ops": ops}
            prompt += SUFFIX
            if prompt in prompts:
                continue
            prompts.add(prompt)
            sid = f"{split}-{family}-{digest(prompt)[:12]}"
            records.append(
                {
                    "sample_id": sid,
                    "dataset": "procedural-v1",
                    "split": split,
                    "family": family,
                    "prompt": prompt,
                    "expected": str(expected),
                    "facts": facts,
                    "generator_seed": SEEDS[split],
                    "difficulty": "initial_multistep",
                    "provenance": {
                        "source_id": sid,
                        "prompt_origin": "algorithmic",
                        "response_origin": "algorithmic",
                        "license": "Apache-2.0",
                        "training_permitted": split == "train",
                        "generator_version": "procedural-v1",
                    },
                }
            )
            count += 1
    rng.shuffle(records)
    return records


def build_pools(destination, per_family=10):
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    pools = {s: generate(s, per_family) for s in SEEDS}
    check_contamination(pools)
    for split, records in pools.items():
        write_jsonl(path / f"{split}.jsonl", records)
    write_json(
        path / "manifest.json",
        {
            "version": "procedural-v1",
            "seeds": SEEDS,
            "ranges": RANGES,
            "counts": {s: len(r) for s, r in pools.items()},
            "hashes": {s: digest(r) for s, r in pools.items()},
            "test_policy": "Sealed from sampler tuning and training selection; no TEST evaluation in first pass.",
        },
    )
    return pools
