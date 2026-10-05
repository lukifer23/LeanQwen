"""Task identity and output policy are separate; v1 stimulus files stay untouched."""

from qwenlean.datasets.procedural import SUFFIX
from qwenlean.utils.io import digest

POLICIES = {
    "P0": "",
    "P1": r"Reason step by step and put the final answer in \boxed{}.",
    "P2": "On the final line, write only the integer answer.",
    "P3": SUFFIX.strip(),
}
POLICY_VERSION = "output_policies_v1"


def task_content(record):
    if "body" in record:
        return dict(record)
    if record.get("dataset") != "procedural-v1" or not record["prompt"].endswith(SUFFIX):
        raise ValueError("Cannot infer body for unknown legacy record")
    body = record["prompt"][: -len(SUFFIX)]
    return {
        **record,
        "body": body,
        "historical_sample_id": record["sample_id"],
        "task_id": "v1-content-"
        + digest({"family": record["family"], "body": body, "facts": record["facts"]})[:20],
        "seed_key": record["sample_id"],
        "content_version": "procedural-v1-content-v2",
    }


def render_task(record, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown output policy")
    task = task_content(record)
    prompt = task["body"] + (" " + POLICIES[policy] if POLICIES[policy] else "")
    rid = task["task_id"] + "-" + policy + "-" + digest(prompt)[:12]
    return {
        **task,
        "sample_id": rid,
        "rendered_prompt_id": rid,
        "prompt": prompt,
        "prompt_policy": policy,
        "prompt_policy_version": POLICY_VERSION,
        "task_content_hash": digest(
            {"body": task["body"], "expected": task["expected"], "facts": task["facts"]}
        ),
    }


def replicate_seed(base_seed, task, replicate):
    if replicate < 0:
        raise ValueError("Replicate index must be nonnegative")
    key = task.get("seed_key", task.get("task_id", task["sample_id"]))
    # Replicate zero retains first-pass seeds for exact P3/control prefix checks.
    if replicate:
        key = f"{key}:replicate:{replicate}"
    return (base_seed + int(digest(key)[:8], 16)) % 2**32
