"""Fail-closed provenance checks for any future training-data entrypoint."""

from qwenlean.utils.io import digest

PERMITTED_ORIGINS = {"algorithmic", "qwen_self", "licensed_dataset", "algorithmic_transform"}


def validate_provenance(record, for_training=False):
    p = record["provenance"]
    for field in ["prompt_origin", "response_origin", "license", "training_permitted", "source_id"]:
        if field not in p:
            raise ValueError(f"Missing provenance: {field}")
    if for_training:
        if record["split"] != "train":
            raise ValueError("Only TRAIN records may enter training selection")
        if p["training_permitted"] is not True:
            raise ValueError("Training permission must be explicit")
        if p["prompt_origin"] not in PERMITTED_ORIGINS:
            raise ValueError("Unverified prompt origin")
        if p["response_origin"] not in PERMITTED_ORIGINS:
            raise ValueError("Unverified response origin")
        if not p["license"] or p["license"].lower() in {"unknown", "unverified"}:
            raise ValueError("License must be verified")
        if p["response_origin"] == "algorithmic_transform" and not p.get("parent_ids"):
            raise ValueError("Derived records require original source IDs")


def check_contamination(pools):
    seen_ids, seen_prompts = {}, {}
    for split, records in pools.items():
        for record in records:
            if record["split"] != split:
                raise ValueError("Split mismatch")
            validate_provenance(record)
            key = digest(" ".join(record["prompt"].split()).casefold())
            if key in seen_prompts or record["sample_id"] in seen_ids:
                raise ValueError(f"Duplicate/contamination: {record['sample_id']}")
            seen_prompts[key] = split
            seen_ids[record["sample_id"]] = split
