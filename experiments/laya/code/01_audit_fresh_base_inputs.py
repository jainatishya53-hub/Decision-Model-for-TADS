#!/usr/bin/env python3

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import time


FIELDS = ("Task", "Topic", "Style", "Audience")
CLASSES_PER_FIELD = 100

cluster_pattern = re.compile(
    r"\b(Task|Topic|Style|Audience)_(\d+)\b"
)

exp_root = Path(os.environ["EXP_ROOT"]).resolve()
paper_work = Path(os.environ["PAPER_WORK"]).resolve()
restart_root = Path(os.environ["BASE_RESTART_ROOT"]).resolve()
laya_source_repo = Path(
    os.environ["LAYA_SOURCE_REPO"]
).resolve()
base_checkpoint = Path(
    os.environ["LAYA_BASE_CHECKPOINT"]
).resolve()

raw_target_path = (
    paper_work
    / "data"
    / "cluster_ids.json"
)

description_path = (
    exp_root
    / "data"
    / "description_aware"
    / "cluster_descriptions.json"
)

report_path = (
    restart_root
    / "metadata"
    / "01_fresh_base_input_audit.json"
)

for required_path in (
    raw_target_path,
    description_path,
):
    assert required_path.is_file(), (
        f"Required file is missing: {required_path}"
    )

assert laya_source_repo.is_dir(), (
    f"Laya source repository is missing: {laya_source_repo}"
)

assert base_checkpoint.is_dir(), (
    f"Base Laya checkpoint is missing: {base_checkpoint}"
)

old_finetuned_checkpoints = [
    (
        exp_root
        / "models"
        / "laya_four_head_run_01"
        / "epoch_4"
        / "model.safetensors"
    ),
    (
        exp_root
        / "models"
        / "laya_description_aware_run_01"
        / "epoch_4"
        / "model.safetensors"
    ),
]

for old_path in old_finetuned_checkpoints:
    assert old_path.resolve() != base_checkpoint, (
        "The selected base checkpoint is an old "
        f"fine-tuned checkpoint: {old_path}"
    )


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def exact_cluster_id(value):
    if not isinstance(value, str):
        return None

    match = cluster_pattern.fullmatch(
        value.strip()
    )

    if match is None:
        return None

    field = match.group(1)
    index = int(match.group(2))

    assert 0 <= index < CLASSES_PER_FIELD, (
        f"Invalid cluster index in {value}"
    )

    return f"{field}_{index}"


def find_cluster_ids(node):
    found = []

    if isinstance(node, dict):
        for key, value in node.items():
            key_id = exact_cluster_id(key)

            if key_id is not None:
                found.append(key_id)

            found.extend(find_cluster_ids(value))

    elif isinstance(node, list):
        for value in node:
            found.extend(find_cluster_ids(value))

    elif isinstance(node, str):
        for field, index_text in cluster_pattern.findall(
            node
        ):
            index = int(index_text)

            if 0 <= index < CLASSES_PER_FIELD:
                found.append(f"{field}_{index}")

    return found


def find_cluster_paths(node, path="$"):
    paths = []

    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}"

            key_id = exact_cluster_id(key)

            if key_id is not None:
                paths.append(
                    {
                        "cluster_id": key_id,
                        "path": child_path,
                    }
                )

            paths.extend(
                find_cluster_paths(value, child_path)
            )

    elif isinstance(node, list):
        for index, value in enumerate(node):
            paths.extend(
                find_cluster_paths(
                    value,
                    f"{path}[{index}]",
                )
            )

    elif isinstance(node, str):
        for field, index_text in cluster_pattern.findall(
            node
        ):
            index = int(index_text)

            if 0 <= index < CLASSES_PER_FIELD:
                paths.append(
                    {
                        "cluster_id": f"{field}_{index}",
                        "path": path,
                    }
                )

    return paths


def normalize_description_catalog(payload):
    catalog = {}

    def register(cluster_id, value):
        normalized_id = exact_cluster_id(cluster_id)

        if normalized_id is None:
            return

        if isinstance(value, str):
            description = value.strip()
            title = ""
            keywords = []

        elif isinstance(value, dict):
            description = ""

            for key in (
                "description",
                "summary",
                "meaning",
                "definition",
                "text",
            ):
                candidate = value.get(key)

                if isinstance(candidate, str):
                    candidate = candidate.strip()

                    if candidate:
                        description = candidate
                        break

            title = value.get("title", "")

            if not isinstance(title, str):
                title = ""

            keywords = value.get("keywords", [])

            if isinstance(keywords, str):
                keywords = [
                    item.strip()
                    for item in re.split(
                        r"[_;,|]",
                        keywords,
                    )
                    if item.strip()
                ]

            if not isinstance(keywords, list):
                keywords = []

        else:
            return

        if description:
            catalog[normalized_id] = {
                "cluster_id": normalized_id,
                "field": normalized_id.split("_")[0],
                "index": int(
                    normalized_id.rsplit("_", 1)[1]
                ),
                "title": title.strip(),
                "description": description,
                "keywords": keywords,
            }

    def walk(node, inherited_id=None):
        if isinstance(node, dict):
            local_id = inherited_id

            for key in (
                "cluster_id",
                "anchor_id",
                "tag_id",
                "id",
            ):
                candidate = exact_cluster_id(
                    node.get(key)
                )

                if candidate is not None:
                    local_id = candidate
                    break

            if local_id is not None:
                register(local_id, node)

            for key, value in node.items():
                key_id = exact_cluster_id(key)

                walk(
                    value,
                    key_id
                    if key_id is not None
                    else local_id,
                )

        elif isinstance(node, list):
            for value in node:
                walk(value, inherited_id)

        elif (
            inherited_id is not None
            and isinstance(node, str)
        ):
            register(inherited_id, node)

    if isinstance(payload, dict):
        for wrapper_key in (
            "clusters",
            "cluster_descriptions",
            "descriptions",
            "anchors",
        ):
            if wrapper_key in payload:
                walk(payload[wrapper_key])
                break
        else:
            walk(payload)
    else:
        walk(payload)

    return catalog


start_time = time.time()

target_rows = read_json(raw_target_path)
description_payload = read_json(description_path)

assert isinstance(target_rows, list), (
    "cluster_ids.json must contain a JSON list"
)

assert len(target_rows) > 0, (
    "The target dataset is empty"
)

catalog = normalize_description_catalog(
    description_payload
)

expected_ids = {
    f"{field}_{index}"
    for field in FIELDS
    for index in range(CLASSES_PER_FIELD)
}

catalog_ids = set(catalog)
missing_descriptions = sorted(
    expected_ids - catalog_ids
)
unexpected_descriptions = sorted(
    catalog_ids - expected_ids
)

assert not missing_descriptions, (
    "Descriptions are missing for these clusters: "
    f"{missing_descriptions}"
)

assert not unexpected_descriptions, (
    "Unexpected cluster descriptions found: "
    f"{unexpected_descriptions}"
)

observed_ids = set()
field_occurrences = Counter()
rows_with_field = Counter()
cardinality_histograms = {
    field: Counter()
    for field in FIELDS
}

sample_cluster_paths = []

for position, row in enumerate(target_rows):
    assert isinstance(row, dict), (
        f"Target row {position} is not an object"
    )

    row_ids = find_cluster_ids(row)

    observed_ids.update(row_ids)

    row_counter = Counter(
        cluster_id.split("_")[0]
        for cluster_id in row_ids
    )

    for field in FIELDS:
        count = row_counter[field]
        cardinality_histograms[field][count] += 1
        field_occurrences[field] += count

        if count > 0:
            rows_with_field[field] += 1

    if position < 3:
        sample_cluster_paths.append(
            {
                "target_position": position,
                "top_level_keys": list(row.keys()),
                "cluster_paths": find_cluster_paths(row),
            }
        )

unassigned_ids = sorted(
    expected_ids - observed_ids
)

unassigned_by_field = {
    field: sorted(
        cluster_id
        for cluster_id in unassigned_ids
        if cluster_id.startswith(f"{field}_")
    )
    for field in FIELDS
}

observed_by_field = {
    field: sorted(
        cluster_id
        for cluster_id in observed_ids
        if cluster_id.startswith(f"{field}_")
    )
    for field in FIELDS
}

base_safetensors = sorted(
    base_checkpoint.rglob("*.safetensors")
)

base_configs = sorted(
    base_checkpoint.rglob("config.json")
)

assert base_safetensors, (
    "No safetensors checkpoint was found inside "
    f"{base_checkpoint}"
)

assert base_configs, (
    "No config.json was found inside "
    f"{base_checkpoint}"
)

try:
    laya_source_commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(laya_source_repo),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
except Exception:
    laya_source_commit = "unavailable"

base_files = []

for path in base_safetensors:
    base_files.append(
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )

report = {
    "status": "PASSED",
    "purpose": (
        "Verify inputs for a fresh four-head Laya "
        "fine-tuning run"
    ),
    "fresh_start_policy": {
        "initial_model": str(base_checkpoint),
        "resume_training": False,
        "reuse_old_optimizer": False,
        "classification_heads": {
            field: CLASSES_PER_FIELD
            for field in FIELDS
        },
        "forbidden_finetuned_checkpoints": [
            {
                "path": str(path),
                "exists": path.is_file(),
            }
            for path in old_finetuned_checkpoints
        ],
    },
    "laya_source": {
        "repository": str(laya_source_repo),
        "commit": laya_source_commit,
    },
    "base_checkpoint": {
        "directory": str(base_checkpoint),
        "safetensors": base_files,
        "config_files": [
            str(path)
            for path in base_configs
        ],
    },
    "target_dataset": {
        "path": str(raw_target_path),
        "rows": len(target_rows),
        "first_row_keys": list(
            target_rows[0].keys()
        ),
        "rows_with_field": dict(rows_with_field),
        "assignment_occurrences": dict(
            field_occurrences
        ),
        "cardinality_histograms": {
            field: {
                str(count): frequency
                for count, frequency
                in sorted(histogram.items())
            }
            for field, histogram
            in cardinality_histograms.items()
        },
        "observed_cluster_counts": {
            field: len(observed_by_field[field])
            for field in FIELDS
        },
        "unassigned_clusters": (
            unassigned_by_field
        ),
        "sample_cluster_paths": (
            sample_cluster_paths
        ),
    },
    "description_catalog": {
        "path": str(description_path),
        "clusters": len(catalog),
        "field_counts": {
            field: sum(
                value["field"] == field
                for value in catalog.values()
            )
            for field in FIELDS
        },
        "missing_descriptions": (
            missing_descriptions
        ),
    },
    "elapsed_seconds": time.time() - start_time,
}

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("=" * 72)
print("FRESH BASE LAYA INPUT AUDIT")
print("=" * 72)
print("Status: PASSED")
print("Target rows:", len(target_rows))
print("Description records:", len(catalog))
print("Laya source commit:", laya_source_commit)
print("Base checkpoint:", base_checkpoint)

print("\nObserved target clusters:")

for field in FIELDS:
    print(
        f"  {field}: "
        f"{len(observed_by_field[field])}/100"
    )

print("\nClusters not assigned to any target row:")

for field in FIELDS:
    missing = unassigned_by_field[field]

    print(
        f"  {field}: {len(missing)} "
        f"{missing}"
    )

print("\nOld fine-tuned checkpoints are forbidden:")

for old_path in old_finetuned_checkpoints:
    print(
        " ",
        old_path,
        "exists=",
        old_path.is_file(),
    )

print("\nThe next training run must start from:")
print(base_checkpoint)

print("\nReport:", report_path)
print("\nFRESH BASE LAYA INPUT AUDIT PASSED")
