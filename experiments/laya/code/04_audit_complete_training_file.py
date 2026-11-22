#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import json
import os


FIELDS = ("Task", "Topic", "Style", "Audience")
CLASSES = 100

restart_root = Path(
    os.environ["BASE_RESTART_ROOT"]
).resolve()

training_path = (
    restart_root
    / "data"
    / "splits"
    / "train_complete.json"
)

report_path = (
    restart_root
    / "metadata"
    / "04_complete_training_file_audit.json"
)

assert training_path.is_file(), (
    f"Missing training file: {training_path}"
)

training_data = json.loads(
    training_path.read_text(encoding="utf-8")
)

records = training_data.get("records")
catalog = training_data.get(
    "complete_cluster_catalog"
)
unassigned = training_data.get(
    "unassigned_clusters"
)

assert isinstance(records, list)
assert len(records) == 3978

assert isinstance(catalog, list)
assert len(catalog) == 400

assert isinstance(unassigned, dict)

catalog_by_id = {}
catalog_counts = Counter()
description_word_counts = []
errors = []

for item in catalog:
    cluster_id = item.get("cluster_id")
    field = item.get("field")
    index = item.get("index")
    description = item.get(
        "refined_description"
    )

    if cluster_id in catalog_by_id:
        errors.append(
            f"Duplicate catalog ID: {cluster_id}"
        )

    if field not in FIELDS:
        errors.append(
            f"Invalid field: {field}"
        )
        continue

    if not isinstance(index, int):
        errors.append(
            f"Invalid index: {cluster_id}"
        )
        continue

    expected_id = f"{field}_{index}"

    if cluster_id != expected_id:
        errors.append(
            f"ID mismatch: {cluster_id} "
            f"!= {expected_id}"
        )

    if not 0 <= index < CLASSES:
        errors.append(
            f"Index outside 0–99: {cluster_id}"
        )

    if (
        not isinstance(description, str)
        or not description.strip()
    ):
        errors.append(
            f"Missing description: {cluster_id}"
        )
        continue

    word_count = len(description.split())

    if not 20 <= word_count <= 110:
        errors.append(
            f"Invalid description length: "
            f"{cluster_id} = {word_count}"
        )

    catalog_by_id[cluster_id] = item
    catalog_counts[field] += 1
    description_word_counts.append(word_count)

expected_ids = {
    f"{field}_{index}"
    for field in FIELDS
    for index in range(CLASSES)
}

actual_ids = set(catalog_by_id)

if actual_ids != expected_ids:
    errors.append(
        "Catalog does not contain exactly the "
        "expected 400 cluster IDs"
    )

for field in FIELDS:
    if catalog_counts[field] != 100:
        errors.append(
            f"{field}: expected 100 catalog "
            f"entries, found {catalog_counts[field]}"
        )

observed_training_ids = {
    field: set()
    for field in FIELDS
}

for record_position, record in enumerate(records):
    targets = record.get("targets")

    if not isinstance(targets, dict):
        errors.append(
            f"Record {record_position}: "
            "targets missing"
        )
        continue

    for field in FIELDS:
        target = targets.get(field)

        if not isinstance(target, dict):
            errors.append(
                f"Record {record_position}: "
                f"{field} missing"
            )
            continue

        cluster_ids = target.get(
            "cluster_ids"
        )
        indices = target.get("indices")
        probabilities = target.get(
            "probabilities"
        )
        descriptions = target.get(
            "descriptions"
        )

        if not all(
            isinstance(value, list)
            for value in (
                cluster_ids,
                indices,
                probabilities,
                descriptions,
            )
        ):
            errors.append(
                f"Record {record_position}, "
                f"{field}: invalid target lists"
            )
            continue

        lengths = {
            len(cluster_ids),
            len(indices),
            len(probabilities),
            len(descriptions),
        }

        if len(lengths) != 1:
            errors.append(
                f"Record {record_position}, "
                f"{field}: inconsistent lengths"
            )
            continue

        if not cluster_ids:
            errors.append(
                f"Record {record_position}, "
                f"{field}: empty target"
            )
            continue

        probability_sum = sum(
            float(value)
            for value in probabilities
        )

        if abs(probability_sum - 1.0) > 1e-7:
            errors.append(
                f"Record {record_position}, "
                f"{field}: probability sum "
                f"is {probability_sum}"
            )

        for (
            cluster_id,
            index,
            description,
        ) in zip(
            cluster_ids,
            indices,
            descriptions,
        ):
            if cluster_id not in catalog_by_id:
                errors.append(
                    f"Unknown target ID: {cluster_id}"
                )
                continue

            catalog_record = catalog_by_id[
                cluster_id
            ]

            if index != catalog_record["index"]:
                errors.append(
                    f"Index mismatch: {cluster_id}"
                )

            if (
                description
                != catalog_record[
                    "refined_description"
                ]
            ):
                errors.append(
                    f"Description mismatch: "
                    f"{cluster_id}"
                )

            observed_training_ids[field].add(
                cluster_id
            )

for field in FIELDS:
    observed = observed_training_ids[field]

    expected_field_ids = {
        f"{field}_{index}"
        for index in range(CLASSES)
    }

    missing = sorted(
        expected_field_ids - observed
    )

    if missing:
        errors.append(
            f"{field} missing from training "
            f"records: {missing}"
        )

    if unassigned.get(field):
        errors.append(
            f"{field} unexpectedly has unassigned "
            f"clusters: {unassigned[field]}"
        )

assert not errors, (
    "Training-file audit failed:\n"
    + "\n".join(errors[:100])
)

report = {
    "status": "PASSED",
    "training_file": str(training_path),
    "training_records": len(records),
    "catalog_clusters": len(catalog),
    "catalog_counts": {
        field: catalog_counts[field]
        for field in FIELDS
    },
    "training_coverage": {
        field: len(
            observed_training_ids[field]
        )
        for field in FIELDS
    },
    "unassigned_cluster_counts": {
        field: len(
            unassigned.get(field, [])
        )
        for field in FIELDS
    },
    "description_lengths": {
        "minimum": min(
            description_word_counts
        ),
        "maximum": max(
            description_word_counts
        ),
    },
    "training_policy": {
        "all_400_descriptions_available": True,
        "fresh_base_laya_required": True,
        "old_finetuned_checkpoint_allowed": False,
    },
}

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
print("COMPLETE TRAINING FILE AUDIT")
print("=" * 72)
print("Status: PASSED")
print("Training records:", len(records))
print("Catalog clusters:", len(catalog))

for field in FIELDS:
    print(
        f"{field}: "
        f"{catalog_counts[field]} descriptions, "
        f"{len(observed_training_ids[field])}/100 "
        "clusters observed in training"
    )

print(
    "Description length range:",
    min(description_word_counts),
    "to",
    max(description_word_counts),
    "words",
)

print("\nUnassigned clusters:")

for field in FIELDS:
    print(
        f"  {field}: "
        f"{len(unassigned.get(field, []))}"
    )

print("\nTraining file:", training_path)
print("Report:", report_path)
print(
    "\nALL 400 CLUSTERS AND "
    "DESCRIPTIONS VERIFIED"
)
