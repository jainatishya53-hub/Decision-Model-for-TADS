#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import stat
import time


FIELDS = (
    "Task",
    "Topic",
    "Style",
    "Audience",
)

EXPECTED_SPLIT_COUNTS = {
    "train": 3978,
    "validation": 850,
    "test": 858,
}

root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

reference_root = Path(
    os.environ["LAYA_REFERENCE_ROOT"]
).resolve()

destination = (
    root
    / "data"
    / "target_supervision"
    / "exact_laya_description_aware_v1"
)

report_path = (
    root
    / "metadata"
    / "target_supervision"
    / "08_exact_laya_target_supervision.json"
)

destination.mkdir(
    parents=True,
    exist_ok=True,
)

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

training_config_path = (
    reference_root
    / "configs"
    / "05_fresh_base_description_training.json"
)

training_config = json.loads(
    training_config_path.read_text(
        encoding="utf-8"
    )
)

source_paths = {
    "complete_target.json": (
        reference_root
        / "data"
        / "complete_target_with_cluster_descriptions.json"
    ),
    "train_complete.json": (
        reference_root
        / "data"
        / "splits"
        / "train_complete.json"
    ),
    "train.jsonl": (
        reference_root
        / "data"
        / "splits"
        / "train.jsonl"
    ),
    "validation.jsonl": (
        reference_root
        / "data"
        / "splits"
        / "validation.jsonl"
    ),
    "test.jsonl": (
        reference_root
        / "data"
        / "splits"
        / "test.jsonl"
    ),
    "cluster_descriptions.json": Path(
        training_config[
            "description_catalogue"
        ]
    ),
    "laya_reference_training_config.json": (
        training_config_path
    ),
}

for path in source_paths.values():
    assert path.is_file(), (
        f"Missing reference file: {path}"
    )


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def validate_record(
    record,
    catalog_by_id,
):
    assert isinstance(
        record.get("conversation_id"),
        str,
    )

    assert isinstance(
        record.get("text"),
        str,
    )

    assert record["text"].strip()

    targets = record.get(
        "targets"
    )

    assert isinstance(targets, dict)
    assert set(targets) == set(FIELDS)

    for field in FIELDS:
        target = targets[field]

        required_keys = {
            "cluster_ids",
            "indices",
            "probabilities",
            "descriptions",
            "titles",
        }

        assert required_keys.issubset(
            target
        )

        lengths = {
            len(target[key])
            for key in required_keys
        }

        assert len(lengths) == 1
        target_length = next(
            iter(lengths)
        )

        # Raw target rows normally contain one Task and
        # three Topic/Style/Audience assignments. Identical
        # conversations were consolidated before splitting,
        # so one training record can preserve more than three
        # unique clusters from its original member rows.
        assert 1 <= target_length <= 100

        assert math.isclose(
            sum(
                float(value)
                for value in target[
                    "probabilities"
                ]
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        )

        for (
            cluster_id,
            index,
            probability,
            description,
            title,
        ) in zip(
            target["cluster_ids"],
            target["indices"],
            target["probabilities"],
            target["descriptions"],
            target["titles"],
        ):
            assert cluster_id == (
                f"{field}_{index}"
            )

            assert cluster_id in (
                catalog_by_id
            )

            catalog_record = (
                catalog_by_id[
                    cluster_id
                ]
            )

            assert int(index) == int(
                catalog_record["index"]
            )

            assert field == (
                catalog_record["field"]
            )

            assert description == (
                catalog_record[
                    "refined_description"
                ]
            )

            assert title == (
                catalog_record["title"]
            )

            assert float(
                probability
            ) > 0.0


started = time.time()

train_complete = json.loads(
    source_paths[
        "train_complete.json"
    ].read_text(
        encoding="utf-8"
    )
)

assert list(
    train_complete.keys()
) == [
    "metadata",
    "records",
    "complete_cluster_catalog",
    "unassigned_clusters",
]

training_records = train_complete[
    "records"
]

catalog = train_complete[
    "complete_cluster_catalog"
]

assert len(training_records) == 3978
assert len(catalog) == 400

catalog_counts = Counter(
    item["field"]
    for item in catalog
)

assert catalog_counts == {
    field: 100
    for field in FIELDS
}

catalog_by_id = {
    item["cluster_id"]: item
    for item in catalog
}

assert len(catalog_by_id) == 400

for item in catalog:
    expected_id = (
        f"{item['field']}_{item['index']}"
    )

    assert item[
        "cluster_id"
    ] == expected_id

    assert isinstance(
        item["refined_description"],
        str,
    )

    assert item[
        "refined_description"
    ].strip()

for record in training_records:
    validate_record(
        record,
        catalog_by_id,
    )

split_records = {
    "train": load_jsonl(
        source_paths["train.jsonl"]
    ),
    "validation": load_jsonl(
        source_paths[
            "validation.jsonl"
        ]
    ),
    "test": load_jsonl(
        source_paths["test.jsonl"]
    ),
}

assert {
    split: len(records)
    for split, records in (
        split_records.items()
    )
} == EXPECTED_SPLIT_COUNTS

for records in split_records.values():
    for record in records:
        validate_record(
            record,
            catalog_by_id,
        )

target_cardinality = {
    split: {
        field: Counter(
            len(record["targets"][field]["indices"])
            for record in records
        )
        for field in FIELDS
    }
    for split, records in split_records.items()
}

split_ids = {
    split: {
        record["conversation_id"]
        for record in records
    }
    for split, records in (
        split_records.items()
    )
}

assert not (
    split_ids["train"]
    & split_ids["validation"]
)

assert not (
    split_ids["train"]
    & split_ids["test"]
)

assert not (
    split_ids["validation"]
    & split_ids["test"]
)

assert len(
    set().union(
        *split_ids.values()
    )
) == 5686

complete_training_by_id = {
    record["conversation_id"]: record
    for record in training_records
}

train_jsonl_by_id = {
    record["conversation_id"]: record
    for record in split_records["train"]
}

assert set(
    complete_training_by_id
) == set(
    train_jsonl_by_id
)

for conversation_id in (
    complete_training_by_id
):
    assert (
        complete_training_by_id[
            conversation_id
        ]["targets"]
        ==
        train_jsonl_by_id[
            conversation_id
        ]["targets"]
    )

source_hashes = {
    name: sha256(path)
    for name, path in (
        source_paths.items()
    )
}

copy_actions = {}

for name, source_path in (
    source_paths.items()
):
    target_path = (
        destination
        / name
    )

    if target_path.exists():
        assert target_path.is_file()

        existing_hash = sha256(
            target_path
        )

        assert existing_hash == (
            source_hashes[name]
        ), (
            "Existing pinned file differs "
            f"from source: {target_path}"
        )

        action = "verified_existing"
    else:
        shutil.copy2(
            source_path,
            target_path,
        )

        copied_hash = sha256(
            target_path
        )

        assert copied_hash == (
            source_hashes[name]
        )

        action = "copied"

    target_path.chmod(
        stat.S_IRUSR
        | stat.S_IRGRP
        | stat.S_IROTH
    )

    copy_actions[name] = {
        "action": action,
        "source": str(source_path),
        "pinned": str(target_path),
        "sha256": source_hashes[name],
        "bytes": target_path.stat().st_size,
        "read_only": True,
    }

manifest = {
    "status": "PASSED",
    "dataset_policy": (
        "Exact immutable copy of the latest "
        "description-aware LAYA target supervision"
    ),
    "regenerated": False,
    "reassigned": False,
    "resplit": False,
    "source_reference_root": str(
        reference_root
    ),
    "pinned_directory": str(
        destination
    ),
    "target_rows_before_consolidation": 6279,
    "unique_conversations": 5686,
    "split_counts": (
        EXPECTED_SPLIT_COUNTS
    ),
    "catalog_counts": dict(
        catalog_counts
    ),
    "files": copy_actions,
    "target_cardinality": {
        split: {
            field: {
                str(cardinality): count
                for cardinality, count in sorted(
                    counts.items()
                )
            }
            for field, counts in fields.items()
        }
        for split, fields in target_cardinality.items()
    },
    "target_schema": {
        "raw_assignment_policy": (
            "One Task and three Topic/Style/Audience "
            "assignments per original target row"
        ),
        "consolidation_policy": (
            "Identical conversations preserve the union "
            "of member assignments as a normalized soft "
            "target distribution; no targets are truncated"
        ),
        "preserved_fields": [
            "cluster_ids",
            "indices",
            "probabilities",
            "descriptions",
            "titles",
        ],
    },
    "elapsed_seconds": (
        time.time() - started
    ),
}

manifest_path = (
    destination
    / "MANIFEST.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

manifest_path.chmod(
    stat.S_IRUSR
    | stat.S_IRGRP
    | stat.S_IROTH
)

report_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print("=" * 78)
print("EXACT LAYA TARGET SUPERVISION PINNING")
print("=" * 78)
print("Status: PASSED")
print("Dataset regenerated: NO")
print("Cluster assignments changed: NO")
print("Descriptions changed: NO")
print("Split membership changed: NO")
print("Target rows:", 6279)
print("Unique conversations:", 5686)
print("Splits:", EXPECTED_SPLIT_COUNTS)
print(
    "Catalog:",
    dict(catalog_counts),
)

print("\nTARGET CARDINALITY AFTER CONVERSATION CONSOLIDATION")

for split, fields in target_cardinality.items():
    print(f"  {split}")

    for field, counts in fields.items():
        print(
            f"    {field}: "
            f"{dict(sorted(counts.items()))}"
        )

print("\nNo cluster targets were truncated or discarded.")

print("\nPINNED FILES")

for name, item in (
    copy_actions.items()
):
    print(
        f"  {name}: "
        f"{item['action']} "
        f"sha256={item['sha256']}"
    )

print("\nPinned directory:", destination)
print("Manifest:", manifest_path)
print("Report:", report_path)

print(
    "\nEXACT LAYA TARGET SUPERVISION "
    "PINNING PASSED"
)
