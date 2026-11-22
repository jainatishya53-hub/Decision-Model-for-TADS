#!/usr/bin/env python3

from collections import Counter, defaultdict
from pathlib import Path
import json
import os
import time


FIELDS = ("Task", "Topic", "Style", "Audience")
CLASSES = 100
SPLITS = ("train", "validation", "test")

exp_root = Path(os.environ["EXP_ROOT"]).resolve()
restart_root = Path(
    os.environ["BASE_RESTART_ROOT"]
).resolve()

canonical_path = (
    restart_root
    / "data"
    / "complete_target_with_cluster_descriptions.json"
)

# Only the already-verified conversation membership is reused.
# Labels, descriptions and model weights are not loaded from here.
membership_directory = (
    exp_root
    / "data"
    / "splits_coverage_safe"
)

output_directory = (
    restart_root
    / "data"
    / "splits"
)

report_path = (
    restart_root
    / "metadata"
    / "03_description_training_splits.json"
)


def read_json(path):
    assert path.is_file(), f"Missing file: {path}"

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    assert path.is_file(), f"Missing file: {path}"

    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(
                    f"Invalid JSONL at {path}:"
                    f"{line_number}: {error}"
                ) from error

    return records


def write_jsonl(path, records):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")


def write_json(path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")


def primary_dataset(dataset_counts):
    assert dataset_counts

    return sorted(
        dataset_counts,
        key=lambda dataset: (
            -dataset_counts[dataset],
            dataset,
        ),
    )[0]


start_time = time.time()

canonical = read_json(canonical_path)

samples = canonical.get("samples")
catalog_records = canonical.get(
    "complete_cluster_catalog"
)

assert isinstance(samples, list)
assert len(samples) == 6279

assert isinstance(catalog_records, list)
assert len(catalog_records) == 400

catalog = {
    record["cluster_id"]: record
    for record in catalog_records
}

assert len(catalog) == 400

expected_ids = {
    f"{field}_{index}"
    for field in FIELDS
    for index in range(CLASSES)
}

assert set(catalog) == expected_ids

groups = defaultdict(list)

for sample in samples:
    conversation_id = sample["conversation_id"]
    groups[conversation_id].append(sample)

assert len(groups) == 5686


def consolidate_group(conversation_id, members):
    first = members[0]

    first_messages = first["messages"]
    first_text = first["model_input_text"]

    for member in members[1:]:
        assert member["messages"] == first_messages, (
            "Conversation hash collision or inconsistent "
            f"messages: {conversation_id}"
        )

        assert member["model_input_text"] == first_text

    dataset_counts = Counter(
        str(member["dataset"])
        for member in members
    )

    consolidated_targets = {}

    for field in FIELDS:
        probability_totals = Counter()

        for member in members:
            member_targets = member[
                "training_targets"
            ][field]

            member_probability_sum = sum(
                float(item["probability"])
                for item in member_targets
            )

            assert abs(
                member_probability_sum - 1.0
            ) < 1e-8

            for item in member_targets:
                cluster_id = item["cluster_id"]

                assert cluster_id in catalog
                assert (
                    catalog[cluster_id]["field"]
                    == field
                )

                probability_totals[cluster_id] += (
                    float(item["probability"])
                )

        # Average the target distribution across duplicate
        # annotations of the same conversation.
        for cluster_id in probability_totals:
            probability_totals[cluster_id] /= len(
                members
            )

        total_probability = sum(
            probability_totals.values()
        )

        assert total_probability > 0

        normalized = {
            cluster_id: probability / total_probability
            for cluster_id, probability
            in probability_totals.items()
        }

        ordered_ids = sorted(
            normalized,
            key=lambda cluster_id: (
                catalog[cluster_id]["index"]
            ),
        )

        probabilities = [
            normalized[cluster_id]
            for cluster_id in ordered_ids
        ]

        assert abs(sum(probabilities) - 1.0) < 1e-8

        consolidated_targets[field] = {
            "cluster_ids": ordered_ids,
            "indices": [
                catalog[cluster_id]["index"]
                for cluster_id in ordered_ids
            ],
            "probabilities": probabilities,
            "descriptions": [
                catalog[cluster_id][
                    "refined_description"
                ]
                for cluster_id in ordered_ids
            ],
            "titles": [
                catalog[cluster_id]["title"]
                for cluster_id in ordered_ids
            ],
        }

    return {
        "conversation_id": conversation_id,
        "text": first_text,
        "messages": first_messages,
        "member_count": len(members),
        "members": [
            {
                "target_position": member[
                    "target_position"
                ],
                "dataset": member["dataset"],
                "source_id": member["source_id"],
            }
            for member in members
        ],
        "dataset_counts": dict(
            sorted(dataset_counts.items())
        ),
        "primary_dataset": primary_dataset(
            dataset_counts
        ),
        "targets": consolidated_targets,
    }


consolidated_records = {
    conversation_id: consolidate_group(
        conversation_id,
        members,
    )
    for conversation_id, members in groups.items()
}

# Build a stable lookup using original dataset and source ID.
# This avoids depending on the exact conversation-hash recipe.
member_to_conversation = {}

for conversation_id, record in consolidated_records.items():
    for member in record["members"]:
        member_key = (
            str(member["dataset"]),
            str(member["source_id"]),
            int(member["target_position"]),
        )

        previous = member_to_conversation.get(
            member_key
        )

        assert (
            previous is None
            or previous == conversation_id
        ), (
            "One source member maps to multiple "
            f"conversations: {member_key}"
        )

        member_to_conversation[
            member_key
        ] = conversation_id

membership = {}
seen_membership = set()

for split in SPLITS:
    old_path = (
        membership_directory
        / f"{split}.jsonl"
    )

    old_records = read_jsonl(old_path)

    ordered_conversation_ids = []

    for record_position, record in enumerate(
        old_records
    ):
        old_conversation_id = record.get(
            "conversation_id"
        )

        # Direct matching is retained when both builders
        # happen to use the same conversation hash.
        if old_conversation_id in consolidated_records:
            conversation_id = old_conversation_id

        else:
            candidate_ids = set()

            old_members = record.get(
                "members",
                [],
            )

            assert isinstance(old_members, list), (
                f"Invalid members in {old_path}, "
                f"record {record_position}"
            )

            for member in old_members:
                if not isinstance(member, dict):
                    continue

                dataset = member.get("dataset")
                source_id = member.get("source_id")

                # Earlier split files call this value
                # "position"; the new canonical file calls
                # the same value "target_position".
                member_position = member.get(
                    "position",
                    member.get("target_position"),
                )

                if (
                    dataset is None
                    or source_id is None
                    or member_position is None
                ):
                    continue

                member_key = (
                    str(dataset),
                    str(source_id),
                    int(member_position),
                )

                candidate = (
                    member_to_conversation.get(
                        member_key
                    )
                )

                if candidate is not None:
                    candidate_ids.add(candidate)

            assert len(candidate_ids) == 1, (
                "Could not uniquely match old split "
                f"record {record_position} from "
                f"{old_path}. Old conversation ID: "
                f"{old_conversation_id}; candidates: "
                f"{sorted(candidate_ids)}"
            )

            conversation_id = next(
                iter(candidate_ids)
            )

        assert conversation_id not in seen_membership, (
            "Conversation leakage across splits: "
            f"{conversation_id}"
        )

        seen_membership.add(conversation_id)

        ordered_conversation_ids.append(
            conversation_id
        )

    membership[split] = ordered_conversation_ids

assert seen_membership == set(consolidated_records), (
    "The split membership does not cover exactly "
    "all 5,686 unique conversations"
)

split_records = {
    split: [
        consolidated_records[conversation_id]
        for conversation_id
        in membership[split]
    ]
    for split in SPLITS
}

assert len(split_records["train"]) == 3978
assert len(split_records["validation"]) == 850
assert len(split_records["test"]) == 858

split_reports = {}

for split in SPLITS:
    records = split_records[split]

    observed = {
        field: set()
        for field in FIELDS
    }

    dataset_counts = Counter()
    original_member_rows = 0

    for record in records:
        dataset_counts.update(
            record["dataset_counts"]
        )

        original_member_rows += record[
            "member_count"
        ]

        for field in FIELDS:
            observed[field].update(
                record["targets"][field][
                    "cluster_ids"
                ]
            )

    unassigned = {
        field: [
            catalog[cluster_id]
            for cluster_id in sorted(
                expected_ids - observed[field],
                key=lambda value: (
                    catalog[value]["index"]
                ),
            )
            if catalog[cluster_id]["field"] == field
        ]
        for field in FIELDS
    }

    # Efficient model-training file.
    jsonl_path = (
        output_directory
        / f"{split}.jsonl"
    )

    write_jsonl(jsonl_path, records)

    # Auditable complete version. The final section always
    # contains descriptions for labels absent from this split.
    complete_path = (
        output_directory
        / f"{split}_complete.json"
    )

    complete_payload = {
        "metadata": {
            "split": split,
            "records": len(records),
            "original_target_rows": (
                original_member_rows
            ),
            "base_model_policy": {
                "checkpoint": os.environ[
                    "LAYA_BASE_CHECKPOINT"
                ],
                "fresh_training": True,
                "resume_from_finetuned_model": False,
            },
            "membership_source": str(
                membership_directory
                / f"{split}.jsonl"
            ),
            "membership_source_use": (
                "Conversation IDs only"
            ),
            "description_source": str(
                canonical_path
            ),
        },
        "records": records,
        "complete_cluster_catalog": (
            catalog_records
        ),

        # Deliberately retained as the final JSON section.
        "unassigned_clusters": unassigned,
    }

    write_json(
        complete_path,
        complete_payload,
    )

    split_reports[split] = {
        "records": len(records),
        "original_target_rows": (
            original_member_rows
        ),
        "dataset_counts": dict(
            sorted(dataset_counts.items())
        ),
        "observed_clusters": {
            field: len(observed[field])
            for field in FIELDS
        },
        "unassigned_clusters": {
            field: len(unassigned[field])
            for field in FIELDS
        },
        "jsonl": str(jsonl_path),
        "complete_json": str(
            complete_path
        ),
    }

# Every one of the 400 classes must be represented
# by at least one training conversation.
for field in FIELDS:
    assert (
        split_reports["train"][
            "observed_clusters"
        ][field]
        == 100
    ), (
        f"Training split does not cover all "
        f"{field} clusters"
    )

report = {
    "status": "PASSED",
    "method": (
        "Group-safe split using verified conversation "
        "membership; duplicate annotations are averaged "
        "into soft class targets"
    ),
    "canonical_source": str(canonical_path),
    "fresh_base_checkpoint": os.environ[
        "LAYA_BASE_CHECKPOINT"
    ],
    "old_finetuned_checkpoint_used": False,
    "unique_conversations": len(
        consolidated_records
    ),
    "original_target_rows": len(samples),
    "splits": split_reports,
    "leakage_checks": {
        "pairwise_disjoint": True,
        "complete_union": True,
    },
    "elapsed_seconds": time.time() - start_time,
}

write_json(report_path, report)

print("=" * 72)
print("DESCRIPTION-AWARE TRAINING SPLITS")
print("=" * 72)
print("Status: PASSED")
print("Original target rows:", len(samples))
print(
    "Unique conversations:",
    len(consolidated_records),
)

for split in SPLITS:
    values = split_reports[split]

    print(f"\n{split.upper()}")
    print("  Records:", values["records"])
    print(
        "  Original member rows:",
        values["original_target_rows"],
    )
    print(
        "  Dataset counts:",
        values["dataset_counts"],
    )
    print(
        "  Observed clusters:",
        values["observed_clusters"],
    )
    print(
        "  Unassigned clusters:",
        values["unassigned_clusters"],
    )
    print("  JSONL:", values["jsonl"])
    print(
        "  Complete JSON:",
        values["complete_json"],
    )

print("\nFresh base checkpoint:")
print(os.environ["LAYA_BASE_CHECKPOINT"])
print("\nOld fine-tuned checkpoint used: False")
print("\nReport:", report_path)
print(
    "\nDESCRIPTION-AWARE TRAINING "
    "SPLIT BUILD PASSED"
)
