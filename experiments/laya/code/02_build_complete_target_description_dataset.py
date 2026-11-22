#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import re
import statistics
import time


FIELDS = ("Task", "Topic", "Style", "Audience")
CLASSES = 100

CLUSTER_PATTERN = re.compile(
    r"^(Task|Topic|Style|Audience)_(\d+)$"
)

exp_root = Path(os.environ["EXP_ROOT"]).resolve()
paper_work = Path(os.environ["PAPER_WORK"]).resolve()
restart_root = Path(
    os.environ["BASE_RESTART_ROOT"]
).resolve()

target_path = (
    paper_work
    / "data"
    / "cluster_ids.json"
)

raw_cluster_text_path = (
    paper_work
    / "data"
    / "tag_id_mapping.json"
)

keyword_path = (
    paper_work
    / "data"
    / "extracted_keywords.json"
)

refined_description_path = (
    exp_root
    / "data"
    / "description_aware"
    / "cluster_descriptions.json"
)

description_audit_path = (
    exp_root
    / "metadata"
    / "description_aware"
    / "13_cluster_description_audit.json"
)

fresh_base_audit_path = (
    restart_root
    / "metadata"
    / "01_fresh_base_input_audit.json"
)

output_path = (
    restart_root
    / "data"
    / "complete_target_with_cluster_descriptions.json"
)

report_path = (
    restart_root
    / "metadata"
    / "02_complete_target_dataset_build.json"
)


def read_json(path):
    assert path.is_file(), f"Missing required file: {path}"

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cluster_sort_key(cluster_id):
    match = CLUSTER_PATTERN.fullmatch(cluster_id)

    assert match is not None, (
        f"Invalid cluster ID: {cluster_id}"
    )

    return (
        FIELDS.index(match.group(1)),
        int(match.group(2)),
    )


def normalize_cluster_id(value):
    if not isinstance(value, str):
        return None

    value = value.strip()
    match = CLUSTER_PATTERN.fullmatch(value)

    if match is None:
        return None

    field = match.group(1)
    index = int(match.group(2))

    assert 0 <= index < CLASSES, (
        f"Cluster index outside 0–99: {value}"
    )

    return f"{field}_{index}"


def flatten_messages(messages):
    assert isinstance(messages, list), (
        "messages must be a list"
    )

    flattened = []

    for position, message in enumerate(messages):
        assert isinstance(message, dict), (
            f"Message {position} is not an object"
        )

        role = str(
            message.get("role", "unknown")
        ).strip()

        content = str(
            message.get("content", "")
        ).strip()

        flattened.append(f"{role}: {content}")

    return "\n".join(flattened).strip()


def conversation_hash(messages):
    canonical = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def normalize_keywords(value):
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        return [
            item.strip()
            for item in re.split(
                r"[_;,|]",
                value,
            )
            if item.strip()
        ]

    return []


def normalize_refined_descriptions(payload):
    catalog = {}

    def register(cluster_id, value):
        cluster_id = normalize_cluster_id(cluster_id)

        if cluster_id is None:
            return

        current = catalog.get(
            cluster_id,
            {
                "cluster_id": cluster_id,
                "title": "",
                "description": "",
                "keywords": [],
            },
        )

        if isinstance(value, str):
            text = value.strip()

            if text and not current["description"]:
                current["description"] = text

        elif isinstance(value, dict):
            for key in (
                "description",
                "summary",
                "meaning",
                "definition",
                "text",
            ):
                candidate = value.get(key)

                if (
                    isinstance(candidate, str)
                    and candidate.strip()
                ):
                    current["description"] = (
                        candidate.strip()
                    )
                    break

            title = value.get("title")

            if (
                isinstance(title, str)
                and title.strip()
            ):
                current["title"] = title.strip()

            keywords = normalize_keywords(
                value.get("keywords", [])
            )

            if keywords:
                current["keywords"] = keywords

        catalog[cluster_id] = current

    def walk(node, inherited_id=None):
        if isinstance(node, dict):
            local_id = inherited_id

            for id_key in (
                "cluster_id",
                "anchor_id",
                "tag_id",
                "id",
            ):
                candidate = normalize_cluster_id(
                    node.get(id_key)
                )

                if candidate is not None:
                    local_id = candidate
                    break

            if local_id is not None:
                register(local_id, node)

            for key, value in node.items():
                key_id = normalize_cluster_id(key)

                if key_id is not None:
                    register(key_id, value)
                    walk(value, key_id)

                elif isinstance(value, (dict, list)):
                    walk(value, local_id)

        elif isinstance(node, list):
            for value in node:
                walk(value, inherited_id)

        elif (
            inherited_id is not None
            and isinstance(node, str)
        ):
            register(inherited_id, node)

    if isinstance(payload, dict):
        wrapper_found = False

        for wrapper in (
            "clusters",
            "cluster_descriptions",
            "descriptions",
            "anchors",
        ):
            if wrapper in payload:
                walk(payload[wrapper])
                wrapper_found = True
                break

        if not wrapper_found:
            walk(payload)

    elif isinstance(payload, list):
        walk(payload)

    return catalog


def normalize_simple_mapping(payload):
    normalized = {}

    assert isinstance(payload, dict), (
        "Expected a dictionary mapping cluster IDs"
    )

    for key, value in payload.items():
        cluster_id = normalize_cluster_id(key)

        if cluster_id is not None:
            normalized[cluster_id] = value

    return normalized


def field_values(generated_content, field):
    assert field in generated_content, (
        f"Missing field {field} in generated_content"
    )

    value = generated_content[field]

    if isinstance(value, list):
        values = value
    else:
        values = [value]

    cluster_ids = []

    for slot, item in enumerate(values):
        cluster_id = normalize_cluster_id(item)

        assert cluster_id is not None, (
            f"Invalid {field} cluster at slot "
            f"{slot}: {item!r}"
        )

        actual_field = cluster_id.split("_", 1)[0]

        assert actual_field == field, (
            f"{field} contains cluster "
            f"from another field: {cluster_id}"
        )

        cluster_ids.append(cluster_id)

    assert cluster_ids, (
        f"No cluster IDs found for {field}"
    )

    return cluster_ids


start_time = time.time()

target_rows = read_json(target_path)
raw_cluster_mapping = normalize_simple_mapping(
    read_json(raw_cluster_text_path)
)
keyword_mapping = normalize_simple_mapping(
    read_json(keyword_path)
)

description_payload = read_json(
    refined_description_path
)

description_catalog = (
    normalize_refined_descriptions(
        description_payload
    )
)

fresh_base_audit = read_json(
    fresh_base_audit_path
)

assert fresh_base_audit["status"] == "PASSED", (
    "Fresh base Laya audit did not pass"
)

if description_audit_path.is_file():
    description_audit = read_json(
        description_audit_path
    )

    assert (
        description_audit.get("status") == "PASSED"
    ), "Qwen description audit did not pass"
else:
    description_audit = {
        "status": "not_available"
    }

expected_cluster_ids = {
    f"{field}_{index}"
    for field in FIELDS
    for index in range(CLASSES)
}

assert set(description_catalog) == (
    expected_cluster_ids
), (
    "Refined Qwen descriptions do not cover "
    "exactly 400 expected clusters"
)

assert set(raw_cluster_mapping) == (
    expected_cluster_ids
), (
    "Raw concatenated cluster mapping does not "
    "cover exactly 400 clusters"
)

assert isinstance(target_rows, list)
assert len(target_rows) == 6279, (
    f"Expected 6279 target rows, "
    f"found {len(target_rows)}"
)

complete_catalog = {}

description_word_counts = []

for cluster_id in sorted(
    expected_cluster_ids,
    key=cluster_sort_key,
):
    field, index_text = cluster_id.rsplit(
        "_",
        1,
    )

    refined = description_catalog[cluster_id]
    description = refined["description"].strip()

    assert description, (
        f"Empty refined description: {cluster_id}"
    )

    word_count = len(description.split())

    assert 20 <= word_count <= 110, (
        f"Unexpected description length for "
        f"{cluster_id}: {word_count} words"
    )

    description_word_counts.append(word_count)

    keywords = refined["keywords"]

    if not keywords:
        keywords = normalize_keywords(
            keyword_mapping.get(cluster_id, [])
        )

    raw_value = raw_cluster_mapping[cluster_id]

    if not isinstance(raw_value, str):
        raw_value = json.dumps(
            raw_value,
            ensure_ascii=False,
            sort_keys=True,
        )

    complete_catalog[cluster_id] = {
        "cluster_id": cluster_id,
        "field": field,
        "index": int(index_text),
        "title": refined["title"],
        "refined_description": description,
        "keywords": keywords,
        "raw_concatenated_cluster_text": (
            raw_value.strip()
        ),
        "training_use": {
            "classification_target": True,
            "semantic_description": True,
            "raw_concatenated_text": False,
        },
    }

samples = []
observed_cluster_ids = set()
assignment_occurrences = Counter()

conversation_ids = Counter()
cardinality_histograms = {
    field: Counter()
    for field in FIELDS
}

for target_position, row in enumerate(target_rows):
    assert isinstance(row, dict), (
        f"Target row {target_position} "
        "is not an object"
    )

    messages = row.get("messages")

    assert isinstance(messages, list), (
        f"Target row {target_position} "
        "has invalid messages"
    )

    generated_content = row.get(
        "generated_content"
    )

    assert isinstance(generated_content, dict), (
        f"Target row {target_position} has "
        "invalid generated_content"
    )

    conversation_id = conversation_hash(messages)
    conversation_ids[conversation_id] += 1

    assignments = {}
    training_targets = {}

    for field in FIELDS:
        cluster_ids = field_values(
            generated_content,
            field,
        )

        cardinality_histograms[field][
            len(cluster_ids)
        ] += 1

        assignment_records = []

        for slot, cluster_id in enumerate(
            cluster_ids
        ):
            catalog_record = complete_catalog[
                cluster_id
            ]

            assignment_records.append(
                {
                    "slot": slot,
                    "cluster_id": cluster_id,
                    "index": catalog_record[
                        "index"
                    ],
                    "title": catalog_record[
                        "title"
                    ],
                    "description": catalog_record[
                        "refined_description"
                    ],
                    "keywords": catalog_record[
                        "keywords"
                    ],
                }
            )

            observed_cluster_ids.add(cluster_id)
            assignment_occurrences[
                cluster_id
            ] += 1

        assignments[field] = assignment_records

        field_counts = Counter(cluster_ids)
        total_slots = len(cluster_ids)

        training_targets[field] = [
            {
                "cluster_id": cluster_id,
                "index": complete_catalog[
                    cluster_id
                ]["index"],
                "occurrences": occurrences,
                "probability": (
                    occurrences / total_slots
                ),
                "description": (
                    complete_catalog[
                        cluster_id
                    ]["refined_description"]
                ),
            }
            for cluster_id, occurrences
            in field_counts.items()
        ]

        probability_sum = sum(
            item["probability"]
            for item in training_targets[field]
        )

        assert abs(probability_sum - 1.0) < 1e-8

    sample = {
        "target_position": target_position,
        "dataset": row.get("dataset"),
        "source_id": row.get("id"),
        "conversation_id": conversation_id,
        "messages": messages,
        "model_input_text": flatten_messages(
            messages
        ),
        "assignments": assignments,
        "training_targets": training_targets,
        "annotation_metadata": {
            "needs_regeneration": row.get(
                "needs_regeneration"
            ),
            "original_response": row.get(
                "response"
            ),
        },
    }

    samples.append(sample)

unassigned_ids = sorted(
    expected_cluster_ids - observed_cluster_ids,
    key=cluster_sort_key,
)

unassigned_clusters = {
    field: [
        complete_catalog[cluster_id]
        for cluster_id in unassigned_ids
        if cluster_id.startswith(f"{field}_")
    ]
    for field in FIELDS
}

ordered_catalog = [
    complete_catalog[cluster_id]
    for cluster_id in sorted(
        complete_catalog,
        key=cluster_sort_key,
    )
]

output = {
    "metadata": {
        "dataset_name": (
            "Complete target supervision for "
            "fresh base Laya fine-tuning"
        ),
        "target_source": str(target_path),
        "raw_cluster_text_source": str(
            raw_cluster_text_path
        ),
        "refined_description_source": str(
            refined_description_path
        ),
        "description_generator": (
            "Qwen2.5-7B-Instruct"
        ),
        "description_policy": (
            "Use Qwen-refined descriptions for "
            "training. Raw concatenated cluster "
            "text is retained only for provenance."
        ),
        "initial_model_policy": {
            "start_from_base_laya": True,
            "base_checkpoint": (
                os.environ[
                    "LAYA_BASE_CHECKPOINT"
                ]
            ),
            "resume_from_old_finetuned_model": (
                False
            ),
        },
        "target_rows": len(samples),
        "unique_conversations": len(
            conversation_ids
        ),
        "duplicate_rows": (
            len(samples) - len(conversation_ids)
        ),
        "cluster_count": len(
            complete_catalog
        ),
        "description_word_statistics": {
            "minimum": min(
                description_word_counts
            ),
            "maximum": max(
                description_word_counts
            ),
            "mean": statistics.mean(
                description_word_counts
            ),
            "median": statistics.median(
                description_word_counts
            ),
        },
    },

    "samples": samples,

    "complete_cluster_catalog": (
        ordered_catalog
    ),

    # This is deliberately the final JSON section.
    # If a cluster is absent from every target row,
    # its full Qwen description is retained here.
    "unassigned_clusters": (
        unassigned_clusters
    ),
}

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with output_path.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        output,
        handle,
        indent=2,
        ensure_ascii=False,
    )

    handle.write("\n")

report = {
    "status": "PASSED",
    "output": str(output_path),
    "output_size_bytes": (
        output_path.stat().st_size
    ),
    "target_rows": len(samples),
    "unique_conversations": len(
        conversation_ids
    ),
    "duplicate_rows": (
        len(samples) - len(conversation_ids)
    ),
    "cluster_catalog_size": len(
        complete_catalog
    ),
    "observed_cluster_counts": {
        field: sum(
            cluster_id.startswith(f"{field}_")
            for cluster_id
            in observed_cluster_ids
        )
        for field in FIELDS
    },
    "unassigned_cluster_counts": {
        field: len(
            unassigned_clusters[field]
        )
        for field in FIELDS
    },
    "assignment_occurrences": {
        field: sum(
            count
            for cluster_id, count
            in assignment_occurrences.items()
            if cluster_id.startswith(f"{field}_")
        )
        for field in FIELDS
    },
    "cardinality_histograms": {
        field: {
            str(cardinality): frequency
            for cardinality, frequency
            in sorted(
                cardinality_histograms[
                    field
                ].items()
            )
        }
        for field in FIELDS
    },
    "description_word_statistics": {
        "minimum": min(
            description_word_counts
        ),
        "maximum": max(
            description_word_counts
        ),
        "mean": statistics.mean(
            description_word_counts
        ),
        "median": statistics.median(
            description_word_counts
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
print("COMPLETE TARGET DESCRIPTION DATASET BUILD")
print("=" * 72)
print("Status: PASSED")
print("Target rows:", len(samples))
print(
    "Unique conversations:",
    len(conversation_ids),
)
print(
    "Duplicate rows:",
    len(samples) - len(conversation_ids),
)
print(
    "Complete cluster catalog:",
    len(complete_catalog),
)

print("\nObserved clusters:")

for field in FIELDS:
    observed_count = sum(
        cluster_id.startswith(f"{field}_")
        for cluster_id in observed_cluster_ids
    )

    print(
        f"  {field}: {observed_count}/100"
    )

print("\nUnassigned clusters retained at file end:")

for field in FIELDS:
    missing = unassigned_clusters[field]

    print(
        f"  {field}: {len(missing)}"
    )

print("\nDescription lengths:")
print(
    "  Minimum:",
    min(description_word_counts),
)
print(
    "  Maximum:",
    max(description_word_counts),
)
print(
    "  Mean:",
    statistics.mean(
        description_word_counts
    ),
)
print(
    "  Median:",
    statistics.median(
        description_word_counts
    ),
)

print("\nOutput:", output_path)
print("Report:", report_path)
print(
    "\nCOMPLETE TARGET DESCRIPTION "
    "DATASET BUILD PASSED"
)
