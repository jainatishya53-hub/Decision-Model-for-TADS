#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import statistics
import sys
import time

from transformers import AutoProcessor


FIELDS = (
    "Task",
    "Topic",
    "Style",
    "Audience",
)

CLASSES = 100
MAXIMUM_LENGTH = 768

QUESTION_TEMPLATE = (
    "Produce the semantic representation used "
    "to determine the {field} cluster."
)

OPTION_TEMPLATE = (
    "Semantic evidence relevant to the "
    "{field} cluster."
)


root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

checkpoint = Path(
    os.environ["PPLX_CHECKPOINT"]
).resolve()

supervision_root = (
    root
    / "data"
    / "target_supervision"
    / "exact_laya_description_aware_v1"
)

cache_root = (
    root
    / "data"
    / "training_cache"
)

report_path = (
    root
    / "metadata"
    / "training_cache"
    / "12_pplx_training_token_cache.json"
)

status_path = (
    cache_root
    / "12_pplx_training_token_cache_status.json"
)

source_paths = {
    "train":
        supervision_root / "train.jsonl",
    "validation":
        supervision_root
        / "validation.jsonl",
    "test":
        supervision_root / "test.jsonl",
}

output_paths = {
    split:
        cache_root
        / f"{split}_pplx_tokens.jsonl"
    for split in source_paths
}

for path in source_paths.values():
    assert path.is_file(), (
        f"Missing input split: {path}"
    )

cache_root.mkdir(
    parents=True,
    exist_ok=True,
)

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

sys.path.insert(
    0,
    str(
        checkpoint
        / "source"
        / "src"
    ),
)

from pplx_decider.protocol import Encoder


def save_json(
    path,
    value,
):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except Exception as error:
                raise RuntimeError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from error

    return records


def make_request(
    text,
):
    return {
        "state": text,
        "questions": {
            field: {
                "type": "choice",
                "instructions":
                    QUESTION_TEMPLATE.format(
                        field=field
                    ),
                "criteria": {
                    "semantic_representation":
                        OPTION_TEMPLATE.format(
                            field=field
                        )
                },
            }
            for field in FIELDS
        },
    }


def validate_targets(
    record,
):
    targets = record.get("targets")

    assert isinstance(targets, dict)

    compact = {}

    for field in FIELDS:
        target = targets[field]

        indices = target["indices"]
        probabilities = target[
            "probabilities"
        ]
        cluster_ids = target[
            "cluster_ids"
        ]

        assert len(indices) >= 1
        assert len(indices) == len(
            probabilities
        )
        assert len(indices) == len(
            cluster_ids
        )

        assert all(
            isinstance(index, int)
            and 0 <= index < CLASSES
            for index in indices
        )

        assert cluster_ids == [
            f"{field}_{index}"
            for index in indices
        ]

        probability_sum = sum(
            float(value)
            for value in probabilities
        )

        assert abs(
            probability_sum - 1.0
        ) < 1e-5

        compact[field] = {
            "indices": indices,
            "probabilities": [
                float(value)
                for value in probabilities
            ],
            "cluster_ids": cluster_ids,
        }

    return compact


def encode_with_truncation(
    encoder,
    tokenizer,
    original_text,
):
    full_request = make_request(
        original_text
    )

    full_encoded = encoder.encode(
        full_request
    )

    original_length = (
        full_encoded.length
    )

    if original_length <= MAXIMUM_LENGTH:
        return (
            full_encoded,
            False,
            original_length,
            len(
                tokenizer.encode(
                    original_text,
                    add_special_tokens=False,
                )
            ),
        )

    raw_token_ids = tokenizer.encode(
        original_text,
        add_special_tokens=False,
    )

    original_state_tokens = len(
        raw_token_ids
    )

    assert original_state_tokens >= 1

    retained = list(
        raw_token_ids
    )

    previous_size = None

    for _ in range(20):
        truncated_text = tokenizer.decode(
            retained,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        encoded = encoder.encode(
            make_request(
                truncated_text
            )
        )

        if encoded.length <= MAXIMUM_LENGTH:
            return (
                encoded,
                True,
                original_length,
                len(retained),
            )

        overflow = (
            encoded.length
            - MAXIMUM_LENGTH
        )

        new_size = max(
            1,
            len(retained)
            - overflow
            - 8,
        )

        if (
            previous_size == new_size
            or new_size >= len(retained)
        ):
            new_size = max(
                1,
                len(retained) - 1,
            )

        previous_size = len(
            retained
        )

        retained = retained[
            :new_size
        ]

    raise RuntimeError(
        "Could not fit record within "
        f"{MAXIMUM_LENGTH} tokens"
    )


with (
    checkpoint
    / "decision_config.json"
).open(
    "r",
    encoding="utf-8",
) as handle:
    decision_config = json.load(handle)

codes = decision_config["codes"]

assert len(codes) == 255

started = time.time()

save_json(
    status_path,
    {
        "status": "STARTING",
        "stage": "loading_processor",
        "checkpoint": str(checkpoint),
    },
)

print(
    "Loading exact PPLX processor...",
    flush=True,
)

processor = AutoProcessor.from_pretrained(
    str(checkpoint),
    local_files_only=True,
)

processor.tokenizer.padding_side = (
    "left"
)

encoder = Encoder(
    processor,
    codes,
)

split_reports = {}

for split, source_path in (
    source_paths.items()
):
    print(
        f"Encoding {split} split...",
        flush=True,
    )

    records = load_jsonl(
        source_path
    )

    output_path = output_paths[
        split
    ]

    temporary = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    lengths = []
    original_lengths = []
    state_token_lengths = []
    retained_state_lengths = []

    truncated = 0
    cardinality = {
        field: Counter()
        for field in FIELDS
    }

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for index, record in enumerate(
            records
        ):
            text = record.get("text")

            assert isinstance(text, str)
            assert text.strip()

            targets = validate_targets(
                record
            )

            (
                encoded,
                was_truncated,
                original_length,
                retained_state_tokens,
            ) = encode_with_truncation(
                encoder,
                processor.tokenizer,
                text,
            )

            assert encoded.length <= (
                MAXIMUM_LENGTH
            )

            assert encoded.keys == FIELDS
            assert len(encoded.markers) == 4
            assert encoded.pixel_values is None
            assert encoded.image_grid_thw is None

            marker_positions = [
                int(value)
                for value
                in encoded.markers.tolist()
            ]

            assert marker_positions == sorted(
                marker_positions
            )

            assert all(
                0 <= marker < encoded.length
                for marker in marker_positions
            )

            original_state_tokens = len(
                processor.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )

            cached_record = {
                "split": split,
                "split_position": index,
                "conversation_id":
                    record.get(
                        "conversation_id"
                    ),
                "primary_dataset":
                    record.get(
                        "primary_dataset"
                    ),
                "input_ids": [
                    int(value)
                    for value in (
                        encoded.input_ids
                        .tolist()
                    )
                ],
                "markers":
                    marker_positions,
                "question_keys":
                    list(encoded.keys),
                "sequence_length":
                    encoded.length,
                "original_sequence_length":
                    original_length,
                "original_state_tokens":
                    original_state_tokens,
                "retained_state_tokens":
                    retained_state_tokens,
                "truncated":
                    was_truncated,
                "targets": targets,
            }

            output_handle.write(
                json.dumps(
                    cached_record,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )

            output_handle.write("\n")

            lengths.append(
                encoded.length
            )

            original_lengths.append(
                original_length
            )

            state_token_lengths.append(
                original_state_tokens
            )

            retained_state_lengths.append(
                retained_state_tokens
            )

            if was_truncated:
                truncated += 1

            for field in FIELDS:
                cardinality[field][
                    len(
                        targets[field][
                            "indices"
                        ]
                    )
                ] += 1

            if (
                (index + 1) % 500 == 0
                or index + 1
                == len(records)
            ):
                print(
                    f"{split}: "
                    f"processed="
                    f"{index + 1}/"
                    f"{len(records)}",
                    flush=True,
                )

                save_json(
                    status_path,
                    {
                        "status": "RUNNING",
                        "stage":
                            f"encoding_{split}",
                        "processed":
                            index + 1,
                        "split_rows":
                            len(records),
                        "elapsed_seconds":
                            time.time()
                            - started,
                    },
                )

    temporary.replace(
        output_path
    )

    split_reports[split] = {
        "rows": len(records),
        "truncated_rows":
            truncated,
        "truncated_rate":
            truncated / len(records),
        "sequence_lengths": {
            "minimum":
                min(lengths),
            "maximum":
                max(lengths),
            "mean":
                statistics.mean(lengths),
            "median":
                statistics.median(
                    lengths
                ),
        },
        "original_sequence_lengths": {
            "minimum":
                min(original_lengths),
            "maximum":
                max(original_lengths),
            "mean":
                statistics.mean(
                    original_lengths
                ),
            "median":
                statistics.median(
                    original_lengths
                ),
        },
        "state_token_lengths": {
            "minimum":
                min(
                    state_token_lengths
                ),
            "maximum":
                max(
                    state_token_lengths
                ),
            "mean":
                statistics.mean(
                    state_token_lengths
                ),
            "median":
                statistics.median(
                    state_token_lengths
                ),
        },
        "retained_state_tokens": {
            "minimum":
                min(
                    retained_state_lengths
                ),
            "maximum":
                max(
                    retained_state_lengths
                ),
            "mean":
                statistics.mean(
                    retained_state_lengths
                ),
            "median":
                statistics.median(
                    retained_state_lengths
                ),
        },
        "target_cardinality": {
            field: dict(
                sorted(
                    counter.items()
                )
            )
            for field, counter
            in cardinality.items()
        },
        "source": str(source_path),
        "source_sha256":
            sha256(source_path),
        "output": str(output_path),
        "output_sha256":
            sha256(output_path),
    }

expected_rows = {
    "train": 3978,
    "validation": 850,
    "test": 858,
}

assert {
    split: report["rows"]
    for split, report
    in split_reports.items()
} == expected_rows

report = {
    "status": "PASSED",
    "checkpoint": str(checkpoint),
    "model":
        os.environ.get(
            "PPLX_MODEL_ID",
            "perplexity-ai/"
            "pplx-decider-v1.2",
        ),
    "revision":
        os.environ.get(
            "PPLX_MODEL_REVISION",
            "",
        ),
    "maximum_length":
        MAXIMUM_LENGTH,
    "fields":
        list(FIELDS),
    "question_instruction_template":
        QUESTION_TEMPLATE,
    "question_option_template":
        OPTION_TEMPLATE,
    "targets_changed": False,
    "descriptions_changed": False,
    "split_membership_changed": False,
    "splits": split_reports,
    "elapsed_seconds":
        time.time() - started,
}

save_json(
    report_path,
    report,
)

save_json(
    status_path,
    {
        **report,
        "stage": "complete",
        "report": str(report_path),
    },
)

print("=" * 78)
print("PPLX TRAINING TOKEN CACHE BUILD")
print("=" * 78)
print("Status: PASSED")
print("Maximum length:", MAXIMUM_LENGTH)
print()

for split in (
    "train",
    "validation",
    "test",
):
    split_report = split_reports[
        split
    ]

    print(split.upper())
    print(
        "  Rows:",
        split_report["rows"],
    )
    print(
        "  Truncated:",
        split_report[
            "truncated_rows"
        ],
        f"({100 * split_report['truncated_rate']:.2f}%)",
    )
    print(
        "  Final lengths:",
        split_report[
            "sequence_lengths"
        ],
    )
    print(
        "  Output:",
        split_report["output"],
    )
    print()

print("Report:", report_path)
print()
print(
    "PPLX TRAINING TOKEN CACHE "
    "BUILD PASSED"
)
