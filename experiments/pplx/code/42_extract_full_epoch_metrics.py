#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import json
import numbers


BENCHMARKS = (
    "mmlu",
    "truthfulqa",
    "gsm8k",
    "bbh",
    "tydiqa",
)


def flatten_json(
    value,
    prefix="",
):
    flattened = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            flattened.extend(
                flatten_json(
                    child,
                    child_prefix,
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            flattened.extend(
                flatten_json(
                    child,
                    f"{prefix}[{index}]",
                )
            )

    elif (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
    ):
        flattened.append(
            (
                prefix.lower(),
                float(value),
            )
        )

    return flattened


def preference_score(
    benchmark,
    key,
):
    key = key.lower()

    if benchmark == "mmlu":
        if (
            "average" in key
            and (
                "accuracy" in key
                or "acc" in key
            )
        ):
            return 100

        if "average_acc" in key:
            return 110

    elif benchmark == "truthfulqa":
        if "mc2" in key:
            return 120

    elif benchmark == "gsm8k":
        if "exact_match" in key:
            return 110

        if "accuracy" in key:
            return 90

    elif benchmark == "bbh":
        if (
            "average" in key
            and "exact_match" in key
        ):
            return 120

        if (
            "average" in key
            and (
                "accuracy" in key
                or "acc" in key
            )
        ):
            return 100

    elif benchmark == "tydiqa":
        if (
            "average" in key
            and "f1" in key
        ):
            return 120

        if key.endswith(".f1"):
            return 90

    return 0


def extract_metric(
    benchmark,
    result_directory,
):
    json_files = sorted(
        result_directory.rglob("*.json")
    )

    assert json_files, (
        f"No JSON results under "
        f"{result_directory}"
    )

    candidates = []

    for json_path in json_files:
        try:
            payload = json.loads(
                json_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        for key, value in flatten_json(
            payload
        ):
            score = preference_score(
                benchmark,
                key,
            )

            if score > 0:
                candidates.append(
                    {
                        "preference": score,
                        "key": key,
                        "value": value,
                        "file": str(
                            json_path
                        ),
                    }
                )

    assert candidates, (
        f"Could not locate the primary "
        f"{benchmark} metric in "
        f"{result_directory}"
    )

    candidates.sort(
        key=lambda item: (
            -item["preference"],
            len(item["key"]),
            item["key"],
            item["file"],
        )
    )

    selected = candidates[0]

    raw_value = selected["value"]

    percentage = (
        raw_value * 100.0
        if -1.0 <= raw_value <= 1.0
        else raw_value
    )

    return {
        "raw_value": raw_value,
        "percentage": percentage,
        "source_key": selected["key"],
        "source_file": selected["file"],
        "candidate_count": len(
            candidates
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    assert 1 <= args.epoch <= 5

    metrics = {}

    for benchmark in BENCHMARKS:
        result_directory = (
            args.result_root
            / benchmark
        )

        assert result_directory.is_dir(), (
            f"Missing result directory: "
            f"{result_directory}"
        )

        metrics[benchmark] = (
            extract_metric(
                benchmark,
                result_directory,
            )
        )

    percentages = [
        metrics[benchmark][
            "percentage"
        ]
        for benchmark in BENCHMARKS
    ]

    summary = {
        "status": "PASSED",
        "epoch": args.epoch,
        "metrics": metrics,
        "average_percentage": (
            sum(percentages)
            / len(percentages)
        ),
        "smoke_metrics_are_final": False,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = args.output.with_suffix(
        ".json.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(args.output)

    print(
        "Epoch:",
        args.epoch,
    )

    for benchmark in BENCHMARKS:
        print(
            f"{benchmark}: "
            f"{metrics[benchmark]['percentage']:.4f}"
        )

    print(
        "Average:",
        f"{summary['average_percentage']:.4f}",
    )

    print("Output:", args.output)


if __name__ == "__main__":
    main()
