#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import os
import re
import subprocess
import sys


restart_root = Path(
    os.environ["BASE_RESTART_ROOT"]
).resolve()

tads_directory = Path(
    "/home/gpuuser2/gpuuser2_a/atishya/"
    "tads/model_finetune"
)

results_root = (
    restart_root
    / "data"
    / "epoch_evaluation"
    / "results"
)

status_root = (
    restart_root
    / "data"
    / "epoch_evaluation"
    / "status"
)

output_directory = (
    restart_root
    / "metadata"
    / "epoch_evaluation"
)

output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

rows = []

for epoch in range(1, 6):
    completion_marker = (
        status_root
        / f"epoch_{epoch}"
        / "epoch.complete"
    )

    assert completion_marker.is_file(), (
        f"Epoch {epoch} is not complete"
    )

    # Read the existing TADS scores for MMLU, GSM8K,
    # BBH and TyDiQA.
    completed = subprocess.run(
        [
            sys.executable,
            "read_results.py",
            "--root_result_path",
            str(results_root),
            "--baseline_tag",
            f"epoch_{epoch}",
        ],
        cwd=tads_directory,
        text=True,
        capture_output=True,
        check=True,
    )

    aggregator_output = (
        completed.stdout
        + "\n"
        + completed.stderr
    )

    pattern = re.compile(
        rf"^\s*epoch_{epoch}\s+"
        r"([0-9.]+)\s+"  # MMLU
        r"([0-9.]+)\s+"  # old TruthfulQA metric
        r"([0-9.]+)\s+"  # GSM8K
        r"([0-9.]+)\s+"  # BBH
        r"([0-9.]+)\s+"  # TyDiQA
        r"([0-9.]+)\s*$",  # old average
        re.MULTILINE,
    )

    match = pattern.search(
        aggregator_output
    )

    assert match is not None, (
        f"Could not parse Epoch {epoch} scores:\n"
        f"{aggregator_output}"
    )

    original_values = [
        float(value)
        for value in match.groups()
    ]

    mmlu = original_values[0]
    old_truthfulqa_metric = original_values[1]
    gsm8k = original_values[2]
    bbh = original_values[3]
    tydiqa = original_values[4]
    old_average = original_values[5]

    # Read the required TruthfulQA MC2 score directly
    # from the saved official evaluation output.
    truthfulqa_metrics_path = (
        results_root
        / "truthfulqa"
        / f"epoch_{epoch}"
        / "metrics.json"
    )

    assert truthfulqa_metrics_path.is_file(), (
        f"Missing TruthfulQA metrics: "
        f"{truthfulqa_metrics_path}"
    )

    truthfulqa_metrics = json.loads(
        truthfulqa_metrics_path.read_text(
            encoding="utf-8"
        )
    )

    assert "MC2" in truthfulqa_metrics, (
        f"MC2 missing from {truthfulqa_metrics_path}"
    )

    mc2_fraction = float(
        truthfulqa_metrics["MC2"]
    )

    assert 0.0 <= mc2_fraction <= 1.0, (
        f"Invalid MC2 value: {mc2_fraction}"
    )

    truthfulqa_mc2 = (
        mc2_fraction * 100.0
    )

    corrected_average = sum(
        [
            mmlu,
            truthfulqa_mc2,
            gsm8k,
            bbh,
            tydiqa,
        ]
    ) / 5.0

    rows.append(
        {
            "epoch": epoch,
            "mmlu": mmlu,
            "truthfulqa_mc2": truthfulqa_mc2,
            "gsm8k": gsm8k,
            "bbh": bbh,
            "tydiqa": tydiqa,
            "corrected_average": corrected_average,
            "old_truthfulqa_metric": old_truthfulqa_metric,
            "old_average": old_average,
        }
    )

csv_path = (
    output_directory
    / "32_corrected_five_epoch_mc2_results.csv"
)

json_path = (
    output_directory
    / "32_corrected_five_epoch_mc2_results.json"
)

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

json_path.write_text(
    json.dumps(
        rows,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

best_row = max(
    rows,
    key=lambda row: row["corrected_average"],
)

print("=" * 78)
print("CORRECTED FIVE-EPOCH RESULTS — TRUTHFULQA MC2")
print("=" * 78)

print(
    "| Epoch | MMLU | TruthfulQA MC2 | "
    "GSM8K | BBH | TyDiQA | Corrected Average |"
)
print(
    "|---:|---:|---:|---:|---:|---:|---:|"
)

for row in rows:
    print(
        f"| {row['epoch']} "
        f"| {row['mmlu']:.1f} "
        f"| {row['truthfulqa_mc2']:.1f} "
        f"| {row['gsm8k']:.1f} "
        f"| {row['bbh']:.1f} "
        f"| {row['tydiqa']:.1f} "
        f"| {row['corrected_average']:.1f} |"
    )

print()
print(
    "Best epoch:",
    best_row["epoch"],
)
print(
    "Best corrected average:",
    f"{best_row['corrected_average']:.4f}",
)
print()
print("CSV:", csv_path)
print("JSON:", json_path)
print()
print("CORRECTED MC2 TABLE BUILD PASSED")
