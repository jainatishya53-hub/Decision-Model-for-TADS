#!/usr/bin/env python3

from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math


PAPER = Path("/home/gpuuser2/gpuuser2_a/atishya/tads_paper_faithful")
RESULT_ROOT = PAPER / "model_finetune/model_output/results/static_row_uid_gte_5_5ep_from_scratch"
EXPERIMENT = PAPER / "data/vanilla_tads_5ep_from_scratch"
OUTPUT = PAPER / "metadata/vanilla_tads_5ep_from_scratch"


def load(path: Path):
    assert path.is_file() and path.stat().st_size > 0, path
    return json.loads(path.read_text())


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_epoch(epoch: int):
    root = RESULT_ROOT / f"epoch_{epoch}"
    mmlu = load(root / "mmlu/metrics.json")
    gsm = load(root / "gsm/metrics.json")
    bbh = load(root / "bbh/metrics.json")
    truth = load(root / "truthfulqa/metrics.json")
    tydi = load(root / "tydiqa/metrics.json")

    assert len(list((root / "mmlu").glob("*.csv"))) == 57
    assert line_count(root / "gsm/predictions.jsonl") == 200
    assert len([key for key in bbh if key != "average_exact_match"]) == 27
    with (root / "truthfulqa/predictions.csv").open(newline="") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 632
    assert line_count(root / "tydiqa/tydiaqa_predictions.jsonl") == 885

    required_truth = {"MC1", "MC2", "truth acc", "info acc", "truth-info acc"}
    assert set(truth) == required_truth
    assert all(0 <= float(value) <= 1 for value in truth.values())

    headline = {
        "epoch": epoch,
        "mmlu": 100 * float(mmlu["average_acc"]),
        "truthfulqa_mc2": 100 * float(truth["MC2"]),
        "gsm8k": 100 * float(gsm["exact_match"]),
        "bbh": 100 * float(bbh["average_exact_match"]),
        "tydiqa": float(tydi["average"]["f1"]),
    }
    headline["five_benchmark_average"] = sum(
        headline[key] for key in ("mmlu", "truthfulqa_mc2", "gsm8k", "bbh", "tydiqa")
    ) / 5

    detailed = {
        "epoch": epoch,
        "mmlu_accuracy": headline["mmlu"],
        "truthfulqa_mc1": 100 * float(truth["MC1"]),
        "truthfulqa_mc2": headline["truthfulqa_mc2"],
        "truthfulqa_truth_acc": 100 * float(truth["truth acc"]),
        "truthfulqa_info_acc": 100 * float(truth["info acc"]),
        "truthfulqa_truth_info_acc": 100 * float(truth["truth-info acc"]),
        "gsm8k_exact_match": headline["gsm8k"],
        "bbh_exact_match": headline["bbh"],
        "tydiqa_f1": headline["tydiqa"],
        "tydiqa_exact_match": float(tydi["average"]["exact_match"]),
        "tydiqa_recall": float(tydi["average"]["recall"]),
    }
    return headline, detailed


assert (EXPERIMENT / "TRAINING_COMPLETE.json").is_file()
assert not OUTPUT.exists(), OUTPUT

headlines = []
details = []
for epoch in range(1, 6):
    for benchmark in ("mmlu", "gsm8k", "bbh", "truthfulqa", "tydiqa"):
        marker = load(EXPERIMENT / "execution" / f"epoch_{epoch}" / f"{benchmark}.json")
        assert marker["status"] == "completed"
    headline, detailed = validate_epoch(epoch)
    headlines.append(headline)
    details.append(detailed)

for index, row in enumerate(headlines):
    for key in ("mmlu", "truthfulqa_mc2", "gsm8k", "bbh", "tydiqa", "five_benchmark_average"):
        delta_key = f"{key}_change_from_previous_epoch"
        row[delta_key] = "" if index == 0 else float(row[key]) - float(headlines[index - 1][key])

OUTPUT.mkdir(parents=True, exist_ok=False)
headline_path = OUTPUT / "vanilla_tads_5epoch_headline.csv"
detailed_path = OUTPUT / "vanilla_tads_5epoch_all_metrics.csv"
report_path = OUTPUT / "vanilla_tads_5epoch_validation.json"

with headline_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(headlines[0]))
    writer.writeheader()
    writer.writerows(headlines)

with detailed_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(details[0]))
    writer.writeheader()
    writer.writerows(details)

report = {
    "status": "COMPLETE_AND_VALIDATED",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "training": {
        "initialization": "original Llama-3.1-8B base; no prior adapter loaded",
        "epochs": 5,
        "examples_after_preprocessing": 9950,
        "optimization_steps": 390,
        "effective_batch_size": 128,
        "learning_rate": 1e-4,
        "scheduler": "linear",
        "warmup_ratio": 0.03,
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "maximum_sequence_length": 2048,
        "seed": 42,
    },
    "headline_metric_policy": {
        "mmlu": "accuracy",
        "truthfulqa": "MC2",
        "gsm8k": "exact_match",
        "bbh": "average_exact_match",
        "tydiqa": "macro_average_f1",
    },
    "epochs": {str(row["epoch"]): row for row in headlines},
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

manifest = OUTPUT / "SHA256SUMS.txt"
manifest.write_text(
    "".join(
        f"{sha256(path)}  {path.name}\n"
        for path in (headline_path, detailed_path, report_path)
    )
)

marker = OUTPUT / "VANILLA_TADS_5EPOCH_FULLY_VALIDATED.json"
marker.write_text(
    json.dumps(
        {
            "status": "COMPLETE_AND_VALIDATED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "headline_csv": str(headline_path),
            "headline_sha256": sha256(headline_path),
            "detailed_csv": str(detailed_path),
            "detailed_sha256": sha256(detailed_path),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print("VANILLA TADS FIVE-EPOCH RESULTS PASSED VALIDATION")
print(headline_path.read_text(), end="")
