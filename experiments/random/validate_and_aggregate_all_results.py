from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
import csv
import hashlib
import json
import math
import os
import re
import subprocess

PAPER = Path(
    "/home/gpuuser2/gpuuser2_a/atishya/tads_paper_faithful"
)
TADS = Path(
    "/home/gpuuser2/gpuuser2_a/atishya/tads"
)

RESULT_ROOT = (
    PAPER
    / "model_finetune/model_output/results/random_10k_4runs"
)
STATE_ROOT = PAPER / "data/random_10k_4runs/evaluation_queue"
LOG_ROOT = PAPER / "logs/random_10k_4runs/evaluations"
FINAL_ROOT = PAPER / "metadata/random_10k_4runs_final"

BENCHMARKS = ("mmlu", "truthfulqa", "gsm8k", "bbh", "tydiqa")
RUNS = (1, 2, 3, 4)

SAMPLING_SEEDS = {
    1: 1787701845470005203,
    2: 1787701859309839880,
    3: 1787701873612384727,
    4: 1787701887533694015,
}

METRIC_NAMES = {
    "mmlu": "accuracy",
    "truthfulqa": "truth-info acc",
    "gsm8k": "exact_match",
    "bbh": "exact_match",
    "tydiqa": "f1",
}

# Exact independently validated values from the frozen E6 experiment.
VANILLA_TADS = {
    "mmlu": 64.28252920717024,
    "truthfulqa": 21.83544303797468,
    "gsm8k": 54.0,
    "bbh": 59.81481481481481,
    "tydiqa": 60.75023335819473,
}

E6 = {
    "mmlu": 63.80986355123517,
    "truthfulqa": 17.24683544303797,
    "gsm8k": 57.0,
    "bbh": 62.31481481481481,
    "tydiqa": 63.19376335430737,
}

FATAL_RE = re.compile(
    r"Traceback \(most recent call last\)"
    r"|CUDA out of memory"
    r"|ChildFailedError"
    r"|RuntimeError"
    r"|AssertionError"
    r"|Killed"
    r"|SIGTERM"
    r"|SIGKILL"
    r"|Segmentation fault"
    r"|Bus error"
)

COMPLETION_TEXT = {
    "mmlu": "Average accuracy:",
    "gsm8k": "Exact match :",
    "bbh": "Average EM:",
    "truthfulqa": "truth-info acc",
    "tydiqa": "Done!",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_json(path: Path):
    assert path.is_file(), f"Missing JSON: {path}"

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def jsonl_records(path: Path) -> list[dict]:
    assert path.is_file(), f"Missing JSONL: {path}"
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            value = json.loads(line)
            assert isinstance(value, dict), (
                f"{path}:{line_number} is not a JSON object"
            )
            rows.append(value)

    return rows


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "1.0", "yes"}:
        return True

    if text in {"false", "0", "0.0", "no"}:
        return False

    raise AssertionError(f"Invalid Boolean value: {value!r}")


def close(left: float, right: float, tolerance: float = 1e-9):
    assert math.isfinite(left)
    assert math.isfinite(right)
    assert abs(left - right) <= tolerance, (
        f"Mismatch: {left} != {right}; "
        f"error={abs(left - right)}"
    )


def validate_completion_marker(
    benchmark: str,
    run: int,
    result_dir: Path,
    log_path: Path,
):
    marker_path = (
        STATE_ROOT
        / benchmark
        / f"run_{run}"
        / "EVALUATION_COMPLETE.json"
    )
    marker = load_json(marker_path)

    assert marker["status"] == "EVALUATION_PROCESS_COMPLETED"
    assert marker["benchmark"] == benchmark
    assert int(marker["run"]) == run
    assert Path(marker["result_directory"]) == result_dir
    assert Path(marker["log"]) == log_path
    assert marker["log_sha256"] == sha256(log_path)

    files = sorted(
        path for path in result_dir.rglob("*")
        if path.is_file()
    )

    assert int(marker["result_file_count"]) == len(files)
    assert int(marker["result_total_bytes"]) == sum(
        path.stat().st_size for path in files
    )

    return marker_path, marker


def validate_log(benchmark: str, log_path: Path):
    assert log_path.is_file(), f"Missing log: {log_path}"

    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).replace("\r", "\n")

    fatal = sorted(set(FATAL_RE.findall(text)))
    assert not fatal, f"Fatal patterns in {log_path}: {fatal}"

    required_text = COMPLETION_TEXT[benchmark]
    assert required_text in text, (
        f"Missing completion evidence {required_text!r} "
        f"in {log_path}"
    )

    return text


def validate_mmlu(result_dir: Path, log_text: str) -> tuple[float, dict]:
    metrics = load_json(result_dir / "metrics.json")
    assert "average_acc" in metrics

    csv_files = sorted(
        path for path in result_dir.glob("*.csv")
        if path.name != "metrics.csv"
    )
    assert len(csv_files) == 57, (
        f"MMLU subject files: {len(csv_files)}"
    )

    correct = 0
    total = 0

    for path in csv_files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None
            assert "correct" in reader.fieldnames

            for row in reader:
                total += 1
                correct += int(parse_bool(row["correct"]))

    assert total == 11213, f"MMLU questions: {total}"

    recomputed = correct / total
    saved = float(metrics["average_acc"])
    close(recomputed, saved)

    logged_matches = re.findall(
        r"Average accuracy:\s*([0-9.]+)",
        log_text,
    )
    assert logged_matches

    close(float(logged_matches[-1]), saved, tolerance=5e-4)

    return saved * 100.0, {
        "questions": total,
        "correct": correct,
        "saved_fraction": saved,
        "recomputed_fraction": recomputed,
    }


def validate_gsm8k(result_dir: Path, log_text: str) -> tuple[float, dict]:
    metrics = load_json(result_dir / "metrics.json")
    assert "exact_match" in metrics

    predictions = jsonl_records(result_dir / "predictions.jsonl")
    assert len(predictions) == 200

    saved = float(metrics["exact_match"])
    assert 0.0 <= saved <= 1.0

    logged_matches = re.findall(
        r"Exact match\s*:\s*([0-9.]+)",
        log_text,
    )
    assert logged_matches
    close(float(logged_matches[-1]), saved)

    return saved * 100.0, {
        "predictions": len(predictions),
        "saved_fraction": saved,
    }


def validate_bbh(result_dir: Path, log_text: str) -> tuple[float, dict]:
    metrics = load_json(result_dir / "metrics.json")
    assert "average_exact_match" in metrics

    task_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if key != "average_exact_match"
    }
    assert len(task_metrics) == 27

    for value in task_metrics.values():
        assert 0.0 <= value <= 1.0

    saved = float(metrics["average_exact_match"])
    recomputed_macro = mean(task_metrics.values())
    close(saved, recomputed_macro)

    prediction_files = sorted(
        (result_dir / "predictions").glob("*.jsonl")
    )
    assert len(prediction_files) == 27

    records_per_task = {}
    total = 0

    for path in prediction_files:
        records = jsonl_records(path)
        assert len(records) == 40
        records_per_task[path.stem] = len(records)
        total += len(records)

    assert total == 1080

    logged_matches = re.findall(
        r"Average EM:\s*([0-9.]+)",
        log_text,
    )
    assert logged_matches
    close(float(logged_matches[-1]), saved)

    return saved * 100.0, {
        "tasks": len(task_metrics),
        "predictions": total,
        "saved_fraction": saved,
        "recomputed_macro_fraction": recomputed_macro,
        "records_per_task": records_per_task,
    }


def validate_truthfulqa(
    result_dir: Path,
    log_text: str,
) -> tuple[float, dict]:
    metrics = load_json(result_dir / "metrics.json")

    required = {
        "MC1",
        "MC2",
        "truth acc",
        "info acc",
        "truth-info acc",
    }
    assert required.issubset(metrics)

    predictions_path = result_dir / "predictions.csv"
    summary_path = result_dir / "summary.csv"

    assert predictions_path.is_file()
    assert summary_path.is_file()

    with predictions_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 632

    fieldnames = list(rows[0].keys())

    recomputed = {}

    for metric_name in required:
        candidates = [
            field
            for field in fieldnames
            if field == metric_name
            or field.endswith(f" {metric_name}")
        ]

        assert len(candidates) == 1, (
            f"TruthfulQA column candidates for {metric_name}: "
            f"{candidates}"
        )

        column = candidates[0]
        values = [float(row[column]) for row in rows]
        recomputed[metric_name] = mean(values)
        close(recomputed[metric_name], float(metrics[metric_name]))

    headline = float(metrics["truth-info acc"])
    assert 0.0 <= headline <= 1.0

    return headline * 100.0, {
        "questions": len(rows),
        "saved_metrics": {
            key: float(metrics[key])
            for key in sorted(required)
        },
        "recomputed_metrics": recomputed,
    }


def validate_tydiqa(result_dir: Path, log_text: str) -> tuple[float, dict]:
    metrics = load_json(result_dir / "metrics.json")

    expected_languages = {
        "arabic",
        "bengali",
        "english",
        "finnish",
        "indonesian",
        "korean",
        "russian",
        "swahili",
        "telugu",
    }

    assert expected_languages.issubset(metrics)
    assert "average" in metrics

    for language in expected_languages:
        assert {
            "f1",
            "exact_match",
            "recall",
        }.issubset(metrics[language])

    saved_average = metrics["average"]
    assert {
        "f1",
        "exact_match",
        "recall",
    }.issubset(saved_average)

    recomputed = {}

    for metric_name in ("f1", "exact_match", "recall"):
        recomputed[metric_name] = mean(
            float(metrics[language][metric_name])
            for language in expected_languages
        )
        close(
            recomputed[metric_name],
            float(saved_average[metric_name]),
        )

    prediction_path = result_dir / "tydiaqa_predictions.jsonl"
    predictions = jsonl_records(prediction_path)
    assert len(predictions) == 885

    headline = float(saved_average["f1"])
    assert 0.0 <= headline <= 100.0

    return headline, {
        "languages": sorted(expected_languages),
        "predictions": len(predictions),
        "saved_average": {
            key: float(value)
            for key, value in saved_average.items()
        },
        "recomputed_average": recomputed,
    }


VALIDATORS = {
    "mmlu": validate_mmlu,
    "truthfulqa": validate_truthfulqa,
    "gsm8k": validate_gsm8k,
    "bbh": validate_bbh,
    "tydiqa": validate_tydiqa,
}


print("Checking repository:")

commit = subprocess.check_output(
    ["git", "-C", str(TADS), "rev-parse", "HEAD"],
    text=True,
).strip()

status = subprocess.check_output(
    ["git", "-C", str(TADS), "status", "--short"],
    text=True,
)

assert commit == "5459de6e411fdd2997156a53beae3df68705b9ec"
assert status == ""

print("Commit:", commit)
print("Clean: YES")

assert not FINAL_ROOT.exists(), f"Output already exists: {FINAL_ROOT}"
FINAL_ROOT.mkdir(parents=True)

run_scores: dict[int, dict[str, float]] = {}
validation_details = {}
evidence_hashes = {}

for run in RUNS:
    print()
    print("=" * 72)
    print(f"VALIDATING RANDOM RUN {run}")
    print("=" * 72)

    run_scores[run] = {}
    validation_details[str(run)] = {}

    for benchmark in BENCHMARKS:
        result_dir = RESULT_ROOT / f"run_{run}" / benchmark
        log_path = LOG_ROOT / f"eval_{benchmark}_run_{run}.log"

        assert result_dir.is_dir(), f"Missing results: {result_dir}"

        marker_path, marker = validate_completion_marker(
            benchmark,
            run,
            result_dir,
            log_path,
        )
        log_text = validate_log(benchmark, log_path)

        score, detail = VALIDATORS[benchmark](
            result_dir,
            log_text,
        )

        assert math.isfinite(score)
        assert 0.0 <= score <= 100.0

        run_scores[run][benchmark] = score
        validation_details[str(run)][benchmark] = detail

        evidence_hashes[
            str(marker_path.relative_to(PAPER))
        ] = sha256(marker_path)

        metrics_path = result_dir / "metrics.json"
        evidence_hashes[
            str(metrics_path.relative_to(PAPER))
        ] = sha256(metrics_path)

        evidence_hashes[
            str(log_path.relative_to(PAPER))
        ] = sha256(log_path)

        print(
            f"{benchmark:12s} "
            f"{METRIC_NAMES[benchmark]:16s} "
            f"{score:.6f}"
        )

random_means = {
    benchmark: mean(
        run_scores[run][benchmark]
        for run in RUNS
    )
    for benchmark in BENCHMARKS
}

random_sample_sd = {
    benchmark: stdev(
        run_scores[run][benchmark]
        for run in RUNS
    )
    for benchmark in BENCHMARKS
}

run_averages = {
    run: mean(run_scores[run][benchmark] for benchmark in BENCHMARKS)
    for run in RUNS
}

vanilla_average = mean(VANILLA_TADS.values())
e6_average = mean(E6.values())
random_average = mean(random_means.values())

detailed_csv = FINAL_ROOT / "random_4runs_detailed_comparison.csv"

with detailed_csv.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "benchmark",
        "metric",
        "random_run_1",
        "random_run_2",
        "random_run_3",
        "random_run_4",
        "random_mean",
        "random_sample_sd",
        "vanilla_tads",
        "e6",
        "random_mean_minus_vanilla",
        "e6_minus_random_mean",
    ]

    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    for benchmark in BENCHMARKS:
        writer.writerow({
            "benchmark": benchmark,
            "metric": METRIC_NAMES[benchmark],
            "random_run_1": f"{run_scores[1][benchmark]:.12f}",
            "random_run_2": f"{run_scores[2][benchmark]:.12f}",
            "random_run_3": f"{run_scores[3][benchmark]:.12f}",
            "random_run_4": f"{run_scores[4][benchmark]:.12f}",
            "random_mean": f"{random_means[benchmark]:.12f}",
            "random_sample_sd": (
                f"{random_sample_sd[benchmark]:.12f}"
            ),
            "vanilla_tads": f"{VANILLA_TADS[benchmark]:.12f}",
            "e6": f"{E6[benchmark]:.12f}",
            "random_mean_minus_vanilla": (
                f"{random_means[benchmark] - VANILLA_TADS[benchmark]:.12f}"
            ),
            "e6_minus_random_mean": (
                f"{E6[benchmark] - random_means[benchmark]:.12f}"
            ),
        })

method_csv = FINAL_ROOT / "method_scores_5_benchmarks.csv"

method_rows = [
    ("Vanilla TADS", VANILLA_TADS),
    ("E6", E6),
]

for run in RUNS:
    method_rows.append((
        f"Random Run {run}",
        run_scores[run],
    ))

method_rows.append(("Random Mean", random_means))
method_rows.append(("Random Sample SD", random_sample_sd))

with method_csv.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "method",
        "mmlu",
        "truthfulqa",
        "gsm8k",
        "bbh",
        "tydiqa",
        "five_benchmark_average",
    ]

    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    for method, values in method_rows:
        average = mean(values[benchmark] for benchmark in BENCHMARKS)

        writer.writerow({
            "method": method,
            **{
                benchmark: f"{values[benchmark]:.12f}"
                for benchmark in BENCHMARKS
            },
            "five_benchmark_average": f"{average:.12f}",
        })

runs_csv = FINAL_ROOT / "random_runs_only.csv"

with runs_csv.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "run",
        "sampling_seed",
        "mmlu",
        "truthfulqa",
        "gsm8k",
        "bbh",
        "tydiqa",
        "five_benchmark_average",
    ]

    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    for run in RUNS:
        writer.writerow({
            "run": run,
            "sampling_seed": SAMPLING_SEEDS[run],
            **{
                benchmark: f"{run_scores[run][benchmark]:.12f}"
                for benchmark in BENCHMARKS
            },
            "five_benchmark_average": f"{run_averages[run]:.12f}",
        })

report = {
    "status": "RANDOM_10K_4RUNS_RESULTS_FULLY_VALIDATED",
    "validated_utc": datetime.now(timezone.utc).isoformat(),
    "repository_commit": commit,
    "runs": list(RUNS),
    "benchmarks": list(BENCHMARKS),
    "sampling_seeds": SAMPLING_SEEDS,
    "scores_percentage": {
        str(run): run_scores[run]
        for run in RUNS
    },
    "random_mean_percentage": random_means,
    "random_sample_sd_percentage": random_sample_sd,
    "run_five_benchmark_averages": run_averages,
    "vanilla_tads_percentage": VANILLA_TADS,
    "e6_percentage": E6,
    "vanilla_five_benchmark_average": vanilla_average,
    "e6_five_benchmark_average": e6_average,
    "random_mean_five_benchmark_average": random_average,
    "random_mean_minus_vanilla": {
        benchmark: random_means[benchmark] - VANILLA_TADS[benchmark]
        for benchmark in BENCHMARKS
    },
    "e6_minus_random_mean": {
        benchmark: E6[benchmark] - random_means[benchmark]
        for benchmark in BENCHMARKS
    },
    "validation_details": validation_details,
    "evidence_sha256": evidence_hashes,
}

report_path = FINAL_ROOT / "validation_and_aggregation_report.json"
report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

output_hashes = {
    path.name: sha256(path)
    for path in (
        detailed_csv,
        method_csv,
        runs_csv,
        report_path,
    )
}

manifest = {
    "status": "COMPLETE",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "outputs": output_hashes,
}

manifest_path = FINAL_ROOT / "final_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

marker_path = FINAL_ROOT / "RANDOM_10K_4RUNS_RESULTS_VALIDATED"

marker_path.write_text(
    "\n".join([
        "RANDOM 10K × 4 RESULTS FULLY VALIDATED",
        f"completed_utc={datetime.now(timezone.utc).isoformat()}",
        "runs=4",
        "benchmarks=5",
        "evaluations=20",
        f"manifest_sha256={sha256(manifest_path)}",
        f"random_mean_average={random_average:.12f}",
        f"vanilla_tads_average={vanilla_average:.12f}",
        f"e6_average={e6_average:.12f}",
        "",
    ]),
    encoding="utf-8",
)

for path in FINAL_ROOT.iterdir():
    if path.is_file():
        path.chmod(0o444)

print()
print("=" * 72)
print("ALL 20 RANDOM-RUN RESULTS PASSED VALIDATION")
print("=" * 72)

print()
print("Per-run five-benchmark averages:")

for run in RUNS:
    print(f"Run {run}: {run_averages[run]:.6f}")

print()
print("Method averages:")
print(f"Vanilla TADS: {vanilla_average:.6f}")
print(f"E6:           {e6_average:.6f}")
print(f"Random mean:  {random_average:.6f}")

print()
print("Installed final artifacts:")

for path in sorted(FINAL_ROOT.iterdir()):
    print(
        path.name,
        path.stat().st_size,
        sha256(path),
    )
