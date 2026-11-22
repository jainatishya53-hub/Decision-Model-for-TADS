#!/usr/bin/env python3
"""Prepare an isolated, paper-faithful TADS target/evaluation split.

This program performs only preparation and validation.  It does not annotate,
cluster, score, select, fine-tune, merge, or evaluate a model.  Existing TADS
artifacts are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SEED = 42
RATIO = 0.2
BENCHMARKS = ("bbh", "gsm", "mmlu", "truthfulqa", "tydiqa")


def clean(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return re.sub(r"\s+", " ", text).strip()


def exact_hash(question: Any, answer: Any, *, context: Any = "",
               options: list[Any] | tuple[Any, ...] = ()) -> str:
    payload = {
        "question": clean(question), "answer": clean(answer),
        "context": clean(context), "options": [clean(x) for x in options],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def choose(size: int) -> set[int]:
    random.seed(SEED)
    return set(random.sample(range(size), int(size * RATIO)))


def link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source.resolve())


def copy_auxiliary(source_root: Path, target_root: Path, eval_root: Path,
                   excluded: set[str]) -> None:
    for item in source_root.iterdir():
        if item.name in excluded:
            continue
        for destination_root in (target_root, eval_root):
            destination = destination_root / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)


def split_truthfulqa(raw: Path, target: Path, heldout: Path) -> dict[str, Any]:
    source_path = raw / "truthfulqa" / "TruthfulQA.csv"
    frame = pd.read_csv(source_path)
    target_positions = choose(len(frame))
    target_frame = frame[[i in target_positions for i in range(len(frame))]]
    eval_frame = frame[[i not in target_positions for i in range(len(frame))]]
    target_hashes = {
        exact_hash(row["Question"], row["Best Answer"])
        for _, row in target_frame.iterrows()
    }
    keep = [
        exact_hash(row["Question"], row["Best Answer"]) not in target_hashes
        for _, row in eval_frame.iterrows()
    ]
    removed = int(len(eval_frame) - sum(keep))
    eval_frame = eval_frame[keep]
    (target / "truthfulqa").mkdir(parents=True)
    (heldout / "truthfulqa").mkdir(parents=True)
    target_frame.to_csv(target / "truthfulqa" / "TruthfulQA.csv", index=False)
    eval_frame.to_csv(heldout / "truthfulqa" / "TruthfulQA.csv", index=False)
    return {"full": len(frame), "target": len(target_frame),
            "heldout": len(eval_frame), "exact_duplicates_removed": removed,
            "split_unit": "CSV row",
            "_target_stable_ids": [f"truthfulqa:test:{i}" for i in target_frame.index],
            "_heldout_stable_ids": [f"truthfulqa:test:{i}" for i in eval_frame.index]}


def gsm_hash(item: dict[str, Any]) -> str:
    answer = re.sub(r"(\d),(\d)", r"\1\2", item["answer"].strip())
    return exact_hash(item["question"], answer)


def split_gsm(raw: Path, target: Path, heldout: Path) -> dict[str, Any]:
    source_path = raw / "gsm" / "test.jsonl"
    lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    values = [json.loads(line) for line in lines]
    target_positions = choose(len(values))
    target_lines = [line for i, line in enumerate(lines) if i in target_positions]
    eval_pairs = [(i, line, values[i]) for i, line in enumerate(lines) if i not in target_positions]
    target_hashes = {gsm_hash(values[i]) for i in target_positions}
    kept_pairs = [(i, line) for i, line, item in eval_pairs if gsm_hash(item) not in target_hashes]
    kept = [line for _, line in kept_pairs]
    removed = len(eval_pairs) - len(kept)
    (target / "gsm").mkdir(parents=True)
    (heldout / "gsm").mkdir(parents=True)
    (target / "gsm" / "test.jsonl").write_text("".join(target_lines), encoding="utf-8")
    (heldout / "gsm" / "test.jsonl").write_text("".join(kept), encoding="utf-8")
    return {"full": len(values), "target": len(target_lines),
            "heldout": len(kept), "exact_duplicates_removed": removed,
            "split_unit": "JSONL row",
            "_target_stable_ids": [f"gsm:test:{i}" for i in sorted(target_positions)],
            "_heldout_stable_ids": [f"gsm:test:{i}" for i, _ in kept_pairs]}


def mmlu_row_hash(row: pd.Series) -> str:
    return exact_hash(row.iloc[0], row.iloc[-1], options=list(row.iloc[1:5]))


def split_mmlu(raw: Path, target: Path, heldout: Path) -> dict[str, Any]:
    source_root = raw / "mmlu"
    target_root = target / "mmlu"
    eval_root = heldout / "mmlu"
    (target_root / "test").mkdir(parents=True)
    (eval_root / "test").mkdir(parents=True)
    target_frames: dict[str, pd.DataFrame] = {}
    eval_frames: dict[str, pd.DataFrame] = {}
    full_count = target_count = 0
    target_stable_ids: list[str] = []
    heldout_stable_ids: list[str] = []
    for path in sorted((source_root / "test").glob("*_test.csv")):
        frame = pd.read_csv(path, header=None)
        positions = choose(len(frame))
        target_frame = frame[[i in positions for i in range(len(frame))]]
        eval_frame = frame[[i not in positions for i in range(len(frame))]]
        target_frames[path.name] = target_frame
        eval_frames[path.name] = eval_frame
        subject = path.name[:-9]
        target_stable_ids.extend(f"mmlu:test:{subject}:{i}" for i in target_frame.index)
        full_count += len(frame)
        target_count += len(target_frame)
    target_hashes = {
        mmlu_row_hash(row)
        for frame in target_frames.values()
        for _, row in frame.iterrows()
    }
    removed = 0
    heldout_count = 0
    for name, frame in target_frames.items():
        frame.to_csv(target_root / "test" / name, index=False, header=False)
    for name, frame in eval_frames.items():
        keep = [mmlu_row_hash(row) not in target_hashes for _, row in frame.iterrows()]
        removed += int(len(frame) - sum(keep))
        frame = frame[keep]
        subject = name[:-9]
        heldout_stable_ids.extend(f"mmlu:test:{subject}:{i}" for i in frame.index)
        heldout_count += len(frame)
        frame.to_csv(eval_root / "test" / name, index=False, header=False)
    copy_auxiliary(source_root, target_root, eval_root, {"test"})
    return {"full": full_count, "target": target_count,
            "heldout": heldout_count, "exact_duplicates_removed": removed,
            "split_unit": "test row per subject; seed reset per subject",
            "_target_stable_ids": target_stable_ids,
            "_heldout_stable_ids": heldout_stable_ids}


def bbh_hash(item: dict[str, Any]) -> str:
    return exact_hash(item["input"], item["target"])


def split_bbh(raw: Path, target: Path, heldout: Path) -> dict[str, Any]:
    source_root = raw / "bbh"
    source_tasks = source_root / "bbh"
    target_root = target / "bbh"
    eval_root = heldout / "bbh"
    (target_root / "bbh").mkdir(parents=True)
    (eval_root / "bbh").mkdir(parents=True)
    target_values: dict[str, list[dict[str, Any]]] = {}
    eval_values: dict[str, list[dict[str, Any]]] = {}
    target_positions_by_task: dict[str, list[int]] = {}
    eval_positions_by_task: dict[str, list[int]] = {}
    full_count = target_count = 0
    for path in sorted(source_tasks.glob("*.json")):
        examples = json.loads(path.read_text(encoding="utf-8"))["examples"]
        positions = choose(len(examples))
        target_values[path.name] = [x for i, x in enumerate(examples) if i in positions]
        eval_values[path.name] = [x for i, x in enumerate(examples) if i not in positions]
        target_positions_by_task[path.name] = [i for i in range(len(examples)) if i in positions]
        eval_positions_by_task[path.name] = [i for i in range(len(examples)) if i not in positions]
        full_count += len(examples)
        target_count += len(target_values[path.name])
    target_hashes = {bbh_hash(x) for rows in target_values.values() for x in rows}
    removed = heldout_count = 0
    for name, rows in target_values.items():
        write_json(target_root / "bbh" / name, {"examples": rows})
    for name, rows in eval_values.items():
        pairs = list(zip(eval_positions_by_task[name], rows))
        kept_pairs = [(i, x) for i, x in pairs if bbh_hash(x) not in target_hashes]
        kept = [x for _, x in kept_pairs]
        eval_positions_by_task[name] = [i for i, _ in kept_pairs]
        removed += len(rows) - len(kept)
        heldout_count += len(kept)
        write_json(eval_root / "bbh" / name, {"examples": kept})
    copy_auxiliary(source_root, target_root, eval_root, {"bbh"})
    return {"full": full_count, "target": target_count,
            "heldout": heldout_count, "exact_duplicates_removed": removed,
            "split_unit": "example per task; seed reset per task",
            "_target_stable_ids": [
                f"bbh:test:{name[:-5]}:{i}"
                for name, positions in target_positions_by_task.items() for i in positions
            ],
            "_heldout_stable_ids": [
                f"bbh:test:{name[:-5]}:{i}"
                for name, positions in eval_positions_by_task.items() for i in positions
            ]}


def tydiqa_qas(raw: dict[str, Any]) -> list[tuple[int, dict[str, Any], str]]:
    rows = []
    for article_index, article in enumerate(raw["data"]):
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                rows.append((article_index, qa, paragraph["context"]))
    return rows


def tydiqa_qa_hash(qa: dict[str, Any], context: str) -> set[str]:
    return {
        exact_hash(qa["question"], answer["text"], context=context)
        for answer in qa["answers"]
    }


def split_tydiqa(raw_root: Path, target: Path, heldout: Path) -> dict[str, Any]:
    source_root = raw_root / "tydiqa"
    dev_path = source_root / "tydiqa-goldp-v1.1-dev.json"
    train_path = source_root / "tydiqa-goldp-v1.1-train.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    positions = choose(len(dev["data"]))
    target_articles = [x for i, x in enumerate(dev["data"]) if i in positions]
    eval_articles = [x for i, x in enumerate(dev["data"]) if i not in positions]
    target_raw = {"version": dev.get("version", ""), "data": target_articles}
    eval_raw = {"version": dev.get("version", ""), "data": eval_articles}
    target_hashes = {
        value_hash
        for _, qa, context in tydiqa_qas(target_raw)
        for value_hash in tydiqa_qa_hash(qa, context)
    }
    removed_qas = 0
    # Remove exact-content duplicate QAs from held-out dev while preserving
    # article/paragraph structure and the original order of remaining records.
    for article in eval_raw["data"]:
        for paragraph in article["paragraphs"]:
            kept = []
            for qa in paragraph["qas"]:
                if tydiqa_qa_hash(qa, paragraph["context"]) & target_hashes:
                    removed_qas += 1
                else:
                    kept.append(qa)
            paragraph["qas"] = kept
    target_root = target / "tydiqa"
    eval_root = heldout / "tydiqa"
    target_root.mkdir(parents=True)
    eval_root.mkdir(parents=True)
    write_json(target_root / dev_path.name, target_raw)
    write_json(eval_root / dev_path.name, eval_raw)
    # Train is used only for the evaluator's one-shot demonstrations.  It is not
    # included in the target distribution under the paper-faithful protocol.
    shutil.copy2(train_path, eval_root / train_path.name)
    full_qas = len(tydiqa_qas(dev))
    target_qas = len(tydiqa_qas(target_raw))
    heldout_qas = len(tydiqa_qas(eval_raw))
    target_answers = sum(len(qa["answers"]) for _, qa, _ in tydiqa_qas(target_raw))
    target_ids = [str(qa["id"]) for _, qa, _ in tydiqa_qas(target_raw)]
    heldout_ids = [str(qa["id"]) for _, qa, _ in tydiqa_qas(eval_raw)]
    return {"full": full_qas, "target": target_qas,
            "target_answer_rows_after_merge": target_answers,
            "heldout": heldout_qas, "exact_duplicates_removed": removed_qas,
            "split_unit": "dev article; seed 42; train excluded from target",
            "evaluation_train_file_role": "one-shot demonstrations only",
            "_target_stable_ids": target_ids,
            "_heldout_stable_ids": heldout_ids}


def collect_hashes(root: Path, benchmark: str) -> tuple[set[str], set[str]]:
    hashes: set[str] = set()
    stable: set[str] = set()
    if benchmark == "truthfulqa":
        frame = pd.read_csv(root / benchmark / "TruthfulQA.csv")
        for i, row in frame.iterrows():
            hashes.add(exact_hash(row["Question"], row["Best Answer"]))
            stable.add(clean(row["Question"]) + "\0" + clean(row["Best Answer"]))
    elif benchmark == "gsm":
        for line in (root / benchmark / "test.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line); hashes.add(gsm_hash(item)); stable.add(gsm_hash(item))
    elif benchmark == "mmlu":
        for path in sorted((root / benchmark / "test").glob("*_test.csv")):
            frame = pd.read_csv(path, header=None)
            for _, row in frame.iterrows(): hashes.add(mmlu_row_hash(row)); stable.add(mmlu_row_hash(row))
    elif benchmark == "bbh":
        for path in sorted((root / benchmark / "bbh").glob("*.json")):
            for item in json.loads(path.read_text(encoding="utf-8"))["examples"]:
                hashes.add(bbh_hash(item)); stable.add(bbh_hash(item))
    else:
        path = root / benchmark / "tydiqa-goldp-v1.1-dev.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        for _, qa, context in tydiqa_qas(raw):
            hashes.update(tydiqa_qa_hash(qa, context)); stable.add(str(qa["id"]))
    return hashes, stable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-work", type=Path, required=True)
    parser.add_argument("--new-work", type=Path, required=True)
    parser.add_argument("--tads-source", type=Path, required=True)
    args = parser.parse_args()
    existing = args.existing_work.resolve()
    new = args.new_work.resolve()
    source = args.tads_source.resolve()
    if new.exists():
        raise RuntimeError(f"Refusing to overwrite existing path: {new}")
    raw_eval = existing / "data" / "eval"
    for benchmark in BENCHMARKS:
        if not (raw_eval / benchmark).exists():
            raise FileNotFoundError(raw_eval / benchmark)
    if not (source / "src" / "tads").is_dir():
        raise FileNotFoundError(source / "src" / "tads")

    (new / "configs").mkdir(parents=True)
    (new / "data" / "eval" / "split_target").mkdir(parents=True)
    (new / "data" / "eval" / "split_eval").mkdir(parents=True)
    (new / "logs").mkdir()
    (new / "metadata").mkdir()
    target = new / "data" / "eval" / "split_target"
    heldout = new / "data" / "eval" / "split_eval"

    # Raw inputs and the 300K source pool are reused read-only via symlinks.
    for benchmark in BENCHMARKS:
        link(raw_eval / benchmark, new / "data" / "eval" / benchmark)
    link(existing / "data" / "train-00000-of-00001.parquet",
         new / "data" / "train-00000-of-00001.parquet")

    config_source = existing / "configs" / "single_gpu.yaml"
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    original_config = json.loads(json.dumps(config))
    config["base_dir"] = str(new)
    config_path = new / "configs" / "single_gpu.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    changed = {key for key in set(config) | set(original_config)
               if config.get(key) != original_config.get(key)}
    if changed != {"base_dir"}:
        raise AssertionError(f"Unexpected config changes: {sorted(changed)}")
    if config["random_seed"] != SEED or config["split_ratio"] != RATIO:
        raise AssertionError("Seed or split ratio differs from locked protocol")

    manifest = {
        "protocol": "paper-faithful 20% test/dev target; disjoint held-out evaluation",
        "seed": SEED, "split_ratio": RATIO,
        "tydiqa_target_source": "dev only",
        "existing_artifacts_modified": False,
        "hyperparameter_changes": ["base_dir only (path isolation; not a model/training hyperparameter)"],
        "benchmarks": {},
    }
    results = {
        "truthfulqa": split_truthfulqa(raw_eval, target, heldout),
        "gsm": split_gsm(raw_eval, target, heldout),
        "mmlu": split_mmlu(raw_eval, target, heldout),
        "bbh": split_bbh(raw_eval, target, heldout),
        "tydiqa": split_tydiqa(raw_eval, target, heldout),
    }

    intersections = {}
    for benchmark in BENCHMARKS:
        target_stable = set(results[benchmark].pop("_target_stable_ids"))
        eval_stable = set(results[benchmark].pop("_heldout_stable_ids"))
        target_hashes, _ = collect_hashes(target, benchmark)
        eval_hashes, _ = collect_hashes(heldout, benchmark)
        intersections[benchmark] = {
            "exact_hash_intersection": len(target_hashes & eval_hashes),
            "stable_identity_intersection": len(target_stable & eval_stable),
        }
        if intersections[benchmark]["exact_hash_intersection"] != 0:
            raise AssertionError(f"Exact content remains shared for {benchmark}")
        if intersections[benchmark]["stable_identity_intersection"] != 0:
            raise AssertionError(f"Stable identity remains shared for {benchmark}")
        manifest["benchmarks"][benchmark] = results[benchmark]
    manifest["intersections"] = intersections

    # Use the repository's own merge implementation; this is the first and only
    # derived TADS data product created by this preparation phase.
    sys_path = str(source / "src")
    import sys
    sys.path.insert(0, sys_path)
    from tads.step1.merge_targets import _combine_split_target_datasets
    merged = _combine_split_target_datasets(str(target))["train"]
    combined_path = new / "data" / "step1_combined_target.parquet"
    merged.to_parquet(str(combined_path))
    manifest["combined_target_rows"] = len(merged)
    manifest["combined_target_path"] = str(combined_path)

    write_json(new / "metadata" / "paper_faithful_split_manifest.json", manifest)
    write_json(new / "metadata" / "reuse_plan.json", {
        "reused": ["conda environment", "downloaded model snapshots", "raw benchmark files",
                   "300,932-row source parquet", "official repository code"],
        "must_rerun_after_split_gate": ["target annotation", "tag clustering", "embedding/cache stage",
                   "tag propagation", "keyword extraction", "source scoring", "filtering",
                   "distribution matching", "fusion", "training export", "LoRA fine-tuning",
                   "LoRA merge", "all five evaluations on split_eval"],
        "not_reused": ["old target annotations", "old tag mapping", "old propagated tags",
                       "old source scores", "old selections", "old fine-tuned model", "old metrics"],
    })
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print("PAPER-FAITHFUL SPLIT PREPARATION PASSED")


if __name__ == "__main__":
    main()
