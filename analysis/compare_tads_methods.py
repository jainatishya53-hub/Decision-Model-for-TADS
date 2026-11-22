"""Compare archived TADS runs on the CPU. Read only trusted experiment files."""

import argparse
from collections import defaultdict
from itertools import combinations
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
import torch


FIELDS = ("Task", "Topic", "Style", "Audience")
METRICS = ("completeness", "information_richness", "rarity", "complexity", "overall_score")
QWEN_NAMES = ("Completeness", "Information Richness", "Rarity", "Complexity", "Overall Score")


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(row):
    return row["dataset"], row["id"]


def messages(row):
    return [(message["role"], message["content"]) for message in row["messages"]]


def message_digest(row):
    return hashlib.sha256(json.dumps(messages(row), ensure_ascii=False).encode("utf-8")).digest()


def check_export(path, positions, source_ids, source_messages):
    if not path.is_file():
        return {"status": "export_missing"}
    exported = read_json(path)
    if not isinstance(exported, list) or len(exported) != len(positions):
        return {"status": "export_count_or_format_mismatch"}
    content_matches = identity_matches = 0
    for index, row in zip(positions, exported):
        if not isinstance(row, dict):
            continue
        if (row.get("dataset"), row.get("id")) == source_ids[index]:
            identity_matches += 1
        if isinstance(row.get("messages"), list) and message_digest(row) == source_messages[index]:
            content_matches += 1
    return {
        "status": "all_messages_match" if content_matches == len(positions) else "partial_or_modified",
        "rows": len(positions), "same_messages": content_matches, "same_dataset_and_id": identity_matches,
    }


def parquet_rows(path, columns):
    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        yield from batch.to_pylist()


def aligned_records(path, columns, source_ids):
    seen = set()
    for row in parquet_rows(path, ["source_position", "dataset", "id", *columns]):
        index = row["source_position"]
        require(isinstance(index, int) and 0 <= index < len(source_ids), f"{path}: invalid position")
        require(index not in seen, f"{path}: duplicate position {index}")
        require(identity(row) == source_ids[index], f"{path}: source identity mismatch at {index}")
        seen.add(index)
        yield index, row
    require(len(seen) == len(source_ids), f"{path}: incomplete source rows ({len(seen)})")


def number(value):
    if isinstance(value, dict):
        for key in ("score_1_to_10", "expected_score_1_to_10", "expected_score"):
            if key in value:
                return number(value[key])
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def score_array(path, source_ids):
    names = set(pq.ParquetFile(path).schema_arrow.names)
    columns = []
    for metric in METRICS:
        preferred = f"{metric}_score_1_to_10"
        column = preferred if preferred in names else metric
        require(column in names, f"{path}: missing {metric}; available columns: {sorted(names)}")
        columns.append(column)
    values = np.full((len(source_ids), len(METRICS)), np.nan)
    for index, row in aligned_records(path, columns, source_ids):
        values[index] = [number(row[column]) for column in columns]
    require(np.isfinite(values).any(axis=0).all(), f"{path}: unrecognized score schema")
    return values


def cluster_array(path, source_ids):
    columns = [f"{field}_indices" for field in FIELDS]
    result = [None] * len(source_ids)
    for index, row in aligned_records(path, columns, source_ids):
        result[index] = tuple(tuple(row[column]) for column in columns)
        for field, values in zip(FIELDS, result[index]):
            require(
                len(values) == len(set(values)) == (1 if field == "Task" else 3)
                and all(isinstance(value, int) and 0 <= value < 100 for value in values),
                f"{path}: invalid {field} assignment at row {index}",
            )
    return result


def normalize(value):
    return " ".join(value.split())


def cluster_aliases(catalog, original_data):
    aliases = {field: defaultdict(set) for field in FIELDS}
    expected = {(field, index) for field in FIELDS for index in range(100)}
    actual = {(entry["field"], entry["index"]) for entry in catalog}
    require(actual == expected and len(catalog) == 400, "Unexpected pinned cluster catalog")
    for entry in catalog:
        field, index = entry["field"], entry["index"]
        for text in (entry["cluster_id"], entry["raw_concatenated_cluster_text"]):
            if text:
                aliases[field][normalize(text)].add(index)

    tags = original_data / "tag_id_mapping.json"
    keywords = original_data / "extracted_keywords.json"
    keyword_map = read_json(keywords) if keywords.is_file() else {}
    if tags.is_file():
        for tag, raw in read_json(tags).items():
            if not isinstance(raw, str):
                continue
            for field in FIELDS:
                indices = aliases[field].get(normalize(raw), set())
                if not indices:
                    continue
                aliases[field][normalize(tag)].update(indices)
                if isinstance(keyword_map.get(tag), str):
                    aliases[field][normalize(keyword_map[tag])].update(indices)
    return aliases


def map_clusters(row, aliases, issues):
    content = row.get("generated_content") or {}
    result = []
    for field in FIELDS:
        values = content.get(field)
        if isinstance(values, str):
            values = [values]
        if not values:
            issues[field]["missing_rows"] += 1
            result.append(None)
            continue
        mapped = []
        for value in values:
            indices = aliases[field].get(normalize(value), set()) if isinstance(value, str) else set()
            if len(indices) != 1:
                key = "ambiguous_labels" if len(indices) > 1 else "unmapped_labels"
                issues[field][key] += 1
                continue
            mapped.append(next(iter(indices)))
        result.append(tuple(mapped) if len(mapped) == len(values) else None)
    return tuple(result)


def score_agreement(a, b, valid):
    per_metric = {}
    for col, metric in enumerate(METRICS):
        mask = valid[:, col]
        x, y = a[mask, col], b[mask, col]
        rho = spearmanr(x, y).statistic if len(x) > 1 and np.ptp(x) and np.ptp(y) else None
        per_metric[metric] = {
            "n": int(mask.sum()),
            "spearman": None if rho is None else float(rho),
            "mae": float(np.mean(np.abs(x - y))) if len(x) else None,
            "mean_a_minus_b": float(np.mean(x - y)) if len(x) else None,
        }

    complete = valid.all(axis=1)
    x = (a[complete] >= 5).all(axis=1)
    y = (b[complete] >= 5).all(axis=1)
    union = int((x | y).sum())
    return {"scores": per_metric, "filter": {
        "n": int(complete.sum()), "both_keep": int((x & y).sum()),
        "only_a_keeps": int((x & ~y).sum()), "only_b_keeps": int((~x & y).sum()),
        "both_reject": int((~x & ~y).sum()),
        "keep_jaccard": int((x & y).sum()) / union if union else None,
    }}


def cluster_agreement(a, b):
    output = {}
    for i, field in enumerate(FIELDS):
        n = exact = first = 0
        jaccard = 0.0
        for row_a, row_b in zip(a, b):
            left, right = row_a[i], row_b[i]
            if not left or not right:
                continue
            n += 1
            first += left[0] == right[0]
            x, y = set(left), set(right)
            exact += x == y
            jaccard += len(x & y) / len(x | y)
        output[field] = {
            "n": n, "exact_set_agreement": exact / n if n else None,
            "mean_jaccard": jaccard / n if n else None,
        }
        if field == "Task":
            output[field]["top1_agreement"] = first / n if n else None
    return output


def check_selection(values, source_count, label):
    require(len(values) == 10000 and len(set(values)) == 10000, f"{label}: expected 10,000 unique rows")
    require(all(type(i) is int and 0 <= i < source_count for i in values), f"{label}: invalid row ID")
    return set(values)


def selection_pairs(selections, source_count):
    rows = {}
    sets = {name: check_selection(values, source_count, name) for name, values in selections.items()}
    for a, b in combinations(sets, 2):
        shared = len(sets[a] & sets[b])
        rows[f"{a} vs {b}"] = {
            "shared": shared, "fraction_of_each": shared / 10000,
            "jaccard": shared / len(sets[a] | sets[b]),
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paper = Path(os.environ["PAPER_WORK"])
    laya = Path(os.environ["LAYA_REFERENCE_ROOT"])
    pplx = Path(os.environ["PPLX_TADS_ROOT"])
    data = paper / "data"
    source = data / "train-00000-of-00001.parquet"
    laya_data, pplx_data = laya / "data", pplx / "data"
    catalog_path = pplx_data / "target_supervision/exact_laya_description_aware_v1/complete_target.json"
    pinned = read_json(catalog_path)["complete_cluster_catalog"]
    aliases = cluster_aliases(pinned, data)
    issues = {field: defaultdict(int) for field in FIELDS}

    print("Verifying Vanilla scored rows against the source Parquet …", flush=True)
    vanilla = torch.load(data / "scored_data.pt", map_location="cpu", weights_only=False)
    require(len(vanilla) == pq.ParquetFile(source).metadata.num_rows == 300932, "Unexpected source size")
    source_ids, source_messages, vanilla_clusters = [], [], []
    vanilla_scores = np.full((len(vanilla), len(METRICS)), np.nan)
    for index, row in enumerate(parquet_rows(source, ["dataset", "id", "messages"])):
        record = vanilla[index]
        require(identity(record) == identity(row) and messages(record) == messages(row), f"Vanilla source mismatch at {index}")
        source_ids.append(identity(row))
        source_messages.append(message_digest(row))
        vanilla_clusters.append(map_clusters(record, aliases, issues))
        ratings = record.get("qwen_scores") or {}
        vanilla_scores[index] = [number(ratings.get(key)) for key in QWEN_NAMES]
    del vanilla

    print("Reading full score and cluster outputs …", flush=True)
    scores = {
        "Vanilla": vanilla_scores,
        "LAYA": score_array(laya_data / "quality_scoring/direct_laya_full/direct_laya_scores_all_300932.parquet", source_ids),
        "PPLX": score_array(pplx_data / "quality_scoring/merged/pplx_scored_all_300932.parquet", source_ids),
    }
    clusters = {
        "Vanilla": vanilla_clusters,
        "LAYA": cluster_array(laya_data / "source_predictions/source_predictions_all_300932.parquet", source_ids),
        "PPLX": cluster_array(pplx_data / "source_predictions/pplx_epoch8_source_predictions_all_300932.parquet", source_ids),
    }
    valid = {name: np.isfinite(value) & (value >= 1) & (value <= 10) for name, value in scores.items()}
    shared_valid = np.logical_and.reduce(list(valid.values()))
    report = {
        "source_rows": len(source_ids),
        "source_sha256": sha256(source),
        "invalid_score_counts": {
            name: {metric: int((~mask[:, i]).sum()) for i, metric in enumerate(METRICS)}
            for name, mask in valid.items()
        },
        "vanilla_cluster_mapping_issues": issues,
        "score_comparison_uses_common_valid_rows": True,
        "comparisons": {},
    }
    for a, b in combinations(scores, 2):
        report["comparisons"][f"{a} vs {b}"] = {
            **score_agreement(scores[a], scores[b], shared_valid),
            "clusters": cluster_agreement(clusters[a], clusters[b]),
        }

    print("Checking final selections …", flush=True)
    original = paper / "model_finetune/selected_data/static_row_uid_gte_5_row_uids.json"
    vanilla_positions = read_json(original)
    if isinstance(vanilla_positions, dict):
        vanilla_positions = vanilla_positions["row_uids"]
    selections = {"Vanilla": vanilla_positions}
    check_selection(vanilla_positions, len(source_ids), "Vanilla")
    export_checks = {"Vanilla": check_export(
        original.with_name("static_row_uid_gte_5_10k.json"), vanilla_positions,
        source_ids, source_messages,
    )}
    final_paths = {
        "LAYA": laya_data / "distribution_matching/original_tads/final_original_tads_laya_10000.parquet",
        "PPLX": pplx_data / "distribution_matching/original_tads_pplx/final_original_tads_pplx_10000.parquet",
    }
    for name, path in final_paths.items():
        selections[name] = []
        for row in parquet_rows(path, ["source_position", "dataset", "id", "messages"]):
            index = row["source_position"]
            require(type(index) is int and 0 <= index < len(source_ids), f"{path}: invalid row ID")
            require(identity(row) == source_ids[index], f"{path}: source identity mismatch at {index}")
            require(message_digest(row) == source_messages[index], f"{path}: source text mismatch at {index}")
            selections[name].append(index)
    for run in range(1, 5):
        path = data / f"random_10k_4runs/run_{run}/source_positions.npy"
        name = f"Random {run}"
        selections[name] = np.load(path, allow_pickle=False).tolist()
        check_selection(selections[name], len(source_ids), name)
        export_checks[name] = check_export(path.with_name("random_10k_train.json"), selections[name], source_ids, source_messages)
    report["selection_overlap"] = selection_pairs(selections, len(source_ids))
    common = set.intersection(*(set(selections[name]) for name in scores))
    report["selected_by_all_three_tads"] = len(common)
    report["random_overlap_summary"] = {}
    for name in scores:
        comparisons = [report["selection_overlap"][f"{name} vs Random {run}"] for run in range(1, 5)]
        counts = [row["shared"] for row in comparisons]
        report["random_overlap_summary"][name] = {
            "mean_shared": mean(counts), "min_shared": min(counts), "max_shared": max(counts),
            "mean_jaccard": mean(row["jaccard"] for row in comparisons),
        }
    report["selected_by_all_four_by_random_draw"] = {
        f"Random {run}": len(common.intersection(selections[f"Random {run}"]))
        for run in range(1, 5)
    }
    report["selection_identity_checks"] = {
        "LAYA": "original position plus dataset, ID, and original messages",
        "PPLX": "original position plus dataset, ID, and original messages",
        "sidecar_vs_export": export_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved: {args.output}")
    for name, result in report["comparisons"].items():
        print(name, json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
