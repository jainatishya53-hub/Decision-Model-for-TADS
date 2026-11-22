#!/usr/bin/env python3
"""Run TADS Step 4 once and emit row_uid-primary and bare-ID compatibility arms."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import torch

from tads.config import load_config
from tads.step3.filter_ood import _filter_all_scores_gte
from tads.step4.distribution_match import (
    _convert_tags,
    _create_reverse_mapping,
    _sample_to_match_distribution,
)
from tads.step4.fusion import _equal_weight_fusion
from tads.utils.data_io import save_samples


EXPECTED_REPOSITORY_COMMIT = "5459de6e411fdd2997156a53beae3df68705b9ec"

EXPECTED_CODE_HASHES = {
    "tads.step3.filter_ood": "04e23099bb22b61fc863eee25ab4934b89145f66c3919211e498d72c9382d0e7",
    "tads.step4.distribution_match": "4cff4c48782ad0472beb58d54322bade7ea1d8fd790d2834c6eb9849eec14ca7",
    "tads.step4.fusion": "189a0ea40bc0c9adceb1b53d15e773ef68f68be028a679a04911e938bb7d8d97",
    "tads.utils.data_io": "c1f8c94fb9ffb7180e4099c93ed1a183d7a05a69dedfe28e19e0c21bdb868277",
}

EXPECTED_INPUT_HASHES = {
    "cluster_ids.json": "92d8b0a4d6664fc1c1fa1c68947358b68fb18c0b81851cd57648c4da038108f0",
    "tag_id_mapping.json": "5d90c4b4fa375edc38e351568dfdd197d8380d5f635aacd9b38b0498ffebe5b4",
    "extracted_keywords.json": "ce3f81e698d8905a4886ba755562fa5026d9ff03120716d1ff7825b46e618629",
    "scored_data.pt": "3c89a623afff119eee4c5052b66f6d3c30e7819307eebd528157a8ec8e864742",
    "samples_gte_1.pt": "4ef36f446b7c7dfc72bcf06c9ecc091eb723b1522b7fdba54efdcd1d1bc270ba",
    "samples_gte_2.pt": "51e6cedb4a25569be30c5af93cb39949bd34ce210ff7bb56c09052b960014c6a",
    "samples_gte_3.pt": "f3639622603e0b432010b2301bcacb891679ec3d73c174b3a64772a9eebe1217",
    "samples_gte_4.pt": "f35f422955b5bd606a7cdffe8f7f6da96089929f7dfca071aca2b45428cc51ce",
    "samples_gte_5.pt": "d903ccabc72eb657d4c00d6e4c220a4a246564cde06ebee868f8ec1d7dd10f52",
    "samples_gte_6.pt": "16c22d77fe30e804ec55067e6a09f509e1bef677c96a360a8ddf5393b2f418cb",
    "samples_gte_7.pt": "4402d03425d87968c945d8ffa15aa49577d45ad232b3cf2d1b7fad3fa46794e7",
    "samples_gte_8.pt": "ed60aad1e47a0d7d5cac479b43010c019dde1ea4dbeb23e585c617a4f5a3eacd",
    "samples_gte_9.pt": "9cc543698041e90693068339a2dff67d33ab72dff7cd93d854dcb8e1bd7a5296",
    "samples_gte_10.pt": "e1f325807b17bff8c463c44c5daaf624a6d7a820582b81cb67efea14f5bba341",
}

EXPECTED_COUNTS = {
    1: 300932,
    2: 293442,
    3: 195403,
    4: 104138,
    5: 46140,
    6: 30665,
    7: 4358,
    8: 54,
    9: 0,
    10: 0,
}

FIELDS = ["Topic", "Style", "Audience", "Task"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def strip_row_uid(sample: dict) -> dict:
    output = sample.copy()
    removed = output.pop("row_uid", None)
    assert isinstance(removed, int)
    return output


def vote_distribution(method_results) -> dict[int, int]:
    votes = Counter()
    for result in method_results:
        votes.update(result)
    return dict(sorted(Counter(votes.values()).items(), reverse=True))


def git_output(source: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments],
        text=True,
    ).strip()


def verify_locked_inputs(config, tads_source: Path) -> dict[str, str]:
    assert git_output(tads_source, "rev-parse", "HEAD") == EXPECTED_REPOSITORY_COMMIT
    assert git_output(tads_source, "status", "--short") == "", (
        "Official repository has uncommitted changes"
    )

    import tads.step3.filter_ood as filter_module
    import tads.step4.distribution_match as match_module
    import tads.step4.fusion as fusion_module
    import tads.utils.data_io as data_io_module

    modules = {
        "tads.step3.filter_ood": filter_module,
        "tads.step4.distribution_match": match_module,
        "tads.step4.fusion": fusion_module,
        "tads.utils.data_io": data_io_module,
    }
    actual_code_hashes = {}
    for name, module in modules.items():
        path = Path(inspect.getsourcefile(module)).resolve()
        actual = sha256(path)
        assert actual == EXPECTED_CODE_HASHES[name], f"Code checksum changed: {path}"
        actual_code_hashes[name] = actual

    paths = {
        "cluster_ids.json": Path(config.cluster_ids_json),
        "tag_id_mapping.json": Path(config.tag_id_mapping_json),
        "extracted_keywords.json": Path(config.extracted_keywords_json),
        "scored_data.pt": Path(config.scored_data_pt),
    }
    for threshold in config.selection.score_thresholds:
        paths[f"samples_gte_{threshold}.pt"] = Path(config.filtered_pt(threshold))

    actual_input_hashes = {}
    for name, path in paths.items():
        assert path.is_file(), f"Missing locked input: {path}"
        actual = sha256(path)
        assert actual == EXPECTED_INPUT_HASHES[name], f"Input checksum changed: {path}"
        actual_input_hashes[name] = actual

    return {**actual_code_hashes, **actual_input_hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tads-source", required=True)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    tads_source = Path(arguments.tads_source).resolve()
    data_dir = Path(config.data_dir).resolve()

    assert config.random_seed == 42
    assert config.selection.target_count == 10000
    assert config.selection.fields == FIELDS
    assert config.selection.score_thresholds == list(range(1, 11))

    final_primary = data_dir / "selected_row_uid_primary"
    final_compat = data_dir / "selected_bare_id_compat"
    official_default = data_dir / "selected"
    staging = data_dir / ".step4_dual_identity_staging"
    completion_marker = data_dir / "STEP4_DUAL_IDENTITY_COMPLETE"

    for path in [final_primary, final_compat, official_default, staging, completion_marker]:
        assert not path.exists(), f"Refusing to overwrite existing path: {path}"

    print("Verifying repository revision, source code, and all locked input checksums...")
    locked_hashes = verify_locked_inputs(config, tads_source)
    print("All locked inputs passed")

    with Path(config.cluster_ids_json).open(encoding="utf-8") as handle:
        reference_data = json.load(handle)
    with Path(config.extracted_keywords_json).open(encoding="utf-8") as handle:
        keyword_mapping = json.load(handle)
    with Path(config.tag_id_mapping_json).open(encoding="utf-8") as handle:
        tag_id_mapping = json.load(handle)

    assert len(reference_data) == 6279
    assert len(keyword_mapping) == 400
    assert len(tag_id_mapping) == 400
    reverse_mapping = _create_reverse_mapping(keyword_mapping, tag_id_mapping)

    print("Loading locked scored source...")
    scored = torch.load(config.scored_data_pt, map_location="cpu", weights_only=False)
    assert isinstance(scored, list) and len(scored) == 300932
    assert all("row_uid" not in sample for sample in scored)
    object_positions = {id(sample): position for position, sample in enumerate(scored)}
    assert len(object_positions) == len(scored)

    primary_staging = staging / "row_uid_primary"
    compat_staging = staging / "bare_id_compat"
    primary_staging.mkdir(parents=True)
    compat_staging.mkdir(parents=True)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    manifest = {
        "protocol": "TADS Step 4 dual-identity baseline",
        "primary_identity": "zero-based original parquet row_uid",
        "compatibility_identity": "repository bare id",
        "distribution_matching_implementation": "official repository function",
        "fusion_implementation": "official repository function",
        "repository_commit": EXPECTED_REPOSITORY_COMMIT,
        "random_seed": config.random_seed,
        "target_count": config.selection.target_count,
        "fields": FIELDS,
        "thresholds": config.selection.score_thresholds,
        "locked_hashes": locked_hashes,
        "threshold_results": {},
    }

    for threshold in config.selection.score_thresholds:
        print("\n" + "=" * 72)
        print(f"THRESHOLD {threshold}")
        print("=" * 72)
        filtered = torch.load(
            config.filtered_pt(threshold), map_location="cpu", weights_only=False
        )
        assert isinstance(filtered, list)
        assert len(filtered) == EXPECTED_COUNTS[threshold]

        if len(filtered) <= config.selection.target_count:
            print(f"Skipping threshold {threshold}: {len(filtered)} <= 10000")
            manifest["threshold_results"][str(threshold)] = {
                "source_count": len(filtered),
                "action": "SKIP",
            }
            continue

        expected_objects = _filter_all_scores_gte(scored, threshold)
        assert len(expected_objects) == len(filtered)
        row_uids = [object_positions[id(sample)] for sample in expected_objects]
        assert len(row_uids) == len(set(row_uids))
        assert row_uids == sorted(row_uids)

        data_with_uid = []
        for sample, row_uid in zip(filtered, row_uids):
            output = sample.copy()
            output["row_uid"] = row_uid
            data_with_uid.append(output)

        primary_methods = []
        compat_methods = []
        field_results = {}

        for field in FIELDS:
            print(f"Converting and matching {field}...")
            converted = [
                _convert_tags(sample, reverse_mapping, field) for sample in data_with_uid
            ]
            selected_primary = _sample_to_match_distribution(
                converted,
                reference_data,
                field,
                config.selection.target_count,
            )
            assert len(selected_primary) == config.selection.target_count
            selected_row_uids = [sample["row_uid"] for sample in selected_primary]
            assert len(set(selected_row_uids)) == config.selection.target_count
            selected_compat = [strip_row_uid(sample) for sample in selected_primary]

            primary_path = primary_staging / (
                f"distribution_10k_{field}_gte_{threshold}.pt"
            )
            compat_path = compat_staging / (
                f"distribution_10k_{field}_gte_{threshold}.pt"
            )
            official_metadata = {
                "reference_file": config.cluster_ids_json,
                "source_count": len(filtered),
                "field": field,
                "threshold": threshold,
                "selected_count": len(selected_primary),
            }
            primary_metadata = {
                **official_metadata,
                "identity_key": "row_uid",
                "row_uid_definition": "zero-based original parquet row position",
            }
            save_samples(selected_primary, str(primary_path), primary_metadata)
            save_samples(selected_compat, str(compat_path), official_metadata)

            primary_methods.append(selected_row_uids)
            compat_ids = [sample["id"] for sample in selected_compat]
            compat_methods.append(compat_ids)
            field_results[field] = {
                "selected_rows": len(selected_primary),
                "unique_row_uids": len(set(selected_row_uids)),
                "unique_bare_ids": len(set(compat_ids)),
                "duplicate_bare_id_occurrences": len(compat_ids) - len(set(compat_ids)),
            }

        random_primary = random.sample(data_with_uid, config.selection.target_count)
        random_compat = [strip_row_uid(sample) for sample in random_primary]
        save_samples(
            random_primary,
            str(primary_staging / f"random_10k_gte_{threshold}.pt"),
            {
                "threshold": threshold,
                "method": "random",
                "identity_key": "row_uid",
            },
        )
        save_samples(
            random_compat,
            str(compat_staging / f"random_10k_gte_{threshold}.pt"),
            {"threshold": threshold, "method": "random"},
        )

        fused_row_uids = _equal_weight_fusion(
            *primary_methods, target_count=config.selection.target_count
        )
        fused_bare_ids = _equal_weight_fusion(
            *compat_methods, target_count=config.selection.target_count
        )
        assert len(fused_row_uids) == config.selection.target_count
        assert len(set(fused_row_uids)) == config.selection.target_count
        assert len(fused_bare_ids) == config.selection.target_count
        assert len(set(fused_bare_ids)) == config.selection.target_count

        fusion_config = {
            "dimensions": FIELDS,
            "gte_threshold": threshold,
            "total_methods": len(FIELDS),
            "final_count": config.selection.target_count,
        }
        atomic_json_dump(
            {
                "fusion_config": {
                    **fusion_config,
                    "identity_key": "row_uid",
                    "row_uid_definition": "zero-based original parquet row position",
                },
                "fused_row_uids": fused_row_uids,
            },
            primary_staging / f"fused_10k_gte_{threshold}.json",
        )
        atomic_json_dump(
            {
                "fusion_config": fusion_config,
                "fused_ids": fused_bare_ids,
            },
            compat_staging / f"fused_10k_gte_{threshold}.json",
        )

        manifest["threshold_results"][str(threshold)] = {
            "source_count": len(filtered),
            "action": "PROCESS",
            "field_results": field_results,
            "primary_vote_distribution": vote_distribution(primary_methods),
            "compatibility_vote_distribution": vote_distribution(compat_methods),
            "primary_fused_count": len(fused_row_uids),
            "compatibility_fused_count": len(fused_bare_ids),
            "primary_fused_unique_row_uids": len(set(fused_row_uids)),
            "compatibility_fused_unique_ids": len(set(fused_bare_ids)),
        }

    processed = [
        int(threshold)
        for threshold, result in manifest["threshold_results"].items()
        if result["action"] == "PROCESS"
    ]
    assert processed == [1, 2, 3, 4, 5, 6]

    atomic_json_dump(manifest, staging / "step4_dual_identity_manifest.json")
    print("\nAll staged outputs completed. Installing final isolated arms...")
    os.replace(primary_staging, final_primary)
    os.replace(compat_staging, final_compat)
    shutil.move(
        str(staging / "step4_dual_identity_manifest.json"),
        str(data_dir / "step4_dual_identity_manifest.json"),
    )
    staging.rmdir()
    completion_marker.write_text(
        "STEP 4 DUAL-IDENTITY OUTPUTS COMPLETE\n", encoding="utf-8"
    )
    print(f"Primary row_uid output: {final_primary}")
    print(f"Bare-ID compatibility output: {final_compat}")
    print("STEP 4 DUAL-IDENTITY MATCHING AND FUSION COMPLETED")


if __name__ == "__main__":
    main()
