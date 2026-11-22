#!/usr/bin/env python3

from pathlib import Path
import ast
import hashlib
import json
import os


METRICS = (
    "Completeness",
    "Information Richness",
    "Rarity",
    "Complexity",
    "Overall Score",
)

METRIC_COLUMNS = {
    "Completeness": "completeness",
    "Information Richness": "information_richness",
    "Rarity": "rarity",
    "Complexity": "complexity",
    "Overall Score": "overall_score",
}

pplx_root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

reference_root = Path(
    os.environ["LAYA_REFERENCE_ROOT"]
).resolve()

config_path = (
    pplx_root
    / "configs"
    / "21_pplx_quality_scoring_contract.json"
)

report_path = (
    pplx_root
    / "metadata"
    / "quality_scoring"
    / "21_pplx_quality_scoring_contract.json"
)


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def find_validated_rubrics():
    matches = []

    for path in sorted(
        reference_root.rglob("*.py")
    ):
        try:
            source = path.read_text(
                encoding="utf-8"
            )
            tree = ast.parse(source)
        except (UnicodeDecodeError, SyntaxError):
            continue

        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
                value_node = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value_node = node.value
            else:
                continue

            if not any(
                isinstance(target, ast.Name)
                and target.id == "QUESTIONS"
                for target in targets
            ):
                continue

            try:
                questions = ast.literal_eval(
                    value_node
                )
            except Exception:
                continue

            if not isinstance(questions, dict):
                continue

            if set(questions) != set(METRICS):
                continue

            valid = True

            for metric in METRICS:
                question = questions[metric]

                if not isinstance(question, dict):
                    valid = False
                    break

                if question.get("type") != "score":
                    valid = False
                    break

                criteria = question.get("criteria")

                if not isinstance(criteria, list):
                    valid = False
                    break

                if len(criteria) != 10:
                    valid = False
                    break

            if valid:
                matches.append(
                    {
                        "path": str(path),
                        "questions": questions,
                    }
                )

    return matches


matches = find_validated_rubrics()

assert matches, (
    "No validated five-question, ten-level "
    "quality rubric was found"
)

canonical_encodings = {
    json.dumps(
        match["questions"],
        sort_keys=True,
        ensure_ascii=False,
    )
    for match in matches
}

assert len(canonical_encodings) == 1, (
    "Multiple different validated rubrics were found"
)

questions = matches[0]["questions"]

state_template = """Evaluate this instruction-response training example.

SAMPLE TO EVALUATE
{sample_text}

ASSIGNED TASK
{Task_descriptions}

ASSIGNED TOPICS
{Topic_descriptions}

ASSIGNED STYLES
{Style_descriptions}

ASSIGNED AUDIENCES
{Audience_descriptions}

Use all assigned semantic anchors as reference points when
judging the example, especially for the Overall Score.
"""

contract = {
    "contract_version": 1,
    "experiment": (
        "pplx_decider_v1_2_direct_quality_scoring"
    ),
    "model": (
        "perplexity-ai/pplx-decider-v1.2"
    ),
    "model_revision": (
        "cce0f3f80033e9d89d80831d5281fd72124aacce"
    ),
    "runtime_policy": {
        "pplx_only": True,
        "laya_model_used": False,
        "laya_checkpoint_used": False,
        "laya_predictions_used": False,
        "laya_scores_used": False,
        "qwen_used": False,
        "external_teacher_scores_used": False,
    },
    "metrics": list(METRICS),
    "metric_columns": METRIC_COLUMNS,
    "questions": questions,
    "state_template": state_template,
    "anchor_rendering": {
        "with_title": (
            "{cluster_id} — {title}: {refined_description}"
        ),
        "without_title": (
            "{cluster_id}: {refined_description}"
        ),
        "line_prefix": "- ",
        "field_order": [
            "Task",
            "Topic",
            "Style",
            "Audience",
        ],
    },
    "score_policy": {
        "number_of_levels": 10,
        "native_expected_level_range": [
            0.0,
            9.0,
        ],
        "final_score_range": [
            1.0,
            10.0,
        ],
        "final_score_formula": (
            "native_expected_level + 1.0"
        ),
        "mapping_requires_native_smoke_validation": True,
        "selection_threshold": 5.0,
        "selection_operator": (
            "all five scores greater than or equal to 5"
        ),
    },
    "comparison_policy": {
        "arms": [
            "base_pplx",
            "epoch8_finetuned_pplx",
        ],
        "same_samples": True,
        "same_prompts": True,
        "same_cluster_assignments": True,
        "same_cluster_descriptions": True,
        "same_truncation_policy": True,
        "use_test_or_teacher_labels_for_selection": False,
    },
}

contract_text = json.dumps(
    contract,
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
) + "\n"

config_path.write_text(
    contract_text,
    encoding="utf-8",
)

report = {
    "status": "PASSED",
    "contract": str(config_path),
    "contract_sha256": sha256_text(
        contract_text
    ),
    "validated_rubric_occurrences": len(matches),
    "unique_validated_rubrics": (
        len(canonical_encodings)
    ),
    "criteria_per_metric": {
        metric: len(
            questions[metric]["criteria"]
        )
        for metric in METRICS
    },
    "runtime_is_pplx_only": True,
    "laya_runtime_dependency": False,
}

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print("=" * 78)
print("PPLX-ONLY QUALITY-SCORING CONTRACT")
print("=" * 78)
print("Status: PASSED")
print("Validated rubric occurrences:", len(matches))
print("Runtime model: perplexity-ai/pplx-decider-v1.2")
print("LAYA runtime dependency: NO")
print("Qwen dependency: NO")

for metric in METRICS:
    print(f"\n{metric}")
    print("  Instructions:", questions[metric].get("instructions"))

    for level, criterion in enumerate(
        questions[metric]["criteria"],
        start=1,
    ):
        print(f"  Level {level}: {criterion}")

print("\nContract:", config_path)
print("Contract SHA256:", sha256_text(contract_text))
print("Report:", report_path)
print("\nPPLX-ONLY QUALITY-SCORING CONTRACT PASSED")
