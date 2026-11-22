#!/usr/bin/env bash
set -uo pipefail

export PAPER_WORK=/home/gpuuser2/gpuuser2_a/atishya/tads_paper_faithful
export TADS_SOURCE=/home/gpuuser2/gpuuser2_a/atishya/tads
export CONDA_ENV=/home/gpuuser2/gpuuser2_a/atishya/tads_private/conda_env
export HF_HOME=/home/gpuuser2/gpuuser2_a/atishya/tads_private/cache/huggingface

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PAPER_WORK:$TADS_SOURCE/src"

export BASE_MODEL="$HF_HOME/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
export TOKENIZER_SOURCE="$PAPER_WORK/model_finetune/model_output/models/e6_final_trainable_3ep_merged"

mkdir -p \
  "$PAPER_WORK/logs/random_10k_4runs" \
  "$PAPER_WORK/data/random_10k_4runs/merge_preparation" \
  "$PAPER_WORK/data/random_10k_4runs/merge_recovery" \
  "$PAPER_WORK/model_finetune/model_output/models/random_10k_4runs"

echo "BATCH MERGE STARTED: $(date -u)"

for RUN in 2 3 4
do
  export RUN

  LORA_DIR="$PAPER_WORK/model_finetune/model_output/models/random_10k_4runs/run_${RUN}_3ep_lora"
  MERGED_DIR="$PAPER_WORK/model_finetune/model_output/models/random_10k_4runs/run_${RUN}_3ep_merged"
  RUN_LOG="$PAPER_WORK/logs/random_10k_4runs/merge_run_${RUN}_3ep.log"
  READY_DIR="$PAPER_WORK/data/random_10k_4runs/merge_preparation/run_${RUN}"
  RECOVERY_DIR="$PAPER_WORK/data/random_10k_4runs/merge_recovery/run_${RUN}"

  export LORA_DIR MERGED_DIR RUN_LOG READY_DIR RECOVERY_DIR

  echo
  echo "============================================================"
  echo "STARTING RUN-$RUN MERGE: $(date -u)"
  echo "============================================================"

  cd "$PAPER_WORK/model_finetune"

  "$CONDA_ENV/bin/python" -u merge_lora.py \
    --base_model_name_or_path "$BASE_MODEL" \
    --lora_model_name_or_path "$LORA_DIR" \
    --tokenizer_name_or_path "$BASE_MODEL" \
    --output_dir "$MERGED_DIR" \
    > "$RUN_LOG" 2>&1

  MERGE_STATUS=$?

  echo "Run-$RUN merge process exit status: $MERGE_STATUS"

  if grep -aEq \
    'Traceback|CUDA out of memory|RuntimeError|AssertionError|Killed|SIGTERM|SIGKILL' \
    "$RUN_LOG"
  then
    echo "FAIL: Fatal pattern found in Run-$RUN merge log"
    exit 1
  fi

  for FILE in \
    config.json \
    generation_config.json \
    model.safetensors.index.json
  do
    if [ ! -s "$MERGED_DIR/$FILE" ]; then
      echo "FAIL: Run-$RUN missing $FILE"
      exit 1
    fi
  done

  SHARD_COUNT="$(
    find "$MERGED_DIR" \
      -maxdepth 1 \
      -name 'model-*.safetensors' \
      -type f |
    wc -l
  )"

  if [ "$SHARD_COUNT" -ne 7 ]; then
    echo "FAIL: Run-$RUN has $SHARD_COUNT shards instead of 7"
    exit 1
  fi

  RECOVERED=false

  for FILE in \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json
  do
    if [ ! -s "$MERGED_DIR/$FILE" ]; then
      RECOVERED=true
      TEMP_FILE="$MERGED_DIR/.${FILE}.recovery.$$"
      cp "$TOKENIZER_SOURCE/$FILE" "$TEMP_FILE"
      mv "$TEMP_FILE" "$MERGED_DIR/$FILE"
    fi
  done

  sha256sum \
    "$MERGED_DIR/tokenizer.json" \
    "$MERGED_DIR/tokenizer_config.json" \
    "$MERGED_DIR/special_tokens_map.json"

  test "$(
    sha256sum "$MERGED_DIR/tokenizer.json" |
    awk "{print \$1}"
  )" = "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b"

  test "$(
    sha256sum "$MERGED_DIR/tokenizer_config.json" |
    awk "{print \$1}"
  )" = "babefc68107163310a6cb953878ee7c32e45979729943d291f2604ad9d264793"

  test "$(
    sha256sum "$MERGED_DIR/special_tokens_map.json" |
    awk "{print \$1}"
  )" = "cc2e013b7545f183ef03e079a3c91c6f364fa37e4068c512d7dd843e59024535"

  export MERGE_STATUS RECOVERED SHARD_COUNT

  "$CONDA_ENV/bin/python" - <<'PY_RECORD'
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

run = int(os.environ["RUN"])
paper = Path(os.environ["PAPER_WORK"])
merged = Path(os.environ["MERGED_DIR"])
run_log = Path(os.environ["RUN_LOG"])
ready = Path(os.environ["READY_DIR"])
recovery = Path(os.environ["RECOVERY_DIR"])

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

record = {
    "status": f"RANDOM RUN-{run} MERGED OUTPUT READY FOR FULL VALIDATION",
    "run": run,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "merge_process_exit_status": int(os.environ["MERGE_STATUS"]),
    "tokenizer_recovered": os.environ["RECOVERED"] == "true",
    "tokenizer_recovery_computational_effect": "NONE",
    "weight_shards": int(os.environ["SHARD_COUNT"]),
    "merge_log_sha256": sha256(run_log),
    "merged_directory": str(merged),
    "tokenizer_sha256": {
        "tokenizer.json": sha256(merged / "tokenizer.json"),
        "tokenizer_config.json": sha256(
            merged / "tokenizer_config.json"
        ),
        "special_tokens_map.json": sha256(
            merged / "special_tokens_map.json"
        ),
    },
}

temporary = ready.parent / f".run_{run}_ready_{os.getpid()}"
temporary.mkdir(parents=True)

record_path = temporary / f"RUN_{run}_MERGED_OUTPUT_READY.json"
record_path.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
record_path.chmod(0o444)
temporary.rename(ready)

if record["tokenizer_recovered"]:
    recovery_tmp = recovery.parent / (
        f".run_{run}_recovery_{os.getpid()}"
    )
    recovery_tmp.mkdir(parents=True)

    recovery_path = recovery_tmp / (
        f"RUN_{run}_TOKENIZER_RECOVERY_VALIDATED.json"
    )
    recovery_path.write_text(
        json.dumps({
            "status": f"RUN-{run} TOKENIZER RECOVERED",
            "run": run,
            "source": os.environ["TOKENIZER_SOURCE"],
            "destination": str(merged),
            "computational_effect": "NONE",
            "tokenizer_sha256": record["tokenizer_sha256"],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    recovery_path.chmod(0o444)
    recovery_tmp.rename(recovery)

print(f"Run-{run} readiness marker installed")
PY_RECORD

  echo "RUN-$RUN MERGE OUTPUT READY"
  du -sh "$MERGED_DIR"
done

echo
echo "============================================================"
echo "RUNS 2–4 MERGE BATCH COMPLETED: $(date -u)"
echo "============================================================"
