"""Rebuild the four Random 10K selections using the paper's source row order."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


SEEDS = (
    1787701845470005203,
    1787701859309839880,
    1787701873612384727,
    1787701887533694015,
)
POOL_SIZE = 300_932
SELECTION_SIZE = 10_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-against", type=Path)
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("tads_trainer", args.trainer)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    encode = trainer.encode_with_messages_format

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    rows = pq.read_table(args.source, columns=["dataset", "id", "messages"])
    if rows.num_rows != POOL_SIZE:
        raise SystemExit(f"Unexpected source row count: {rows.num_rows}")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)

    for number, seed in enumerate(SEEDS, start=1):
        rng = np.random.Generator(np.random.PCG64(seed))
        accepted, drawn = [], []

        for raw_position in rng.permutation(POOL_SIZE):
            position = int(raw_position)
            row = {"messages": rows["messages"][position].as_py()}
            labels = encode(
                row, tokenizer, max_seq_length=args.max_length, add_bos=False
            )["labels"]
            drawn.append(position)
            if bool((labels != -100).any().item()):
                accepted.append(position)
            if len(accepted) == SELECTION_SIZE:
                break

        if len(accepted) != SELECTION_SIZE:
            raise SystemExit(f"run_{number}: too few trainable rows")

        positions = np.asarray(accepted, dtype=np.int64)
        if args.verify_against:
            original = args.verify_against / f"run_{number}"
            for filename, regenerated in (
                ("drawn_positions.npy", drawn),
                ("source_positions.npy", positions),
            ):
                saved = np.load(original / filename, allow_pickle=False)
                if not np.array_equal(saved, regenerated):
                    raise SystemExit(f"MISMATCH: run_{number}/{filename}")
        else:
            target = args.output / f"run_{number}"
            target.mkdir()
            np.save(target / "source_positions.npy", positions)
            np.save(target / "drawn_positions.npy", np.asarray(drawn, dtype=np.int64))
            records = rows.take(positions).to_pylist()
            with (target / "random_10k_train.json").open("x", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False)
                handle.write("\n")

        print(
            f"run_{number}: retained={len(accepted)} "
            f"rejected={len(drawn) - len(accepted)}"
            + (" | exact archive match" if args.verify_against else ""),
            flush=True,
        )


if __name__ == "__main__":
    main()
