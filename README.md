# Decision Model for TADS

This project studies how decision models affect task-aware data selection (TADS). We start with the same pool of 300,932 instruction examples and compare four ways to select 10,000 examples for fine-tuning Llama 3.1 8B.

| Method | How it selects examples |
| --- | --- |
| Vanilla TADS | Reproduces the original TADS pipeline. It assigns source examples to clusters with BGE-M3 embeddings and scores their quality with Qwen. |
| D-TADS with LAYA | Uses fine-tuned LAYA for cluster assignment and base LAYA for quality scoring. |
| D-TADS with PPLX-Decider | Uses fine-tuned PPLX-Decider for cluster assignment and base PPLX-Decider for quality scoring. |
| Random sampling | Visits the source examples in a seeded random order and keeps the first 10,000 with at least one token that contributes to training loss. The experiment uses four draws. |

LAYA and PPLX-Decider select the data. **Llama 3.1 8B is the downstream model trained on the selected data.**

## How the experiments work

1. **Prepare the inputs.** Build the shared source pool and the target cluster catalog for Task, Topic, Style, and Audience.
2. **Assign clusters.** Run the Vanilla assignment pipeline or train and run one of the two decision-model classifiers.
3. **Score and filter.** Score the source examples using the five quality criteria. Apply the cutoff to each criterion.
4. **Select 10,000 examples.** Match target label frequencies and combine the four field-specific selections. Run the four random draws separately, using the saved NumPy PCG64 seeds and the original source row order.
5. **Train and evaluate.** Fine-tune Llama 3.1 8B on each selected dataset. Evaluate with MMLU, TruthfulQA MC2, GSM8K, BBH, and TyDiQA.
6. **Compare the methods.** Report benchmark scores, elapsed time, label and score agreement, and overlap between the final selections.

## Reading the results

The TADS runs record downstream scores after each of five epochs. The random runs have three-epoch results. The archived Vanilla and PPLX runs used different GSM8K evaluation coverage, so the recorded averages are descriptive. Comparing the saved models directly would require evaluating both on the same questions.

## Repository layout

| Path | Contents |
| --- | --- |
| `shared/` | Input preparation and the TADS source, trainer, and evaluators used by the paper. |
| `experiments/vanilla/` | Vanilla TADS configuration, downstream run, and benchmark aggregation. |
| `experiments/laya/` | LAYA classifier training, source assignment and scoring, 10K export, and downstream evaluation. |
| `experiments/pplx/` | PPLX-Decider adaptation, source assignment and scoring, 10K export, and sequential downstream evaluation. |
| `experiments/random/` | Verified Random draw reconstruction and archived-run merge, evaluation, and aggregation scripts. |
| `analysis/` | Per-example score and cluster agreement and overlap between the final selections. |

## Inputs and environments

The original runners expect a prepared experiment workspace. Set up the original source Parquet in its original row order, the TADS target and benchmark splits, and the local model snapshots before running the scripts. Keep datasets, model weights, and run outputs outside this repository. Several archived runners and configs contain paths from the original server. Inspect those paths and set them for your installation before running. Use the recorded model revisions and benchmark flags to compare with the paper.

The upstream TADS checkout was pinned to `5459de6e411fdd2997156a53beae3df68705b9ec`. PPLX-Decider v1.2 used revision `cce0f3f`. The Random draw replay uses the Llama 3.1 8B tokenizer at revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, the shared TADS message encoder, and a sequence limit of 2048. It reproduced all four saved position arrays in order. The four runs retained 10,000 examples each and skipped 6, 7, 11, and 8 examples, respectively, because no tokens contributed to training loss after preprocessing.

### Rebuild the four Random training sets

From the repository root, set `PAPER_WORK` to the prepared Vanilla workspace and `LLAMA_TOKENIZER` to the local tokenizer snapshot. Run in the TADS Python environment.

```bash
python experiments/random/rebuild_random_draws.py \
  --source "$PAPER_WORK/data/train-00000-of-00001.parquet" \
  --trainer shared/upstream_tads/model_finetune/finetune.py \
  --tokenizer "$LLAMA_TOKENIZER" \
  --max-length 2048 \
  --output /path/to/new/random-draws
```

To check the original archive instead of writing new training files, replace `--output` and its path with `--verify-against "$PAPER_WORK/data/random_10k_4runs"`.

### Rebuild the score and overlap report

Set `PAPER_WORK`, `LAYA_REFERENCE_ROOT`, and `PPLX_TADS_ROOT` to the prepared run directories and export them. With the saved model predictions, scores, and selections in those directories, run the following in the TADS Python environment.

```bash
python analysis/compare_tads_methods.py --output /path/to/new/tads-agreement.json
```

The report contains score rank correlations, absolute score differences, cluster agreement, and overlap between the final selections. The analysis reads a legacy PyTorch artifact, so use only your own trusted experiment files.

## References

- [Original TADS paper](https://openreview.net/forum?id=R40WoYbYab)
- [Original TADS implementation](https://github.com/tmlr-group/TADS)
- [LAYA model card](https://huggingface.co/convaiinnovations/laya)
- [PPLX-Decider v1.2 model card](https://huggingface.co/perplexity-ai/pplx-decider-v1.2)

### Train one Random draw

Activate the TADS GPU environment. Set `LLAMA_MODEL` to the local Llama 3.1
8B snapshot and `RANDOM_DRAWS` to the directory created by the replay script.
Use a fresh output directory for each run.

    bash experiments/random/train_random_3epochs.sh "$LLAMA_MODEL" "$RANDOM_DRAWS/run_1/random_10k_train.json" /path/to/new/run_1_3ep_lora

All four selected row lists matched the original archive. The launcher
reconstructs the recorded settings and was checked for Bash syntax.
We did not rerun downstream GPU training for this release.
