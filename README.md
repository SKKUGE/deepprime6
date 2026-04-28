# DeepPrime6
Official repository for DeepPrime6: Deep learning-based prediction of prime editing efficiencies and transfer learning pipelines.

## Environment

This project targets Python 3.8–3.10.

### Option A: conda

```bash
conda env create -f environment.yaml
conda activate myenv
```

### Option B: pip

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

```bash
tar xzf data/pe6-data.tar.gz -C data/ (TBD)
```

## State-of-the-Art (SOTA) Models

The following table lists the final SOTA models reported in the paper. You can use the provided `scripts/predict.sh` script to evaluate these checkpoints or `scripts/train.sh` to fine-tune them further.

| PE type | Model | Run ID | Checkpoint path | Spearman |
|---|---|---:|---|---:|
| PE6a | PE6a (MainFT) | TBD | TBD | TBD |
| PEmaxdRNaseH | PEmaxdRNaseH (MainFT) | TBD | TBD | TBD |
| PE6b | PE6b (MainFT) | TBD | TBD | TBD |
| PE6c | PE6c (MainFT) | TBD | TBD | TBD |

## Quick Start

We provide simple bash wrapper scripts to abst
ract away Hydra configurations and quickly run training or inference.
Here is an example of training the PE6a model:

### Training a model

```bash
# Train a model using a predefined experiment configuration
bash scripts/train.sh pe6a-DP-baseline

# Train with custom hyperparameters (e.g., using CPU and a different batch size)
bash scripts/train.sh pe6a-DP-baseline trainer=cpu data.batch_size=128
```

### Evaluating a checkpoint

```bash
# Evaluate a checkpoint on the test set
bash scripts/predict.sh pe6a-DP-baseline logs/PE6a-ft/runs/2026-04-28_15-00-00/checkpoints/epoch_004.ckpt

# Evaluate with custom overrides
bash scripts/predict.sh pe6a-DP-baseline logs/PE6a-ft/runs/2026-04-28_15-00-00/checkpoints/epoch_004.ckpt trainer=cpu
```

## Train / Predict

If you need full control over the execution, you can use the detailed commands below:

### Single experiment (any PE type)

```bash
# Fine-tune pretrained DeepPrime weights on PE6a
python src/train.py experiment=pe6a-DP-baseline seed=42 logger=csv

# Train from scratch (no pretrained weights) on PE6a
python src/train.py experiment=from-scratch/pe6a-DP-baseline seed=42 logger=csv
```

`seed` is fixed to `42` in `configs/train.yaml`; override at the command line when
running Optuna sweeps (which sample `seed` as a hyperparameter).

### Hyperparameter Optimization (Sweeps)

To run multiple experiments or hyperparameter sweeps using Hydra's multirun mode:

```bash
# Run multiple from-scratch experiments in the background
python src/train.py -m \
    experiment=from-scratch/pe6a-DP-from-scratch,from-scratch/pe6b-DP-from-scratch,from-scratch/pe6c-DP-from-scratch,from-scratch/pemaxdrnaseh-DP-from-scratch \
    hparams_search=pe6-skw-from-scratch \
    >> output-from-scratch.log 2>&1 &
```

The `-m` flag enables multirun mode, allowing you to comma-separate multiple experiment configurations or use an `hparams_search` configuration for hyperparameter optimization (e.g., using Optuna).

## Hydra tips

Override any config key from the command line:

```bash
# Change accelerator and log to CSV instead of W&B
python src/train.py experiment=pe6a-DP-baseline trainer=cpu logger=csv

# Resume from a checkpoint
python src/train.py experiment=pe6a-DP-baseline ckpt_path=/path/to/last.ckpt
```

## Inference (evaluation)

```bash
# Evaluate a checkpoint on the test split
python src/eval.py \
    experiment=pe6a-DP-baseline \
    ckpt_path=/absolute/path/to/checkpoint.ckpt \
    logger=csv
```

The experiment config passed to `eval.py` determines which **data split** and
**model architecture** are used.  The `ckpt_path` must point to the `.ckpt` file
saved by the corresponding training run.

## Project structure

```
configs/
  data/            Data module configs
  experiment/      Per-experiment overrides (pe6a, pe6b, ...)
  model/           Model architecture configs
  trainer/         Trainer configs (cpu, gpu, ddp, …)
src/
  train.py         Training entry point
  eval.py          Inference / evaluation entry point
  data/            DataModule implementations
  models/          LightningModule implementations and components
scripts/
  train.sh         General-purpose training wrapper
  predict.sh       General-purpose evaluation wrapper
data/              Datasets (see Data section above)
```

## Environment variables

Copy `.env.example` to `.env` and fill in your credentials:

| Variable | Purpose |
|---|---|
| `WANDB_API_KEY` | Weights & Biases logging (optional) |
| `GENET_EPEGRNA_DIR` | Path to DeepPrime weight directory used by zero-shot script |

## Notes

- No secrets are committed; all credentials are loaded from environment variables.
- Checkpoints and Hydra run outputs are written under `logs/` (git-ignored).
