# DeepPrime6
Official repository for DeepPrime6: Deep learning-based prediction of prime editing efficiencies and transfer learning pipelines.

## Environment

### Option A: conda

```bash
conda env create -f environment.yaml
conda activate myenv
```

### Option B: pip

```bash
python -m pip install --upgrade pip
# Note: On some systems, you may need to bypass constraints for protobuf/tensorflow
PIP_CONSTRAINT="" python -m pip install -r requirements.txt
```

## Troubleshooting: Dependency Conflicts

If you encounter an error like `ResolutionImpossible` or conflicts between `genet` and `tensorflow`:

1. **Python 3.11+**: `genet` metadata strictly requires `tensorflow < 2.10.0`, but Python 3.11 requires `tensorflow >= 2.12.0`. We have verified that **`tensorflow 2.15.0`** works. To bypass strict dependency checks, install `genet` separately without dependencies:
   ```bash
   # 1. Install other dependencies (bypass system constraints if needed)
   PIP_CONSTRAINT="" pip install -r requirements.txt

   # 2. Install genet separately to avoid metadata conflicts
   PIP_CONSTRAINT="" pip install genet==0.15.1 --no-deps
   ```

2. **Protobuf Conflict**: If your environment forces `protobuf 4.x` (common in some managed environments), you must clear the `PIP_CONSTRAINT` variable as shown above.

## Data

The codebase supports training and evaluation on custom datasets provided in CSV or Parquet format.

### Custom Dataset Format

If you are providing your own dataset (e.g., `data/my_dataset.csv`), ensure your CSV file includes the following required columns for the preprocessing pipeline (`skip_preprocessing=False`):

(#TBD: update this with minimum required columns)
- **`WideTargetSequence`**: Extended sequence context around the target site (e.g., 200nt).
- **`Guide`**: The 20-nt spacer sequence.
- **`Edit_type`**: Type of edit (`Sub`, `Ins`, or `Del`).
- **`Edit length`**: Length of the intended edit (e.g., `1`).
- **`Edit position`**: Position of the edit relative to the nick site.
- **`PBS`**: Primer binding site sequence.
- **`RTT`**: Reverse transcriptase template sequence.
- **`OligoSequence_fixed_length`**: Context sequence from which the 74-nt sequence is extracted.
- **`leading G`**: Whether there is a leading G in the guide (e.g., `G` or `-`).
- **Target columns**: Efficiency target metric (e.g., `Normalized+3rep_HEK-M-3-7D+pe_ratio_%`).

*Note: The target columns are mapped to readable names like `PE6a(+PEmaxCas9)` in `src/utils/dataprep.py` using `RENAME_MAP` and `RENAME_MAP_FOR_VIS`. If your efficiency column is named differently, update these dictionaries or rename your column to match an existing key.*

If your data is **already preprocessed** and contains all DeepPrime feature columns (e.g., `Target`, `Masked_EditSeq`, `PBS_len`, thermodynamic/GC features, etc.), you can set `data.skip_preprocessing=True` in your Hydra configuration to bypass the preprocessing step.

### Example: Verifying the Pipeline

We provide a minimal verifiable dataset in `data/sample_data.csv`, which contains 10 example rows with all the required columns for PE6a(+PEmaxCas9) editing efficiencies.

To test the entire pipeline (preprocessing -> training -> evaluation) locally using this minimal dataset, run the following commands:

**1. Training**
```bash
# We set model.model_weights.baseline=null to train from scratch for testing
# We override the data_dir to point to the sample dataset
bash scripts/train.sh pe6a-DP-baseline \
    data.data_dir=data/sample_data.csv \
    trainer=cpu \
    data.batch_size=2 \
    model.model_weights.baseline=null \
    logger=csv \
    ~callbacks.rich_progress_bar
```

**2. Inference (Evaluation)**
After training, evaluate the generated checkpoint (replace the checkpoint path with your generated path):
```bash
bash scripts/predict.sh pe6a-DP-baseline \
    logs/PE6a-ft/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/epoch_xxx.ckpt \
    data.data_dir=data/sample_data.csv \
    trainer=cpu \
    data.batch_size=2 \
    logger=csv \
    ~callbacks.rich_progress_bar
```

## State-of-the-Art (SOTA) Models

The following table lists the final SOTA models reported in the paper. You can use the provided `scripts/predict.sh` script to evaluate these checkpoints or `scripts/train.sh` to fine-tune them further.

| PE type | Model | Run ID | Checkpoint path | Spearman |
|---|---|---:|---|---:|
| PE6a | PE6a (MainFT) | `bt1dgmi7` | `weights/pe6a_mainft.ckpt` | 0.659 |
| PEmaxdRNaseH | PEmaxdRNaseH (MainFT) | `85mculrb` | `weights/pemaxdrnaseh_mainft.ckpt` | 0.711 |
| PE6b | PE6b (MainFT) | `2hyiekc1` | `weights/pe6b_mainft.ckpt` | 0.685 |
| PE6c | PE6c (MainFT) | `5wu8hpi4` | `weights/pe6c_mainft.ckpt` | 0.694 |

### Additional Pre-trained Weights

We also provide models trained from scratch (without DeepPrime transfer learning) for comparison:

- **PE6a (Scratch):** `weights/pe6a_scratch.ckpt`
- **PE6b (Scratch):** `weights/pe6b_scratch.ckpt`
- **PE6c (Scratch):** `weights/pe6c_scratch.ckpt`
- **PEmaxdRNaseH (Scratch):** `weights/pemaxdrnaseh_scratch.ckpt`

## Quick Start

We provide simple bash wrapper scripts to abstract away Hydra configurations and quickly run training or inference.
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
# Evaluate the PE6a SOTA model on the test set
bash scripts/predict.sh pe6a-DP-baseline weights/pe6a_mainft.ckpt

# Evaluate with custom overrides (e.g., using CPU)
bash scripts/predict.sh pe6a-DP-baseline weights/pe6a_mainft.ckpt trainer=cpu
```

## Customizing Configurations

This project uses [Hydra](https://hydra.cc/) for configuration management.
We provide a single example experiment config (`configs/experiment/pe6a-DP-baseline.yaml`) as a reference.
You can use it as a template to create configs for other PE types or to customize training parameters.

### Creating a new experiment config for a different PE type

Copy the example and modify the PE-type-specific fields (PE6b as a example):

```bash
cp configs/experiment/pe6a-DP-baseline.yaml configs/experiment/pe6b-DP-baseline.yaml
```

Then edit the new file — the key fields to change are:

```yaml
# configs/experiment/pe6b-DP-baseline.yaml
task_name: "PE6b-ft"                      # ← run output directory name

tags: ["6b-ft", "deep-prime-FT"]          # ← experiment tags (for tracking and identification)

data:
    datafilter:
        PE_types: ["PE6b(+PEmaxCas9)"]    # ← target PE type to filter on your dataset
```

### Commonly overridden parameters

Any config value can be overridden from the command line without editing YAML files:

```bash
# Change accelerator
python src/train.py experiment=pe6a-DP-baseline trainer=cpu

# Change batch size and learning rate
python src/train.py experiment=pe6a-DP-baseline data.batch_size=128 model.optimizer.lr=1e-4

# Switch logger (csv or wandb)
python src/train.py experiment=pe6a-DP-baseline logger=csv

# Train from scratch (disable pretrained DeepPrime weights)
python src/train.py experiment=pe6a-DP-baseline model.model_weights.baseline=null

# Point to your own dataset
python src/train.py experiment=pe6a-DP-baseline data.data_dir=data/my_dataset.csv

# Resume from a checkpoint
python src/train.py experiment=pe6a-DP-baseline ckpt_path=/path/to/last.ckpt
```

### Config directory structure

```
configs/
  train.yaml              Root training config (defaults & global settings)
  eval.yaml               Root evaluation config
  data/
    pe6.yaml              DataModule config (splits, batch size, preprocessing)
  model/
    pe6_deep_prime_only_vanilla.yaml   Model architecture + optimizer + loss
  experiment/
    pe6a-DP-baseline.yaml Example experiment (use as template for other PE types)
  callbacks/
    default.yaml          Checkpoint, early stopping, progress bar
  trainer/
    default.yaml          GPU trainer (300 epochs, fp64)
    cpu.yaml              CPU override
    gpu.yaml              GPU override
  logger/
    csv.yaml              CSV logger (default, no external service needed)
    wandb.yaml            Weights & Biases logger (optional)
  paths/                  Root/data/log directory paths
  extras/                 Misc settings (warnings, config printing)
  hydra/                  Hydra output directory patterns
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

## Environment variables

Copy `.env.example` to `.env` and fill in your credentials:

| Variable | Purpose |
|---|---|
| `WANDB_API_KEY` | Weights & Biases logging (optional) |

## Notes

- No secrets are committed; all credentials are loaded from environment variables.
- Checkpoints and Hydra run outputs are written under `logs/` (git-ignored).
