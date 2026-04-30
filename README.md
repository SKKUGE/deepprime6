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

- **`ID`**: Unique identifier for the sample (optional, used for logging).
- **`WideTargetSequence`**: Extended sequence context around the target site (at least 21 nt upstream and 53 nt downstream from nick site).
- **`Guide`**: The 19 or 20-nt spacer sequence.
- **`Edit_type`**: Type of edit (`Sub`, `Ins`, or `Del`).
- **`Edit length`**: Length of the intended edit (e.g., `1`).
- **`Edit position`**: Position of the edit relative to the nick site.
- **`PBS`**: Primer binding site sequence.
- **`RTT`**: Reverse transcriptase template sequence.
- **`leading G`**: Whether there is a leading G in the guide (`G` or `-`).
- **Target columns**: Efficiency target metrics. The pipeline expects original research names which are mapped to readable names like `PE6a(+PEmaxCas9)` in `src/utils/dataprep.py`.
    - Example for PE6a: `Normalized+3rep_HEK-M-3-7D+pe_ratio_%`
    - Example for read counts (used for filtering): `HEK-M-3-7D-UAR+total_read_counts`

*Note: The target columns are mapped to readable names like `PE6a(+PEmaxCas9)` in `src/utils/dataprep.py` using `RENAME_MAP` and `RENAME_MAP_FOR_VIS`. If your efficiency column is named differently, update these dictionaries or rename your column to match an existing key.*

If your data is **already preprocessed** and contains all DeepPrime feature columns (e.g., `Target`, `Masked_EditSeq`, `PBS_len`, thermodynamic/GC features, etc.), you can set `data.skip_preprocessing=True` in your Hydra configuration to bypass the preprocessing step.

### Usage Guide

We provide simple bash wrapper scripts to abstract away Hydra configurations and quickly run training, evaluation, and inference.

### 1. Training (Train)
Train a model using a predefined experiment configuration.

```bash
# General training (defaults to GPU if available)
bash scripts/train.sh pe6a-DP-baseline

# Training with sample data on CPU (for verification)
bash scripts/train.sh pe6a-DP-baseline \
    data.data_dir=data/sample_data.csv \
    trainer=cpu \
    data.batch_size=2 \
    model.model_weights.baseline=null
```

### 2. Evaluation (Test)
Evaluate a trained model on a test set (requires ground truth labels). This uses `trainer.test()` under the hood.

```bash
# Evaluate the ensemble baseline models (*.pt)
bash scripts/eval.sh pe6a-DP-baseline \
    --ckpt "src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/*.pt" \
    trainer=cpu

# Evaluate a specific Lightning checkpoint (.ckpt)
bash scripts/eval.sh pe6a-DP-baseline \
    --ckpt logs/PE6a-ft/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/epoch_xxx.ckpt
```

### 3. Inference (Predict)
Run inference on new data where labels are not required. This uses `trainer.predict()` and saves predictions to a CSV file.

```bash
# Run inference with the SOTA fine-tuned model
bash scripts/predict.sh pe6a-DP-baseline \
    src/models/weights/DeepPrime6-weights/pe6a_mainft.ckpt

# Run inference on custom data
bash scripts/predict.sh pe6a-DP-baseline \
    src/models/weights/DeepPrime6-weights/pe6a_mainft.ckpt \
    data.data_dir=data/my_new_data.csv
```

## Model Weights

The following table lists the model weights reported in the paper. You can use the provided `scripts/predict.sh` script to evaluate these checkpoints or `scripts/train.sh` to fine-tune them further.


[Model weights (move to Google Drive)][1]

| PE type | Checkpoint path |
|---|---:|
| PE6a | `src/models/weights/DeepPrime6-weights/pe6a_mainft.ckpt` |
| PEmaxdRNaseH | `src/models/weights/DeepPrime6-weights/pemaxdrnaseh_mainft.ckpt` |
| PE6b | `src/models/weights/DeepPrime6-weights/pe6b_mainft.ckpt` |
| PE6c | `src/models/weights/DeepPrime6-weights/pe6c_mainft.ckpt` |

[1]: https://drive.google.com/file/d/1L9IRg5CMOv_NA2mFYY1aARYEBT-BTP43/view?usp=sharing

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

### Hyperparameter Optimization
```bash
nohup python src/train.py -m \
  experiment=pe6a-DP-baseline \
  hparams_search=pe6-optuna \
  >> output-pe6a-optuna.log 2>&1 &
```


### Config directory structure

```
configs/
  train.yaml              Root training config (defaults & global settings)
  eval.yaml               Root evaluation config
  data/
    pe6.yaml              DataModule config (splits, batch size, preprocessing)
  model/
    pe6_deep_prime_only.yaml           Model architecture + optimizer + loss
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



## Environment variables

Copy `.env.example` to `.env` and fill in your credentials:

| Variable | Purpose |
|---|---|
| `WANDB_API_KEY` | Weights & Biases logging (optional) |

## Notes

- No secrets are committed; all credentials are loaded from environment variables.
- Checkpoints and Hydra run outputs are written under `logs/` (git-ignored).
