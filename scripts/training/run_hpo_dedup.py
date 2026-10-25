#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization & Re-Training Pipeline on Deduplicated Dataset (Zero-Leakage).

Features:
- Search space strictly aligned with configs/hparams_search/pe6-optuna.yaml
- Evaluates on clean deduplicated zero-leakage splits (processed_data_dedup.parquet)
- Objective: Maximize val/pearson_best with EarlyStopping (patience: 5)
- Bayesian optimization via Optuna TPESampler
- Persistent JSONL logging (logs/hpo_{model}/trials.jsonl) with resume capability
- Automatic disk cleanup of temporary trial checkpoints to prevent disk overflow
- Synchronized YAML configuration updates across configs/experiment/ and configs/experiment/from-scratch/
- Final Retraining: Fine-Tuning + From-Scratch (20-ensemble) with new optimal hyperparameters
- Automated Track A Benchmark & Figure 3 reproduction (Panel B & C)
"""

import os
import sys
import json
import glob
import shutil
import argparse
import subprocess
import yaml
import optuna
import pandas as pd

def find_project_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".project-root")) or (
            os.path.exists(os.path.join(curr, "src")) and os.path.exists(os.path.join(curr, "configs"))
        ):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.getcwd())

PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PYTHON_EXEC = os.environ.get("PYTHON_EXEC", sys.executable)

MODEL_CONFIGS = {
    "pe6a": {
        "exp": "pe6a-DP-baseline",
        "task_name": "PE6a-ft",
        "yaml_path": "configs/experiment/pe6a-DP-baseline.yaml",
        "scratch_yaml": "configs/experiment/from-scratch/pe6a-DP-from-scratch.yaml",
        "target_ckpt": "src/models/weights/DeepPrime6-weights/pe6a_mainft.ckpt",
        "scratch_ckpt": "src/models/weights/DeepPrime6-scratch-weights/pe6a_scratch.ckpt",
    },
    "pe6b": {
        "exp": "pe6b-DP-baseline",
        "task_name": "PE6b-ft",
        "yaml_path": "configs/experiment/pe6b-DP-baseline.yaml",
        "scratch_yaml": "configs/experiment/from-scratch/pe6b-DP-from-scratch.yaml",
        "target_ckpt": "src/models/weights/DeepPrime6-weights/pe6b_mainft.ckpt",
        "scratch_ckpt": "src/models/weights/DeepPrime6-scratch-weights/pe6b_scratch.ckpt",
    },
    "pe6c": {
        "exp": "pe6c-DP-baseline",
        "task_name": "PE6c-ft",
        "yaml_path": "configs/experiment/pe6c-DP-baseline.yaml",
        "scratch_yaml": "configs/experiment/from-scratch/pe6c-DP-from-scratch.yaml",
        "target_ckpt": "src/models/weights/DeepPrime6-weights/pe6c_mainft.ckpt",
        "scratch_ckpt": "src/models/weights/DeepPrime6-scratch-weights/pe6c_scratch.ckpt",
    },
    "pemaxdrnaseh": {
        "exp": "pemaxdrnaseh-DP-baseline",
        "task_name": "PEmaxdRNaseH-ft",
        "yaml_path": "configs/experiment/pemaxdrnaseh-DP-baseline.yaml",
        "scratch_yaml": "configs/experiment/from-scratch/pemaxdrnaseh-DP-from-scratch.yaml",
        "target_ckpt": "src/models/weights/DeepPrime6-weights/pemaxdrnaseh_mainft.ckpt",
        "scratch_ckpt": "src/models/weights/DeepPrime6-scratch-weights/pemaxdrnaseh_scratch.ckpt",
    },
}

def objective(trial, model_key, jsonl_path):
    info = MODEL_CONFIGS[model_key]
    exp_name = info["exp"]

    # Sample hyperparameters according to configs/hparams_search/pe6-optuna.yaml
    seed = trial.suggest_categorical("seed", [42, 1234, 2024])
    batch_size = trial.suggest_categorical("data.batch_size", [64, 128, 256])
    lr = trial.suggest_float("model.optimizer.lr", 0.0001, 0.01, log=True)
    t_0 = trial.suggest_categorical("model.scheduler.T_0", [10, 20, 30, 40, 50])
    t_mult = trial.suggest_categorical("model.scheduler.T_mult", [1, 2])
    scale_threshold = trial.suggest_float("model.criterion.scale_threshold", 2.0, 4.0)
    min_scale = trial.suggest_float("model.criterion.minimum_scaling_factor", 4.0, 6.0)
    alpha = trial.suggest_float("model.criterion.alpha", 5.0, 7.0)

    trial_dir = os.path.join(PROJECT_ROOT, "logs", f"hpo_{model_key}", f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    cmd = [
        PYTHON_EXEC,
        "src/train.py",
        f"experiment={exp_name}",
        f"seed={seed}",
        f"data.batch_size={batch_size}",
        f"model.optimizer.lr={lr:.7f}",
        f"model.scheduler.T_0={t_0}",
        f"model.scheduler.T_mult={t_mult}",
        f"model.criterion.scale_threshold={scale_threshold:.4f}",
        f"model.criterion.minimum_scaling_factor={min_scale:.4f}",
        f"model.criterion.alpha={alpha:.4f}",
        "data.dataloader.num_workers=4",
        "data.dataloader.persistent_workers=false",
        "trainer.max_epochs=25",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "test=false",
        f"hydra.run.dir={trial_dir}",
    ]

    log_file = os.path.join(trial_dir, "trial.log")
    with open(log_file, "w") as f_out:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=f_out, stderr=subprocess.STDOUT)

    if res.returncode != 0:
        print(f"[{model_key} Trial {trial.number}] FAILED with code {res.returncode}. Check {log_file}", flush=True)
        return 0.0

    # Read metrics.csv to get best validation pearson
    metrics_files = glob.glob(os.path.join(trial_dir, "csv", "*", "metrics.csv"))
    if not metrics_files:
        print(f"[{model_key} Trial {trial.number}] No metrics.csv found in {trial_dir}", flush=True)
        return 0.0

    df_metrics = pd.read_csv(metrics_files[0])
    if "val/pearson_best" in df_metrics.columns:
        best_val = df_metrics["val/pearson_best"].dropna().max()
    elif "val/pearson" in df_metrics.columns:
        best_val = df_metrics["val/pearson"].dropna().max()
    else:
        best_val = 0.0

    best_val_float = float(best_val) if not pd.isna(best_val) else 0.0

    print(f"[{model_key} Trial {trial.number:02d}] Finished - val/pearson_best: {best_val_float:.4f} (lr: {lr:.5f}, bs: {batch_size}, T_0: {t_0}, seed: {seed})", flush=True)

    # Persist trial result to JSONL
    trial_record = {
        "trial_number": trial.number,
        "val_pearson_best": best_val_float,
        "params": trial.params,
    }
    with open(jsonl_path, "a") as f_jl:
        f_jl.write(json.dumps(trial_record) + "\n")

    # Disk cleanup: Delete intermediate checkpoints to avoid disk buildup
    ckpt_dir = os.path.join(trial_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    return best_val_float

def run_hpo_for_model(model_key, n_trials=50):
    print(f"\n==========================================================================", flush=True)
    print(f"=== Starting Optuna HPO for {model_key.upper()} ({n_trials} Trials) ===", flush=True)
    print(f"==========================================================================", flush=True)

    hpo_dir = os.path.join(PROJECT_ROOT, "logs", f"hpo_{model_key}")
    os.makedirs(hpo_dir, exist_ok=True)
    jsonl_path = os.path.join(hpo_dir, "trials.jsonl")

    sampler = optuna.samplers.TPESampler(seed=1234, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Resume past trials from JSONL if exists
    existing_trials = 0
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f_jl:
            for line in f_jl:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    study.add_trial(
                        optuna.trial.create_trial(
                            params=rec["params"],
                            distributions={
                                "seed": optuna.distributions.CategoricalDistribution([42, 1234, 2024]),
                                "data.batch_size": optuna.distributions.CategoricalDistribution([64, 128, 256]),
                                "model.optimizer.lr": optuna.distributions.LogUniformDistribution(0.0001, 0.01),
                                "model.scheduler.T_0": optuna.distributions.CategoricalDistribution([10, 20, 30, 40, 50]),
                                "model.scheduler.T_mult": optuna.distributions.CategoricalDistribution([1, 2]),
                                "model.criterion.scale_threshold": optuna.distributions.UniformDistribution(2.0, 4.0),
                                "model.criterion.minimum_scaling_factor": optuna.distributions.UniformDistribution(4.0, 6.0),
                                "model.criterion.alpha": optuna.distributions.UniformDistribution(5.0, 7.0),
                            },
                            value=rec["val_pearson_best"],
                        )
                    )
                    existing_trials += 1
                except Exception as e:
                    print(f"Warning: could not parse record: {line} ({e})")

    remaining_trials = max(0, n_trials - existing_trials)
    print(f"Model: {model_key} | Existing trials in JSONL: {existing_trials} | Remaining: {remaining_trials}", flush=True)

    if remaining_trials > 0:
        study.optimize(lambda trial: objective(trial, model_key, jsonl_path), n_trials=remaining_trials)

    print(f"\n=== {model_key.upper()} HPO Completed! ===", flush=True)
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Val Pearson: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best parameters to JSON
    out_json = os.path.join(PROJECT_ROOT, "logs", f"best_hparams_{model_key}.json")
    with open(out_json, "w") as f:
        json.dump({
            "model": model_key,
            "best_trial": study.best_trial.number,
            "best_val_pearson": float(study.best_value),
            "best_params": study.best_params,
        }, f, indent=2)
    print(f"Saved best params to {out_json}", flush=True)

    # Update YAML configs with new best parameters
    for y_key in ["yaml_path", "gei21_yaml", "scratch_yaml"]:
        if y_key in MODEL_CONFIGS[model_key]:
            yp = os.path.join(PROJECT_ROOT, MODEL_CONFIGS[model_key][y_key])
            if os.path.exists(yp):
                update_yaml_config(yp, study.best_params)

    return study.best_params, study.best_value

def update_yaml_config(yaml_path, best_params):
    print(f"Updating YAML config at: {yaml_path}", flush=True)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "model" not in cfg: cfg["model"] = {}
    if "optimizer" not in cfg["model"]: cfg["model"]["optimizer"] = {}
    if "scheduler" not in cfg["model"]: cfg["model"]["scheduler"] = {}
    if "criterion" not in cfg["model"]: cfg["model"]["criterion"] = {}
    if "data" not in cfg: cfg["data"] = {}

    cfg["model"]["optimizer"]["lr"] = float(round(best_params["model.optimizer.lr"], 7))
    cfg["model"]["scheduler"]["T_0"] = int(best_params["model.scheduler.T_0"])
    cfg["model"]["scheduler"]["T_mult"] = int(best_params["model.scheduler.T_mult"])
    cfg["model"]["criterion"]["scale_threshold"] = float(round(best_params["model.criterion.scale_threshold"], 4))
    cfg["model"]["criterion"]["minimum_scaling_factor"] = float(round(best_params["model.criterion.minimum_scaling_factor"], 4))
    cfg["model"]["criterion"]["alpha"] = float(round(best_params["model.criterion.alpha"], 4))
    cfg["data"]["batch_size"] = int(best_params["data.batch_size"])
    cfg["seed"] = int(best_params["seed"])

    # Ensure dataset is always the clean deduplicated zero-leakage dataset
    cfg["data"]["data_dir"] = "${paths.data_dir}/processed_data_dedup.parquet"
    cfg["data"]["skip_preprocessing"] = True
    if "datasplit" not in cfg["data"]:
        cfg["data"]["datasplit"] = {}
    cfg["data"]["datasplit"]["group_column"] = "Guide"
    cfg["data"]["datasplit"]["test_size"] = 0.2
    cfg["data"]["datasplit"]["val_size"] = 0.25

    with open(yaml_path, "w") as f:
        f.write("# @package _global_\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Successfully updated {yaml_path} with new optimal parameters!", flush=True)

def retrain_model(model_key, from_scratch=False):
    info = MODEL_CONFIGS[model_key]
    exp_name = info["exp"]
    target_ckpt = info["scratch_ckpt"] if from_scratch else info["target_ckpt"]
    mode_name = "From-Scratch" if from_scratch else "Fine-Tuning"

    print(f"\n>>> Retraining {model_key.upper()} [{mode_name}] with updated optimal hyperparameters...", flush=True)
    import time
    start_time = time.time()

    cmd = [
        PYTHON_EXEC,
        "src/train.py",
        f"experiment={exp_name}",
        "data.dataloader.num_workers=4",
        "data.dataloader.pin_memory=True",
        "data.dataloader.persistent_workers=false",
    ]
    if from_scratch:
        cmd.extend([
            "model.model_weights.baseline=null",
            "model.model_weights.requires_grad=true",
            "+model.model_weights.num_ensemble=20",
        ])

    print(f"Command: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"Retraining failed for {model_key} ({mode_name}) with code {res.returncode}")

    # Find the newest checkpoint
    ckpt_pattern = os.path.join(PROJECT_ROOT, "logs", "*", "runs", "*", "checkpoints", "*.ckpt")
    all_ckpts = sorted(glob.glob(ckpt_pattern), key=os.path.getmtime)
    valid_ckpts = [c for c in all_ckpts if os.path.getmtime(c) >= start_time - 10]
    if not valid_ckpts:
        raise FileNotFoundError(f"No new checkpoint found for {model_key} matching {ckpt_pattern} after start_time {start_time}")

    latest_ckpt = valid_ckpts[-1]
    print(f"Latest checkpoint found: {latest_ckpt} (size: {os.path.getsize(latest_ckpt)/(1024*1024):.1f} MB)", flush=True)

    os.makedirs(os.path.dirname(target_ckpt), exist_ok=True)
    shutil.copy2(latest_ckpt, target_ckpt)
    print(f"Archived {mode_name} weights to: {target_ckpt}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Run Optuna HPO & Retraining on deduplicated dataset")
    parser.add_argument("--model", type=str, default="all", choices=["all", "pe6a", "pe6b", "pe6c", "pemaxdrnaseh"], help="Model to optimize")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials per model")
    parser.add_argument("--retrain", action="store_true", help="Retrain models after HPO")
    args = parser.parse_args()

    models_to_run = ["pe6a", "pe6b", "pe6c", "pemaxdrnaseh"] if args.model == "all" else [args.model]

    all_results = {}
    for m in models_to_run:
        best_p, best_v = run_hpo_for_model(m, n_trials=args.n_trials)
        all_results[m] = {"best_val_pearson": best_v, "params": best_p}

    print("\n==========================================================================", flush=True)
    print("=== SUMMARY OF ALL HPO SEARCHES ===", flush=True)
    print("==========================================================================", flush=True)
    for m, res in all_results.items():
        print(f"Model: {m.upper()} | Best Val Pearson: {res['best_val_pearson']:.4f}")
        for pk, pv in res["params"].items():
            print(f"    {pk}: {pv}")

    if args.retrain:
        print("\n==========================================================================", flush=True)
        print("=== RETRAINING ALL MODELS WITH NEW OPTIMAL HYPERPARAMETERS ===", flush=True)
        print("==========================================================================", flush=True)
        for m in models_to_run:
            retrain_model(m, from_scratch=False)
            retrain_model(m, from_scratch=True)
        print("\nAll Retraining Completed Successfully!", flush=True)

        print("\n==========================================================================", flush=True)
        print("=== REPRODUCING TRACK A BENCHMARK & FIGURE 3 (PANEL B & C) ===", flush=True)
        print("==========================================================================", flush=True)
        reproduce_cmd = ["bash", "scripts/reproduce_track_a_figures.sh", "--fresh"]
        print(f"Running: {' '.join(reproduce_cmd)}", flush=True)
        res_rep = subprocess.run(reproduce_cmd, cwd=PROJECT_ROOT)
        if res_rep.returncode != 0:
            raise RuntimeError(f"Track A reproduction failed with exit code {res_rep.returncode}")
        print("\n=== Track A & Figure 3 Reproduced Successfully! ===", flush=True)

if __name__ == "__main__":
    main()
