#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization & Re-Training Pipeline for FROM-SCRATCH Models on Deduplicated Dataset (Zero-Leakage).

Features:
- Search space strictly aligned with configs/hparams_search/pe6-optuna.yaml
- From-scratch model architecture (baseline=null, requires_grad=true, num_ensemble=20)
- Evaluates on clean deduplicated zero-leakage splits (processed_data_dedup.parquet)
- Objective: Maximize val/pearson_best with 25 epochs per trial (same conditions as fine-tuning HPO)
- Bayesian optimization via Optuna TPESampler (seed=1234, n_startup_trials=10)
- Persistent JSONL logging (logs/hpo_scratch_{model}/trials.jsonl) with resume capability
- Automatic disk cleanup of temporary trial checkpoints to prevent disk overflow
- Synchronized YAML configuration updates to configs/experiment/from-scratch/
- Final Retraining: 20-ensemble From-Scratch models saved to src/models/weights/DeepPrime6-scratch-weights/
- Automated Refresh of all downstream benchmarks in notebooks/plots/ (Panel B, Figure 3, etc.)
"""

import os
import sys
import json
import glob
import shutil
import argparse
import subprocess
import time
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
        "exp": "from-scratch/pe6a-DP-from-scratch",
        "task_name": "PE6a-scratch",
        "scratch_yaml": "configs/experiment/from-scratch/pe6a-DP-from-scratch.yaml",
        "target_ckpt": "src/models/weights/DeepPrime6-scratch-weights/pe6a_scratch.ckpt",
    },
    "pe6b": {
        "exp": "from-scratch/pe6b-DP-from-scratch",
        "task_name": "PE6b-scratch",
        "scratch_yaml": "configs/experiment/from-scratch/pe6b-DP-from-scratch.yaml",
        "target_ckpt": "src/models/weights/DeepPrime6-scratch-weights/pe6b_scratch.ckpt",
    },
    "pe6c": {
        "exp": "from-scratch/pe6c-DP-from-scratch",
        "task_name": "PE6c-scratch",
        "scratch_yaml": "configs/experiment/from-scratch/pe6c-DP-from-scratch.yaml",
        "target_ckpt": "src/models/weights/DeepPrime6-scratch-weights/pe6c_scratch.ckpt",
    },
    "pemaxdrnaseh": {
        "exp": "from-scratch/pemaxdrnaseh-DP-from-scratch",
        "task_name": "PEmaxdRNaseH-scratch",
        "scratch_yaml": "configs/experiment/from-scratch/pemaxdrnaseh-DP-from-scratch.yaml",
        "target_ckpt": "src/models/weights/DeepPrime6-scratch-weights/pemaxdrnaseh_scratch.ckpt",
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

    trial_dir = os.path.join(PROJECT_ROOT, "logs", f"hpo_scratch_{model_key}", f"trial_{trial.number:03d}")
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

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_file = os.path.join(trial_dir, "trial.log")
    with open(log_file, "w") as f_out:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=f_out, stderr=subprocess.STDOUT, env=env)

    if res.returncode != 0:
        print(f"[{model_key.upper()} Scratch Trial {trial.number}] FAILED with code {res.returncode}. Check {log_file}", flush=True)
        return 0.0

    # Read metrics.csv to get best validation pearson
    metrics_files = glob.glob(os.path.join(trial_dir, "csv", "*", "metrics.csv"))
    if not metrics_files:
        print(f"[{model_key.upper()} Scratch Trial {trial.number}] No metrics.csv found in {trial_dir}", flush=True)
        return 0.0

    df_metrics = pd.read_csv(metrics_files[0])
    if "val/pearson_best" in df_metrics.columns:
        best_val = df_metrics["val/pearson_best"].dropna().max()
    elif "val/pearson" in df_metrics.columns:
        best_val = df_metrics["val/pearson"].dropna().max()
    else:
        best_val = 0.0

    best_val_float = float(best_val) if not pd.isna(best_val) else 0.0

    print(f"[{model_key.upper()} Scratch Trial {trial.number:02d}] Finished - val/pearson_best: {best_val_float:.4f} (lr: {lr:.5f}, bs: {batch_size}, T_0: {t_0}, seed: {seed})", flush=True)

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

def run_scratch_hpo_for_model(model_key, n_trials=50):
    print(f"\n==========================================================================", flush=True)
    print(f"=== Starting From-Scratch Optuna HPO for {model_key.upper()} ({n_trials} Trials) ===", flush=True)
    print(f"==========================================================================", flush=True)

    hpo_dir = os.path.join(PROJECT_ROOT, "logs", f"hpo_scratch_{model_key}")
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
    print(f"Model: {model_key.upper()} | Existing trials in JSONL: {existing_trials} | Remaining: {remaining_trials}", flush=True)

    if remaining_trials > 0:
        study.optimize(lambda trial: objective(trial, model_key, jsonl_path), n_trials=remaining_trials)

    print(f"\n=== {model_key.upper()} From-Scratch HPO Completed! ===", flush=True)
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Val Pearson: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best parameters to JSON
    out_json = os.path.join(PROJECT_ROOT, "logs", f"best_hparams_scratch_{model_key}.json")
    with open(out_json, "w") as f:
        json.dump({
            "model": model_key,
            "mode": "from-scratch",
            "best_trial": study.best_trial.number,
            "best_val_pearson": float(study.best_value),
            "best_params": study.best_params,
        }, f, indent=2)
    print(f"Saved best params to {out_json}", flush=True)

    # Update YAML config with new best parameters
    yaml_path = os.path.join(PROJECT_ROOT, MODEL_CONFIGS[model_key]["scratch_yaml"])
    if os.path.exists(yaml_path):
        update_yaml_config(yaml_path, study.best_params)

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

    # Ensure from-scratch model weight parameters are explicitly set
    if "model_weights" not in cfg["model"]:
        cfg["model"]["model_weights"] = {}
    cfg["model"]["model_weights"]["baseline"] = None
    cfg["model"]["model_weights"]["requires_grad"] = True
    cfg["model"]["model_weights"]["num_ensemble"] = 20

    with open(yaml_path, "w") as f:
        f.write("# @package _global_\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Successfully updated {yaml_path} with new optimal parameters!", flush=True)

def retrain_scratch_model(model_key):
    info = MODEL_CONFIGS[model_key]
    exp_name = info["exp"]
    target_ckpt = info["target_ckpt"]
    task_name = info["task_name"]

    print(f"\n>>> Retraining From-Scratch {model_key.upper()} ({task_name}) with new optimal hyperparameters...", flush=True)
    start_time = time.time()

    cmd = [
        PYTHON_EXEC,
        "src/train.py",
        f"experiment={exp_name}",
        "data.dataloader.num_workers=4",
        "data.dataloader.pin_memory=True",
        "data.dataloader.persistent_workers=false",
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print(f"Command: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Retraining failed for {model_key} (From-Scratch) with code {res.returncode}")

    # Find the newest checkpoint
    ckpt_pattern = os.path.join(PROJECT_ROOT, "logs", task_name, "runs", "*", "checkpoints", "*.ckpt")
    all_ckpts = sorted(glob.glob(ckpt_pattern), key=os.path.getmtime)
    valid_ckpts = [c for c in all_ckpts if os.path.getmtime(c) >= start_time - 10]
    if not valid_ckpts:
        fallback_pattern = os.path.join(PROJECT_ROOT, "logs", "*", "runs", "*", "checkpoints", "*.ckpt")
        all_ckpts = sorted(glob.glob(fallback_pattern), key=os.path.getmtime)
        valid_ckpts = [c for c in all_ckpts if os.path.getmtime(c) >= start_time - 10]

    if not valid_ckpts:
        raise FileNotFoundError(f"No new checkpoint found for {model_key} after start_time {start_time}")

    latest_ckpt = valid_ckpts[-1]
    print(f"Latest checkpoint found: {latest_ckpt} (size: {os.path.getsize(latest_ckpt)/(1024*1024):.1f} MB)", flush=True)

    os.makedirs(os.path.dirname(target_ckpt), exist_ok=True)
    shutil.copy2(latest_ckpt, target_ckpt)
    print(f"Archived From-Scratch weights to: {target_ckpt}", flush=True)

def refresh_downstream_evaluations():
    print("\n==========================================================================", flush=True)
    print("=== REFRESHING DOWNSTREAM EXPERIMENTS & BENCHMARK FIGURES ===", flush=True)
    print("==========================================================================", flush=True)

    # 1. Regenerate From-Scratch predictions on deduplicated test set
    gen_cmd = [PYTHON_EXEC, "scripts/evaluation/generate_scratch_predictions.py"]
    print(f"Running: {' '.join(gen_cmd)}", flush=True)
    res_gen = subprocess.run(gen_cmd, cwd=PROJECT_ROOT)
    if res_gen.returncode != 0:
        raise RuntimeError(f"generate_scratch_predictions failed with exit code {res_gen.returncode}")

    # 2. Re-run Track A analysis & visualization
    analysis_cmd = [PYTHON_EXEC, "src/utils/run_track_a_analysis.py"]
    print(f"Running: {' '.join(analysis_cmd)}", flush=True)
    res_analysis = subprocess.run(analysis_cmd, cwd=PROJECT_ROOT)
    if res_analysis.returncode != 0:
        raise RuntimeError(f"Track A analysis failed with exit code {res_analysis.returncode}")

    # 3. Re-package experiment zip archives
    pkg_cmd = [PYTHON_EXEC, "scripts/utils/package_experiments.py"]
    print(f"Running: {' '.join(pkg_cmd)}", flush=True)
    res_pkg = subprocess.run(pkg_cmd, cwd=PROJECT_ROOT)
    if res_pkg.returncode != 0:
        raise RuntimeError(f"Package experiments failed with exit code {res_pkg.returncode}")

    print("\n=== All benchmark results, panels, and packages refreshed successfully! ===", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Run Optuna HPO & Retraining for From-Scratch models on deduplicated dataset")
    parser.add_argument("--model", type=str, default="all", choices=["all", "pe6a", "pe6b", "pe6c", "pemaxdrnaseh"], help="Model to optimize")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials per model")
    parser.add_argument("--retrain", action="store_true", help="Retrain models after HPO")
    parser.add_argument("--refresh_only", action="store_true", help="Skip HPO/retrain and only refresh downstream evaluations")
    args = parser.parse_args()

    if args.refresh_only:
        refresh_downstream_evaluations()
        return

    models_to_run = ["pe6a", "pe6b", "pe6c", "pemaxdrnaseh"] if args.model == "all" else [args.model]

    all_results = {}
    for m in models_to_run:
        best_p, best_v = run_scratch_hpo_for_model(m, n_trials=args.n_trials)
        all_results[m] = {"best_val_pearson": best_v, "params": best_p}

    print("\n==========================================================================", flush=True)
    print("=== SUMMARY OF ALL FROM-SCRATCH HPO SEARCHES ===", flush=True)
    print("==========================================================================", flush=True)
    for m, res in all_results.items():
        print(f"Model: {m.upper()} (From-Scratch) | Best Val Pearson: {res['best_val_pearson']:.4f}")
        for pk, pv in res["params"].items():
            print(f"    {pk}: {pv}")

    if args.retrain:
        print("\n==========================================================================", flush=True)
        print("=== RETRAINING FROM-SCRATCH MODELS WITH NEW OPTIMAL HYPERPARAMETERS ===", flush=True)
        print("==========================================================================", flush=True)
        for m in models_to_run:
            retrain_scratch_model(m)
        print("\nAll From-Scratch Retraining Completed Successfully!", flush=True)

        refresh_downstream_evaluations()

if __name__ == "__main__":
    main()
