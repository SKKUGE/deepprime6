#!/usr/bin/env python3
"""
train_and_evaluate_clean_scratch.py

1. Trains 4 prime editor models (PE2max-dRNaseH, PE6a, PE6b, PE6c) from scratch
   on the clean deduplicated training split (leakage-free on Guide).
2. Copies best checkpoints to src/models/weights/DeepPrime6-scratch-weights/.
3. Runs inference on unseen test set (rq3_test_library.csv) producing:
   - data/rq3_predictions_scratch_drnaseh.csv
   - data/rq3_predictions_scratch_pe6a.csv
   - data/rq3_predictions_scratch_pe6b.csv
   - data/rq3_predictions_scratch_pe6c.csv
4. Re-runs src/utils/run_track_a_analysis.py to update benchmark plots and summary CSVs.
"""

import os
import sys
import glob
import shutil
import subprocess
import time

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

PYTHON_EXEC = sys.executable

MODELS = [
    {
        "name": "pemaxdrnaseh",
        "experiment": "from-scratch/pemaxdrnaseh-DP-from-scratch",
        "task_name": "PEmaxdRNaseH-scratch",
        "ckpt_name": "pemaxdrnaseh_scratch.ckpt",
        "pred_csv": "data/rq3_predictions_scratch_drnaseh.csv",
    },
    {
        "name": "pe6a",
        "experiment": "from-scratch/pe6a-DP-from-scratch",
        "task_name": "PE6a-scratch",
        "ckpt_name": "pe6a_scratch.ckpt",
        "pred_csv": "data/rq3_predictions_scratch_pe6a.csv",
    },
    {
        "name": "pe6b",
        "experiment": "from-scratch/pe6b-DP-from-scratch",
        "task_name": "PE6b-scratch",
        "ckpt_name": "pe6b_scratch.ckpt",
        "pred_csv": "data/rq3_predictions_scratch_pe6b.csv",
    },
    {
        "name": "pe6c",
        "experiment": "from-scratch/pe6c-DP-from-scratch",
        "task_name": "PE6c-scratch",
        "ckpt_name": "pe6c_scratch.ckpt",
        "pred_csv": "data/rq3_predictions_scratch_pe6c.csv",
    },
]

def run_command(cmd, desc):
    print(f"\n==========================================")
    print(f"[*] {desc}")
    print(f"Command: {cmd}")
    print(f"==========================================")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    res = subprocess.run(cmd, shell=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")

def find_latest_checkpoint(task_name, start_time=0):
    pattern = os.path.join("logs", task_name, "runs", "*", "checkpoints", "*.ckpt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found matching {pattern}")
    # Sort by modification time descending, prioritizing checkpoints created after start_time
    valid = [c for c in ckpts if os.path.getmtime(c) >= start_time - 10]
    if valid:
        valid.sort(key=os.path.getmtime, reverse=True)
        return valid[0]
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]

def main():
    scratch_weights_dir = "src/models/weights/DeepPrime6-scratch-weights"
    os.makedirs(scratch_weights_dir, exist_ok=True)
    
    # 1. Train all 4 models from scratch
    for m in MODELS:
        print(f"\n>>> Starting From Scratch training for: {m['name']} ({m['task_name']})")
        start_time = time.time()
        train_cmd = (
            f"{PYTHON_EXEC} src/train.py "
            f"experiment={m['experiment']} "
            f"model.model_weights.baseline=null "
            f"task_name={m['task_name']} "
            f"tags='[\"{m['name']}-scratch\", \"scratch\"]' "
            f"trainer.accelerator=gpu "
            f"data.dataloader.num_workers=8"
        )
        run_command(train_cmd, f"Training {m['name']} from scratch")
        
        # Find best checkpoint created during this run
        best_ckpt = find_latest_checkpoint(m['task_name'], start_time)
        target_ckpt = os.path.join(scratch_weights_dir, m['ckpt_name'])
        print(f"[*] Best checkpoint found: {best_ckpt}")
        print(f"[*] Copying to: {target_ckpt}")
        shutil.copy2(best_ckpt, target_ckpt)
        
    # 2. Run inference for all 4 models on unseen test set using canonical script
    gen_cmd = f"{PYTHON_EXEC} scripts/evaluation/generate_scratch_predictions.py"
    run_command(gen_cmd, "Generating clean predictions for From-Scratch models")
        
    # 3. Re-run analysis & visualization
    analysis_cmd = f"{PYTHON_EXEC} src/utils/run_track_a_analysis.py"
    run_command(analysis_cmd, "Re-running Track A Analysis & Visualization")
    
    print("\n==========================================")
    print("[SUCCESS] All clean From Scratch models trained and evaluated!")
    print("==========================================")

if __name__ == "__main__":
    main()
