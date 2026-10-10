#!/usr/bin/env python3
"""
Master Training Script for Final 4 DeepPrime6 Models:
- PE6a (pe6a-DP-baseline)
- PE6b (pe6b-DP-baseline)
- PE6c (pe6c-DP-baseline)
- PEmaxdRNaseH (pemaxdrnaseh-DP-baseline)

Using:
- Deduplicated pegRNA dataset: data/processed_data_dedup.parquet
- Strict Guide(spacer) grouping split (Zero leakage across train/val/test)
- Best hyperparameters verified from Optuna fine-tuning search
- Automatic checkpoint archiving to src/models/weights/DeepPrime6-weights/
"""

import os
import sys
import glob
import shutil
import subprocess
import time
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

MODELS = [
    {
        "name": "pe6a",
        "exp": "pe6a-DP-baseline",
        "target_ckpt": "src/models/weights/DeepPrime6-weights/pe6a_mainft.ckpt",
    },
    {
        "name": "pe6b",
        "exp": "pe6b-DP-baseline",
        "target_ckpt": "src/models/weights/DeepPrime6-weights/pe6b_mainft.ckpt",
    },
    {
        "name": "pe6c",
        "exp": "pe6c-DP-baseline",
        "target_ckpt": "src/models/weights/DeepPrime6-weights/pe6c_mainft.ckpt",
    },
    {
        "name": "pemaxdrnaseh",
        "exp": "pemaxdrnaseh-DP-baseline",
        "target_ckpt": "src/models/weights/DeepPrime6-weights/pemaxdrnaseh_mainft.ckpt",
    },
]

PYTHON_EXEC = "/home/work/.conda/bin/python3"

def main():
    print("==========================================================================", flush=True)
    print("=== Master Re-Training Pipeline: 4 DeepPrime6 Models (Zero-Leakage) ===", flush=True)
    print("==========================================================================", flush=True)

    summary_results = []

    for model_info in MODELS:
        mname = model_info["name"]
        exp_name = model_info["exp"]
        dest_ckpt = model_info["target_ckpt"]

        print(f"\n>>> Starting training for {mname} (experiment={exp_name})...", flush=True)
        start_time = time.time()

        cmd = [
            PYTHON_EXEC,
            "src/train.py",
            f"experiment={exp_name}",
            "data.dataloader.num_workers=8",
            "data.dataloader.pin_memory=True",
            "data.dataloader.persistent_workers=True",
        ]
        print(f"Command: {' '.join(cmd)}", flush=True)

        res = subprocess.run(cmd, cwd=PROJECT_ROOT)
        elapsed = time.time() - start_time
        print(f">>> {mname} run finished with exit code {res.returncode} in {elapsed/60:.1f} minutes", flush=True)

        if res.returncode != 0:
            print(f"ERROR: Training failed for {mname}!", file=sys.stderr)
            sys.exit(res.returncode)

        # Find newest checkpoint in logs for this specific task created after start_time
        ckpt_pattern = os.path.join(PROJECT_ROOT, "logs", "*", "runs", "*", "checkpoints", "*.ckpt")
        all_ckpts = sorted(glob.glob(ckpt_pattern), key=os.path.getmtime)
        valid_ckpts = [c for c in all_ckpts if os.path.getmtime(c) >= start_time - 10]
        if not valid_ckpts:
            raise FileNotFoundError(f"No new checkpoint found for {mname} matching {ckpt_pattern} after start_time {start_time}")

        latest_ckpt = valid_ckpts[-1]
        print(f"Latest checkpoint found: {latest_ckpt} (size: {os.path.getsize(latest_ckpt)/(1024*1024):.1f} MB)", flush=True)

        os.makedirs(os.path.dirname(dest_ckpt), exist_ok=True)
        shutil.copy2(latest_ckpt, dest_ckpt)
        print(f"Archived best weights to: {dest_ckpt}", flush=True)

        summary_results.append({
            "model": mname,
            "experiment": exp_name,
            "duration_min": round(elapsed / 60, 2),
            "ckpt_path": dest_ckpt,
            "ckpt_size_mb": round(os.path.getsize(dest_ckpt)/(1024*1024), 1)
        })

    print("\n==========================================================================", flush=True)
    print("=== Training Pipeline Completed Successfully! ===", flush=True)
    print("==========================================================================", flush=True)
    df_summary = pd.DataFrame(summary_results)
    print(df_summary.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
