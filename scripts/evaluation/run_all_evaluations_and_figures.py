#!/usr/bin/env python3
"""
Master Evaluation & Figure Generation Pipeline

Executes the complete post-training evaluation across all 4 key benchmarks:
1. Figure 3: Main library strictly unseen test set performance (PEmaxdRNaseH, PE6a, PE6b, PE6c vs DeepPrime-base)
2. Small Library: PE6 to PE7 orthogonal generalization benchmark (pe6to7_with_avgs)
3. FRT Library: 112 pegRNA design predictions & multi-model distribution analysis
4. Orthogonal Benchmarks: DeepPrime (Table S2-1) & PRIDICT2.0 (23k fold 0) strictly unseen benchmarks

Consolidates all metrics tables and publication-ready figures.
"""

import os
import sys
import subprocess
import time
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)

PYTHON_EXEC = "/home/work/.conda/bin/python3"

def run_step(step_name, cmd):
    print(f"\n{'='*75}", flush=True)
    print(f"=== Starting Step: {step_name} ===", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"{'='*75}", flush=True)

    t0 = time.time()
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"ERROR: {step_name} failed with exit code {res.returncode}!", file=sys.stderr)
        sys.exit(res.returncode)

    print(f"[Done] {step_name} completed in {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)

def main():
    print("==========================================================================", flush=True)
    print("=== Master Final Evaluation & Figure Generation Pipeline ===", flush=True)
    print("==========================================================================", flush=True)

    start_all = time.time()

    # 1. Figure 3: Main Library Test Set Performance
    run_step(
        "1. Figure 3 Main Library Test Set Performance",
        [PYTHON_EXEC, "scripts/evaluation/01_evaluate_figure3_main_library.py"]
    )

    # 2. Small Library: PE6 & PE7 Orthogonal Benchmark
    run_step(
        "2. Small Library Orthogonal Benchmark (pe6to7)",
        [PYTHON_EXEC, "scripts/evaluation/02_evaluate_small_library_pe6to7.py"]
    )

    # 3. FRT Small Library Predictions & Plots
    run_step(
        "3a. FRT Small Library Inference",
        [PYTHON_EXEC, "scripts/small_lib_predictions/predict_frt_small_library.py"]
    )
    run_step(
        "3b. FRT Small Library Plots",
        [PYTHON_EXEC, "scripts/small_lib_predictions/plot_frt_predictions.py"]
    )

    # 4. Orthogonal Benchmarks: DeepPrime & PRIDICT2.0
    run_step(
        "4a. Orthogonal Testsets Leakage Filtering",
        [PYTHON_EXEC, "scripts/orthogonal_evaluation/01_filter_testsets.py"]
    )
    run_step(
        "4b. Orthogonal Benchmark Inference",
        [PYTHON_EXEC, "scripts/orthogonal_evaluation/02_run_inference.py"]
    )
    run_step(
        "4c. Orthogonal Benchmark Visualizations",
        [PYTHON_EXEC, "scripts/orthogonal_evaluation/03_plot_benchmark_results.py"]
    )

    total_time = time.time() - start_all
    print("\n==========================================================================", flush=True)
    print(f"=== ALL FINAL EVALUATIONS AND FIGURES GENERATED IN {total_time/60:.2f} MINUTES! ===", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    main()
