"""End-to-End (E2E) integration tests for Train, Eval, Predict, and Inference pipelines."""

import os
import subprocess
import sys
import pandas as pd

PYTHON_EXE = sys.executable
SAMPLE_DATA = "data/sample_data.csv"
SAMPLE_CKPT = "src/models/weights/DeepPrime6-weights/pe6a_mainft.ckpt"


def test_e2e_train_fast_dev_run(tmp_path):
    """Verifies that src/train.py executes end-to-end via Lightning Trainer (fast_dev_run)."""
    out_dir = str(tmp_path / "train_run")
    cmd = [
        PYTHON_EXE,
        "src/train.py",
        "experiment=pe6a-DP-baseline",
        f"data.data_dir={SAMPLE_DATA}",
        "data.skip_preprocessing=false",
        "trainer.fast_dev_run=True",
        "trainer.accelerator=cpu",
        "data.dataloader.num_workers=0",
        "data.dataloader.persistent_workers=false",
        f"paths.output_dir={out_dir}",
        f"model.prediction_save_path={out_dir}/test_predictions.csv",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"train.py failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert os.path.isdir(out_dir), "Output directory was not created."


def test_e2e_eval_pipeline(tmp_path):
    """Verifies that src/eval.py executes end-to-end to test a checkpoint on sample data."""
    out_dir = str(tmp_path / "eval_run")
    cmd = [
        PYTHON_EXE,
        "src/eval.py",
        "experiment=pe6a-DP-baseline",
        f"ckpt_path={SAMPLE_CKPT}",
        f"data.data_dir={SAMPLE_DATA}",
        "data.skip_preprocessing=false",
        "trainer.fast_dev_run=True",
        "trainer.accelerator=cpu",
        "data.dataloader.num_workers=0",
        "data.dataloader.persistent_workers=false",
        f"paths.output_dir={out_dir}",
        f"model.prediction_save_path={out_dir}/test_predictions.csv",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"eval.py failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert "Testing DataLoader 0" in res.stdout or "Test metric" in res.stdout


def test_e2e_predict_pipeline(tmp_path):
    """Verifies that src/predict.py executes end-to-end and outputs prediction CSV."""
    out_dir = str(tmp_path / "predict_run")
    cmd = [
        PYTHON_EXE,
        "src/predict.py",
        "experiment=pe6a-DP-baseline",
        f"ckpt_path={SAMPLE_CKPT}",
        f"data.data_dir={SAMPLE_DATA}",
        "data.skip_preprocessing=false",
        "trainer.accelerator=cpu",
        "data.dataloader.num_workers=0",
        "data.dataloader.persistent_workers=false",
        f"paths.output_dir={out_dir}",
        f"model.prediction_save_path={out_dir}/test_predictions.csv",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"predict.py failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert "Inference completed" in res.stdout
    assert os.path.exists(f"{out_dir}/test_predictions.csv"), "Prediction CSV was not written to out_dir."


def test_e2e_inference_cli(tmp_path):
    """Verifies that scripts/run_inference.py runs comprehensive inference on raw sample CSV."""
    out_csv = str(tmp_path / "sample_predictions.csv")
    cmd = [
        PYTHON_EXE,
        "scripts/run_inference.py",
        "--input", SAMPLE_DATA,
        "--models", "dp6",
        "--output", out_csv,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"run_inference.py failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert os.path.exists(out_csv), f"Output prediction file {out_csv} not generated."

    pred_df = pd.read_csv(out_csv)
    assert len(pred_df) == len(pd.read_csv(SAMPLE_DATA)), f"Expected {len(pd.read_csv(SAMPLE_DATA))} prediction rows, got {len(pred_df)}"
    assert "pred_dp6_pe6a" in pred_df.columns, f"'pred_dp6_pe6a' column missing from {pred_df.columns}"
    assert not pred_df["pred_dp6_pe6a"].isna().any(), "Predictions contain NaNs."
