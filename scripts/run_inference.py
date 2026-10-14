#!/usr/bin/env python3
"""
CLI entrypoint to run canonical DeepPrime & DeepPrime6 inference on any dataset.

Usage:
    python scripts/run_inference.py --input data/rq3_test_library_dedup.parquet --output data/predictions.parquet --models all
"""

import os
import sys
import argparse
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.inference import run_comprehensive_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Run canonical DeepPrime / DeepPrime6 inference.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV, parquet, or Excel file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output CSV or parquet file")
    parser.add_argument("--models", "-m", type=str, default="all",
                        help="Comma-separated models: all, base, ft, scratch, pridict2 (default: all)")
    parser.add_argument("--batch_size", "-b", type=int, default=2048, help="GPU batch size")
    parser.add_argument("--device", "-d", type=str, default=None, help="Device (e.g. cuda:0 or cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    models_arg = [m.strip().lower() for m in args.models.split(",")]
    include_all = "all" in models_arg
    inc_base = include_all or "base" in models_arg or "dp_base" in models_arg
    inc_ft = include_all or "ft" in models_arg or "dp6_ft" in models_arg or "dp6" in models_arg
    inc_scratch = include_all or "scratch" in models_arg or "dp6_scratch" in models_arg
    inc_pridict2 = include_all or "pridict2" in models_arg

    print(f"Loading input data: {args.input}...")
    res_df = run_comprehensive_inference(
        df_or_path=args.input,
        include_base=inc_base,
        include_dp6_ft=inc_ft,
        include_dp6_scratch=inc_scratch,
        include_pridict2=inc_pridict2,
        batch_size=args.batch_size,
        device=args.device
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    if args.output.endswith(".parquet"):
        res_df.to_parquet(args.output, index=False)
    elif args.output.endswith(".csv"):
        res_df.to_csv(args.output, index=False)
    elif args.output.endswith(".xlsx"):
        res_df.to_excel(args.output, index=False)
    else:
        res_df.to_csv(args.output + ".csv", index=False)
    print(f"Predictions saved to {args.output} ({len(res_df):,} rows)")

if __name__ == "__main__":
    main()
