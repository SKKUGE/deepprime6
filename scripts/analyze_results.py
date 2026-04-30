import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def analyze(file_path, label):
    try:
        df = pd.read_csv(file_path)
        if 'Target' not in df.columns or 'Prediction' not in df.columns:
            print(f"Error: {file_path} missing Target or Prediction columns")
            return
        
        target = df['Target']
        pred = df['Prediction']
        
        p_corr, _ = pearsonr(target, pred)
        s_corr, _ = spearmanr(target, pred)
        rmse = np.sqrt(np.mean((target - pred)**2))
        
        print(f"\n--- Analysis for {label} ({file_path}) ---")
        print(f"Pearson Correlation:  {p_corr:.4f}")
        print(f"Spearman Correlation: {s_corr:.4f}")
        print(f"RMSE:                 {rmse:.4f}")
        print(f"Samples:             {len(df)}")
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze DeepPrime6 prediction results")
    parser.add_argument("--predictions", type=str, help="Path to predictions CSV")
    parser.add_argument("--label", type=str, default="Prediction Results", help="Label for the analysis")
    
    args = parser.parse_args()
    
    if args.predictions:
        analyze(args.predictions, args.label)
    else:
        # Default behavior for verification
        analyze("data/rq3_predictions.csv", "Sample Predictions")
        # Legacy checks if they exist
        if os.path.exists("test_predictions.csv"):
            analyze("test_predictions.csv", "PE6a (Baseline)")
        if os.path.exists("pe6c_test_predictions.csv"):
            analyze("pe6c_test_predictions.csv", "PE6c (Fine-tuned)")
