import pandas as pd
import numpy as np
import sys

def check_equality(file1, file2, value_col):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Sort just in case the order changed (though it shouldn't)
    if 'ID' in df1.columns:
        df1 = df1.sort_values('ID').reset_index(drop=True)
        df2 = df2.sort_values('ID').reset_index(drop=True)
    elif 'REF_ID' in df1.columns:
        df1 = df1.sort_values('REF_ID').reset_index(drop=True)
        df2 = df2.sort_values('REF_ID').reset_index(drop=True)
        
    try:
        pd.testing.assert_series_equal(df1[value_col], df2[value_col], check_exact=False, atol=1e-5, rtol=1e-5)
        print(f"VERIFIED: {file1} exactly matches {file2} on column {value_col}")
        return True
    except AssertionError as e:
        print(f"FAILED: {file1} does not match {file2} on column {value_col}")
        print(e)
        return False

def main():
    success = True
    print("--- Verifying Reproducibility of Inferences ---")
    
    # PRIDICT2
    if not check_equality("data/rq3_predictions_pridict2.csv", "data/rq3_predictions_pridict2.csv.bak", "PRIDICT2_0_editing_Score_deep_HEK"):
        success = False
    
    # DeepPrime Base
    if not check_equality("data/rq3_predictions_dp_base.csv", "data/rq3_predictions_dp_base.csv.bak", "Prediction"):
        success = False
        
    # DeepPrime6 PE6a
    if not check_equality("data/rq3_predictions_dp6.csv", "data/rq3_predictions_dp6.csv.bak", "Prediction"):
        success = False
        
    # DeepPrime6 PE6b
    if not check_equality("data/rq3_predictions_dp6_pe6b.csv", "data/rq3_predictions_dp6_pe6b.csv.bak", "Prediction"):
        success = False
        
    # DeepPrime6 PE6c
    if not check_equality("data/rq3_predictions_dp6_pe6c.csv", "data/rq3_predictions_dp6_pe6c.csv.bak", "Prediction"):
        success = False
        
    # DeepPrime6 dRNaseH
    if not check_equality("data/rq3_predictions_dp6_drnaseh.csv", "data/rq3_predictions_dp6_drnaseh.csv.bak", "Prediction"):
        success = False
        
    if success:
        print("\nSUCCESS: All inferences are perfectly reproducible! The newly calculated scores matched the cached scores exactly.")
        sys.exit(0)
    else:
        print("\nERROR: Reproducibility check failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()
