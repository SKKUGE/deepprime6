import pandas as pd
import numpy as np
import sys
import os
import traceback

pridict2_dir = os.path.abspath("PRIDICT2")
sys.path.append(pridict2_dir)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.utils.predict_pridict2 import extract_single_pegRNA_features

def test_pridict():
    df = pd.read_parquet('data/rq3_test_library.parquet')
    
    pr_input_df = pd.DataFrame()
    pr_input_df['REF_ID'] = df['REF_ID']
    pr_input_df['WildTypeSequence'] = df['WildTypeSequence']
    pr_input_df['PrimeEditedSequence'] = df['PrimeEditedSequence']
    pr_input_df['Edit_type'] = df['Edit_type']
    pr_input_df['Guide'] = df['Guide']
    pr_input_df['PBS'] = df['PBS']
    pr_input_df['RTT'] = df['RTT']

    pred_df = pd.read_csv("data/rq3_predictions_pridict2.csv")
    zeros = (pred_df['PRIDICT2_0_editing_Score_deep_HEK'] == 0).sum()
    print(f"Number of 0s in rq3_predictions_pridict2.csv: {zeros}")
    
    merged = pd.merge(pr_input_df, pred_df, on='REF_ID', how='inner')
    zero_rows = merged[merged['PRIDICT2_0_editing_Score_deep_HEK'] == 0]
    print(f"Number of 0s mapped: {len(zero_rows)}")
    
    exception_in_zero = 0
    non_exceptions = 0
    
    for idx, row in zero_rows.iterrows():
        try:
            feat = extract_single_pegRNA_features(row)
            if feat is None:
                exception_in_zero += 1
            else:
                non_exceptions += 1
        except Exception as e:
            exception_in_zero += 1
            if exception_in_zero <= 5:
                print(f"Exception for REF_ID {row['REF_ID']}: {e}")
                traceback.print_exc()
            
    print(f"Among the {len(zero_rows)} zeros, {exception_in_zero} failed feature extraction.")
    print(f"Among the {len(zero_rows)} zeros, {non_exceptions} succeeded feature extraction.")

if __name__ == '__main__':
    test_pridict()
