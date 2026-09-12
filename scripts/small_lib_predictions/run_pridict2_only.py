import pandas as pd
import numpy as np
import sys
import os

# Add PRIDICT2 to path
sys.path.append('PRIDICT2')
from src.utils.predict_pridict2 import load_pridict_model, deeppridict, compute_average_predictions, extract_single_pegRNA_features

def main():
    input_file = "data/small_library/pe6to7_predictions.parquet"
    df = pd.read_parquet(input_file)
    
    pr_input_df = df.copy()
    def get_pr_edit_type(row):
        if row['Edit_type'] == 'sub': return 'Sub'
        elif row['Edit_type'] == 'del': return 'Del'
        elif row['Edit_type'] == 'ins': return 'Ins'
        else: return 'Sub'
    pr_input_df['Edit_type'] = df.apply(get_pr_edit_type, axis=1)
    pr_input_df['REF'] = df['OriginalBase']
    pr_input_df['ALT'] = df['EditedBase']
    pr_input_df['nick_position'] = df['Nicking']
    pr_input_df['WT_context'] = df['WildTypeSequence']
    pr_input_df['REF_ID'] = df['ID']
    
    pridict_models = load_pridict_model([0])
    df['pred_pridict2'] = np.nan
    
    features_list_pr = []
    
    for idx, row in pr_input_df.iterrows():
        try:
            feat = extract_single_pegRNA_features(row)
            if feat is not None:
                features_list_pr.append(feat)
        except Exception as e:
            pass
            
    if len(features_list_pr) > 0:
        pegdataframe = pd.DataFrame(features_list_pr)
        for col in ['wide_initial_target', 'wide_mutated_target', 'deepeditposition', 
                    'deepeditposition_lst', 'Correction_Type', 'Correction_Length', 
                    'protospacerlocation_only_initial', 'PBSlocation',
                    'RT_initial_location', 'RT_mutated_location',
                    'RToverhangmatches', 'RToverhanglength', 
                    'RTlength', 'PBSlength', 'RTmt', 'RToverhangmt','PBSmt','protospacermt',
                    'extensionmt','original_base_mt','edited_base_mt','original_base_mt_nan',
                    'edited_base_mt_nan']:
            if col not in pegdataframe.columns:
                pegdataframe[col] = 0
                
        all_avg_preds = deeppridict(pegdataframe, pridict_models)
        tmp = [all_avg_preds[model_id] for model_id in all_avg_preds]
        tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
        agg_df = compute_average_predictions(tmp_df, grp_cols=['seq_id', 'dataset_name'])
        
        hek_cond = agg_df['dataset_name'] == 'HEK'
        agg_hek = agg_df[hek_cond].sort_values('seq_id')
        
        for _, r in agg_hek.iterrows():
            v_idx = int(r['seq_id'])
            ref_id = pegdataframe.iloc[v_idx]['sequence_name']
            df.loc[df['ID'] == ref_id, 'pred_pridict2'] = r['pred_averageedited'] * 100
            
    df.to_parquet(input_file, index=False)
    print("Done generating PRIDICT2 predictions.")

if __name__ == '__main__':
    main()
