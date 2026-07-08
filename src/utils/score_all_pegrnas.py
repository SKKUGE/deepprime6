import os
import sys
import time
import glob
import pandas as pd
import numpy as np
import torch
import multiprocessing
from scipy.stats import spearmanr

# Set project root path
sys.path.append("/home/work/workdir/deepprime6-genomebiol-revision")
from Bio.Seq import Seq, reverse_complement, transcribe
from Bio.SeqUtils import MeltingTemp as mt
from RNA import fold_compound

# Set up PRIDICT2 path
pridict2_dir = os.path.abspath("PRIDICT2")
sys.path.append(pridict2_dir)
from src.models.pe6_module_dp_only import PE6DeepPrimeModule
from src.utils.predict_pridict2 import load_pridict_model, deeppridict, compute_average_predictions, extract_single_pegRNA_features

# Feature names needed for DeepPrime
FEATURE_COLS = [
    "PBS_len", "RTT_len", "RT-PBS_len", "Edit_pos", "Edit_len", "RHA_len",
    "type_sub", "type_ins", "type_del", "Tm1_PBS", "Tm2_RTT_cTarget_sameLength",
    "Tm3_RTT_cTarget_replaced", "Tm4_cDNA_PAM-oppositeTarget", "Tm5_RTT_cDNA",
    "deltaTm_Tm4-Tm2", "GC_count_PBS", "GC_count_RTT", "GC_count_RT-PBS",
    "GC_contents_PBS", "GC_contents_RTT", "GC_contents_RT-PBS",
    "MFE_RT-PBS-polyT", "MFE_Spacer", "DeepSpCas9_score"
]

def process_row_features(row_tuple):
    idx, row_dict = row_tuple
    pbs = row_dict['PBS_pegRNA_DNA'].upper()
    rtt = row_dict['RTT_template_DNA'].upper()
    pbs_len = row_dict['PBSlen']
    rt_len = row_dict['RTlen']
    edited74 = row_dict['Edited74_On'].upper()
    
    # 1. GC content
    nGCcnt1 = pbs.count("G") + pbs.count("C")
    nGCcnt2 = rtt.count("G") + rtt.count("C")
    nGCcnt3 = (pbs + rtt).count("G") + (pbs + rtt).count("C")
    fGCcont1 = 100 * (nGCcnt1 / len(pbs)) if len(pbs) > 0 else 0
    fGCcont2 = 100 * (nGCcnt2 / len(rtt)) if len(rtt) > 0 else 0
    fGCcont3 = 100 * (nGCcnt3 / len(pbs + rtt)) if len(pbs + rtt) > 0 else 0
    
    # 2. Tm NN
    try:
        tm1 = mt.Tm_NN(transcribe(pbs))
    except Exception:
        tm1 = 0.0
        
    rc_pbs = reverse_complement(pbs)
    rc_rtt = reverse_complement(rtt)
    
    # Complement base at index 21 of Edited74_On for wild-type
    wt_first_base = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}[rc_rtt[0]]
    wt_rc_rtt = wt_first_base + rc_rtt[1:]
    
    try:
        tm2 = mt.Tm_NN(wt_rc_rtt)
        tm3 = tm2
        tm4 = mt.Tm_NN(reverse_complement(rtt), c_seq=reverse_complement(wt_rc_rtt))
        tm5 = mt.Tm_NN(transcribe(rtt))
    except Exception:
        tm2, tm3, tm4, tm5 = 0.0, 0.0, 0.0, 0.0
        
    # 3. MFEs
    sInputSeq = (reverse_complement(pbs + rtt) + "TTTTTT").replace("T", "U")
    try:
        _, fMFE3 = fold_compound(sInputSeq).mfe()
    except Exception:
        fMFE3 = 0.0
        
    # Reconstruct spacer (Guide)
    S = edited74.find(rc_pbs)
    nick_site = S + pbs_len
    spacer = edited74[nick_site - 17 : nick_site + 3]
    spacer_no_x = spacer.replace("X", "A")
    sInputSeq_spacer = spacer_no_x.replace("T", "U")
    try:
        _, fMFE4 = fold_compound(sInputSeq_spacer).mfe()
    except Exception:
        fMFE4 = 0.0
        
    # 4. SpCas9 30nt target
    target_list = list(edited74)
    for i in range(len(target_list)):
        if target_list[i] == 'X':
            target_list[i] = 'A'
    target_list[21] = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}[target_list[21]]
    target_seq = "".join(target_list)
    
    target_seq_padded = "A" * 10 + target_seq + "A" * 10
    nicking_index_padded = (nick_site - 1) + 10
    seq30_start = nicking_index_padded - 21
    seq30_end = nicking_index_padded + 9
    seq30 = target_seq_padded[seq30_start:seq30_end]
    
    # 74nt Target (padded) and Masked_EditSeq (keep x's)
    masked_edit_seq = edited74
    target_74_padded = "A" * 10 + target_seq + "A" * 10
    
    return {
        'idx': idx,
        'PBS_len': pbs_len,
        'RTT_len': rt_len,
        'RT-PBS_len': pbs_len + rt_len,
        'Edit_pos': 1,
        'Edit_len': 1,
        'RHA_len': rt_len,
        'type_sub': 1,
        'type_ins': 0,
        'type_del': 0,
        'Tm1_PBS': round(tm1, 1),
        'Tm2_RTT_cTarget_sameLength': round(tm2, 1),
        'Tm3_RTT_cTarget_replaced': round(tm3, 1),
        'Tm4_cDNA_PAM-oppositeTarget': round(tm4, 1),
        'Tm5_RTT_cDNA': round(tm5, 1),
        'deltaTm_Tm4-Tm2': round(tm4 - tm2, 1),
        'GC_count_PBS': nGCcnt1,
        'GC_count_RTT': nGCcnt2,
        'GC_count_RT-PBS': nGCcnt3,
        'GC_contents_PBS': fGCcont1,
        'GC_contents_RTT': fGCcont2,
        'GC_contents_RT-PBS': fGCcont3,
        'MFE_RT-PBS-polyT': round(fMFE3, 1),
        'MFE_Spacer': round(fMFE4, 1),
        'spacer': spacer_no_x,
        'seq30': seq30,
        'Masked_EditSeq_74': masked_edit_seq,
        'Target_74': target_74_padded
    }

def one_hot_encode_sequence(sequence):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "X": 4, "N": 4}
    map_seq = [mapping[i] for i in sequence.upper()]
    arr_seq = np.eye(5)[map_seq]
    return np.delete(arr_seq, -1, axis=1)

def run_deep_spcas9_inference(seq30_list, device):
    from genet.predict import SpCas9
    print("Running DeepSpCas9 inference on GPU...")
    # GenET SpCas9 handles batches and GPU configuration internally
    sp = SpCas9()
    scores = sp.predict(seq30_list)["SpCas9"].tolist()
    return scores

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    # 1. Load data
    print("Loading converted ClinVar 710k pegRNA dataset...")
    df = pd.read_csv("data/MFE_randompeg_RHA30_0.71M_result.csv")
    print(f"Loaded {len(df)} pegRNA configurations.")
    
    # 2. Extract features in parallel using multiprocessing
    print("Running parallel feature extraction...")
    t0 = time.time()
    row_tuples = list(df.iterrows())
    row_dicts = [(idx, dict(row)) for idx, row in row_tuples]
    
    # Use 3 CPU cores as checked
    pool = multiprocessing.Pool(processes=3)
    features_list = pool.map(process_row_features, row_dicts)
    pool.close()
    pool.join()
    print(f"Feature extraction completed in {time.time() - t0:.2f}s.")
    
    # Convert features to DataFrame
    feat_df = pd.DataFrame(features_list).sort_values('idx').reset_index(drop=True)
    
    # Run SpCas9 prediction on all spacers
    print("Running DeepSpCas9 predictions...")
    t_sp = time.time()
    seq30_list = feat_df['seq30'].tolist()
    # Batch predict SpCas9 scores
    sp_scores = run_deep_spcas9_inference(seq30_list, device)
    feat_df['DeepSpCas9_score'] = sp_scores
    print(f"DeepSpCas9 completed in {time.time() - t_sp:.2f}s.")
    
    # 3. Model predictions: DeepPrime base and DeepPrime6 (4 checkpoints)
    print("Loading DeepPrime models...")
    ckpt_dir = "src/models/weights/DeepPrime6-weights/"
    checkpoints = {
        "pred_dp6_pe6a": "pe6a_mainft.ckpt",
        "pred_dp6_pe6b": "pe6b_mainft.ckpt",
        "pred_dp6_pe6c": "pe6c_mainft.ckpt",
        "pred_dp6_drnaseh": "pemaxdrnaseh_mainft.ckpt"
    }
    
    # Instantiate models
    dp6_models = {}
    for name, ckpt_name in checkpoints.items():
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        model = PE6DeepPrimeModule.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval()
        dp6_models[name] = model
        print(f"Loaded DeepPrime6 {name}")
        
    # Also load the base model (reload base weights into one of the checkpoints' structures)
    base_model = PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6a_mainft.ckpt"), map_location=device)
    base_model.eval()
    m_files = sorted(glob.glob("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/*.pt"))
    for i, m_file in enumerate(m_files):
        genet_sd = torch.load(m_file, map_location=device, weights_only=True)
        remapped_sd = base_model._remap_genet_keys(genet_sd, base_model.feature_extractor.models[i])
        base_model.feature_extractor.models[i].load_state_dict(remapped_sd, strict=False)
    print("Loaded DeepPrime Base model.")
    
    # Prepare batch execution inputs for DeepPrime
    print("Running DeepPrime and DeepPrime6 predictions...")
    t_dp = time.time()
    
    # Compute mean/std on the full dataset for standardization
    biofeatures_raw = feat_df[FEATURE_COLS].copy()
    
    # Rename columns to match DeepPrime original feature names
    feature_rename_map = {
        "PBS_len": "PBSlen", "RTT_len": "RTlen", "RT-PBS_len": "RT-PBSlen",
        "Tm1_PBS": "Tm1", "Tm2_RTT_cTarget_sameLength": "Tm2", "Tm3_RTT_cTarget_replaced": "Tm2new",
        "Tm5_RTT_cDNA": "Tm3", "Tm4_cDNA_PAM-oppositeTarget": "Tm4", "deltaTm_Tm4-Tm2": "TmD",
        "GC_count_PBS": "nGCcnt1", "GC_count_RTT": "nGCcnt2", "GC_count_RT-PBS": "nGCcnt3",
        "GC_contents_PBS": "fGCcont1", "GC_contents_RTT": "fGCcont2", "GC_contents_RT-PBS": "fGCcont3",
        "MFE_RT-PBS-polyT": "MFE3", "MFE_Spacer": "MFE4",
    }
    biofeatures_renamed = biofeatures_raw.rename(columns=feature_rename_map)
    # Order to match standard feature indexing
    standard_features = [
        "PBSlen", "RTlen", "RT-PBSlen", "Edit_pos", "Edit_len", "RHA_len",
        "type_sub", "type_ins", "type_del", "Tm1", "Tm2", "Tm2new", "Tm4", "Tm3", "TmD",
        "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
        "MFE3", "MFE4", "DeepSpCas9_score"
    ]
    biofeatures_ordered = biofeatures_renamed[standard_features]
    
    # Standardization
    norm_mean = biofeatures_ordered.mean()
    norm_std = biofeatures_ordered.std()
    norm_biofeatures = (biofeatures_ordered - norm_mean) / norm_std
    norm_biofeatures = norm_biofeatures.fillna(0).values
    
    # Sequence encoding
    print("One-hot encoding sequences...")
    # CNN expects 74nt inputs.
    # Target_74 has length 94 (with +10 padding). Slicing indices 10 to 84 retrieves the 74nt sequence.
    target_enc_list = [one_hot_encode_sequence(t[10:84]) for t in feat_df['Target_74']]
    masked_enc_list = [one_hot_encode_sequence(m) for m in feat_df['Masked_EditSeq_74']]
    
    genetic_features = np.stack([
        np.stack([t, m], axis=0) for t, m in zip(target_enc_list, masked_enc_list)
    ], axis=0)
    genetic_features = 2 * genetic_features - 1
    
    # Batched execution on GPU
    batch_size = 4096
    num_samples = len(feat_df)
    preds_dp_base = np.zeros(num_samples)
    preds_dp6_pe6a = np.zeros(num_samples)
    preds_dp6_pe6b = np.zeros(num_samples)
    preds_dp6_pe6c = np.zeros(num_samples)
    preds_dp6_drnaseh = np.zeros(num_samples)
    
    print("Running forward passes in chunks...")
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            g_batch = torch.from_numpy(genetic_features[start_idx:end_idx].copy()).float().to(device)
            g_batch = g_batch.permute(0, 3, 1, 2)  # shape (batch, 4, 2, 74)
            b_batch = torch.from_numpy(norm_biofeatures[start_idx:end_idx].copy()).float().to(device)
            
            # Predict Base
            preds_dp_base[start_idx:end_idx] = base_model((g_batch, b_batch)).cpu().numpy().flatten()
            # Predict DP6 variants
            preds_dp6_pe6a[start_idx:end_idx] = dp6_models["pred_dp6_pe6a"]((g_batch, b_batch)).cpu().numpy().flatten()
            preds_dp6_pe6b[start_idx:end_idx] = dp6_models["pred_dp6_pe6b"]((g_batch, b_batch)).cpu().numpy().flatten()
            preds_dp6_pe6c[start_idx:end_idx] = dp6_models["pred_dp6_pe6c"]((g_batch, b_batch)).cpu().numpy().flatten()
            preds_dp6_drnaseh[start_idx:end_idx] = dp6_models["pred_dp6_drnaseh"]((g_batch, b_batch)).cpu().numpy().flatten()
            
            if (start_idx + batch_size) % 102400 == 0 or end_idx == num_samples:
                print(f"Processed {end_idx}/{num_samples} rows for DeepPrime...")
                
    feat_df['pred_dp_base'] = preds_dp_base
    feat_df['pred_dp6_pe6a'] = preds_dp6_pe6a
    feat_df['pred_dp6_pe6b'] = preds_dp6_pe6b
    feat_df['pred_dp6_pe6c'] = preds_dp6_pe6c
    feat_df['pred_dp6_drnaseh'] = preds_dp6_drnaseh
    print(f"DeepPrime predictions completed in {time.time() - t_dp:.2f}s.")
    
    # 4. PRIDICT2.0 Predictions
    print("Running PRIDICT2.0 predictions...")
    t_pr = time.time()
    pridict_models = load_pridict_model(run_ids=[0])
    
    # Process PRIDICT2.0 in chunks of 50k to prevent RAM blowup
    pr_chunk_size = 50000
    preds_pridict2_hek = np.zeros(num_samples)
    preds_pridict2_k562 = np.zeros(num_samples)
    
    # Setup dataframe needed by PRIDICT2 parser
    pr_input_df = pd.DataFrame()
    pr_input_df['REF_ID'] = df['ID']
    pr_input_df['WildTypeSequence'] = feat_df['Target_74'].apply(lambda x: x[10:84])
    pr_input_df['PrimeEditedSequence'] = feat_df['Masked_EditSeq_74']
    pr_input_df['Edit_type'] = 'Sub'
    pr_input_df['Guide'] = feat_df['spacer']
    pr_input_df['PBS'] = df['PBS_pegRNA_DNA']
    pr_input_df['RTT'] = df['RTT_template_DNA']
    
    print("Running PRIDICT2.0 feature parsing and inference...")
    for chunk_start in range(0, num_samples, pr_chunk_size):
        chunk_end = min(chunk_start + pr_chunk_size, num_samples)
        chunk_df = pr_input_df.iloc[chunk_start:chunk_end].reset_index(drop=True)
        
        # Parse features
        features_list_pr = []
        for idx, row in chunk_df.iterrows():
            try:
                feat = extract_single_pegRNA_features(row)
                if feat is not None:
                    features_list_pr.append(feat)
                else:
                    # Fallback default feature
                    features_list_pr.append({'sequence_name': row['REF_ID']})
            except Exception:
                features_list_pr.append({'sequence_name': row['REF_ID']})
                
        pegdataframe = pd.DataFrame(features_list_pr)
        # Fill missing keys if any failed
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
                
        # Call deeppridict
        all_avg_preds = deeppridict(pegdataframe, pridict_models)
        tmp = [all_avg_preds[model_id] for model_id in all_avg_preds]
        tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
        agg_df = compute_average_predictions(tmp_df, grp_cols=['seq_id', 'dataset_name'])
        
        # Retrieve predictions
        hek_cond = agg_df['dataset_name'] == 'HEK'
        k562_cond = agg_df['dataset_name'] == 'K562'
        preds_pridict2_hek[chunk_start:chunk_end] = agg_df.loc[hek_cond, 'pred_averageedited'].values * 100
        preds_pridict2_k562[chunk_start:chunk_end] = agg_df.loc[k562_cond, 'pred_averageedited'].values * 100
        
        print(f"Processed {chunk_end}/{num_samples} rows for PRIDICT2.0...")
        
    feat_df['pred_pridict2_hek'] = preds_pridict2_hek
    feat_df['pred_pridict2_k562'] = preds_pridict2_k562
    print(f"PRIDICT2.0 predictions completed in {time.time() - t_pr:.2f}s.")
    
    # 5. Save all predictions
    print("Saving full 3-model predictions file...")
    output_df = df.copy()
    output_df['pred_dp_base'] = feat_df['pred_dp_base']
    output_df['pred_dp6_pe6a'] = feat_df['pred_dp6_pe6a']
    output_df['pred_dp6_pe6b'] = feat_df['pred_dp6_pe6b']
    output_df['pred_dp6_pe6c'] = feat_df['pred_dp6_pe6c']
    output_df['pred_dp6_drnaseh'] = feat_df['pred_dp6_drnaseh']
    output_df['pred_pridict2_hek'] = feat_df['pred_pridict2_hek']
    output_df['pred_pridict2_k562'] = feat_df['pred_pridict2_k562']
    
    output_df.to_csv("data/MFE_randompeg_RHA30_0.71M_scored.csv", index=False)
    print("Full predictions saved to data/MFE_randompeg_RHA30_0.71M_scored.csv")
    
    # 6. Candidate Selection (REQ-2, REQ-3, REQ-4)
    print("Selecting candidates...")
    # Normalize scores of representative models:
    # 1. DeepPrime (Base)
    # 2. DeepPrime6 (We use PE6a variant as representative DeepPrime6, or average? Let's use pe6a)
    # 3. PRIDICT2.0 (HEK)
    output_df['pct_dp_base'] = output_df['pred_dp_base'].rank(pct=True)
    output_df['pct_dp6'] = output_df['pred_dp6_pe6a'].rank(pct=True)
    output_df['pct_pridict2'] = output_df['pred_pridict2_hek'].rank(pct=True)
    
    # Disagreement Score (standard deviation across normalized percentiles)
    output_df['disagreement'] = output_df[['pct_dp_base', 'pct_dp6', 'pct_pridict2']].std(axis=1)
    # Average Percentile Score
    output_df['avg_pct'] = output_df[['pct_dp_base', 'pct_dp6', 'pct_pridict2']].mean(axis=1)
    
    # Select:
    # 1. Highly Efficient (15 candidates):
    #    - 10 Consensus: Top avg_pct, lowest disagreement
    #    - 5 Disagreement: Highest disagreement where at least one model is top tercile (>0.67) and another is not
    # 2. Modest (20 candidates):
    #    - 14 Consensus: avg_pct around 0.5 (between 0.45 and 0.55), lowest disagreement
    #    - 6 Disagreement: Highest disagreement where avg_pct is around 0.5
    # 3. Inefficient (15 candidates):
    #    - 10 Consensus: Bottom avg_pct, lowest disagreement
    #    - 5 Disagreement: Highest disagreement where at least one model is bottom tercile (<0.33) and another is not
    
    # Define tercile condition masks
    is_high = output_df[['pct_dp_base', 'pct_dp6', 'pct_pridict2']].max(axis=1) > 0.67
    is_low = output_df[['pct_dp_base', 'pct_dp6', 'pct_pridict2']].min(axis=1) < 0.33
    is_modest = (output_df['avg_pct'] > 0.4) & (output_df['avg_pct'] < 0.6)
    
    # High Bin Candidates
    high_candidates_pool = output_df[is_high].copy()
    high_consensus = high_candidates_pool.sort_values(by=['avg_pct', 'disagreement'], ascending=[False, True]).head(10)
    high_disagreement = high_candidates_pool.drop(high_consensus.index).sort_values(by='disagreement', ascending=False).head(5)
    high_selected = pd.concat([high_consensus, high_disagreement])
    high_selected['bin'] = 'Highly Efficient'
    
    # Modest Bin Candidates
    modest_candidates_pool = output_df[is_modest].copy()
    # Drop already selected if any overlap
    modest_candidates_pool = modest_candidates_pool.drop(index=high_selected.index, errors='ignore')
    modest_consensus = modest_candidates_pool.sort_values(by='disagreement', ascending=True).head(14)
    modest_disagreement = modest_candidates_pool.drop(modest_consensus.index).sort_values(by='disagreement', ascending=False).head(6)
    modest_selected = pd.concat([modest_consensus, modest_disagreement])
    modest_selected['bin'] = 'Modest'
    
    # Inefficient Bin Candidates
    low_candidates_pool = output_df[is_low].copy()
    # Drop already selected
    low_candidates_pool = low_candidates_pool.drop(index=pd.concat([high_selected, modest_selected]).index, errors='ignore')
    low_consensus = low_candidates_pool.sort_values(by=['avg_pct', 'disagreement'], ascending=[True, True]).head(10)
    low_disagreement = low_candidates_pool.drop(low_consensus.index).sort_values(by='disagreement', ascending=False).head(5)
    low_selected = pd.concat([low_consensus, low_disagreement])
    low_selected['bin'] = 'Inefficient'
    
    # Combine Selected Candidates
    selected_50 = pd.concat([high_selected, modest_selected, low_selected])
    
    # Add Spacer and PAM to selected candidates table
    selected_feat_df = feat_df.loc[selected_50.index]
    selected_50['Spacer_Sequence'] = selected_feat_df['spacer']
    # PAM sequence: indices nick_site + 3 to nick_site + 6 of Edited74_On
    pams = []
    for idx, row in selected_50.iterrows():
        n_site = selected_feat_df.loc[idx, 'nick_site']
        edited74 = row['Edited74_On'].upper()
        pams.append(edited74[n_site + 3 : n_site + 6])
    selected_50['PAM'] = pams
    
    # Save the selected 50 candidates to CSV
    selected_50_cols = [
        'ID', 'bin', 'Edited74_On', 'Spacer_Sequence', 'PAM',
        'PBS_pegRNA_DNA', 'PBSlen', 'RTT_template_DNA', 'RTlen',
        'pred_dp_base', 'pred_dp6_pe6a', 'pred_pridict2_hek',
        'pct_dp_base', 'pct_dp6', 'pct_pridict2', 'disagreement'
    ]
    selected_50_output = selected_50[selected_50_cols]
    selected_50_output.to_csv("data/selected_50_validation_candidates.csv", index=False)
    print("Selected 50 candidates saved to data/selected_50_validation_candidates.csv")
    
    # Calculate and verify correlation matrices
    print("Pairwise correlation in full library:")
    full_corr = output_df[['pred_dp_base', 'pred_dp6_pe6a', 'pred_pridict2_hek']].corr(method='spearman')
    print(full_corr.to_string())
    
    print("Pairwise correlation in selected 50 candidates:")
    selected_corr = selected_50[['pred_dp_base', 'pred_dp6_pe6a', 'pred_pridict2_hek']].corr(method='spearman')
    print(selected_corr.to_string())

if __name__ == "__main__":
    main()
