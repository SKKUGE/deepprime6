# randompeg scoring script

import os
import sys
import time
import glob
import logging
import pandas as pd
import numpy as np
import torch
import multiprocessing
from scipy.stats import spearmanr

# Set project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Bio.Seq import Seq, reverse_complement, transcribe
from Bio.SeqUtils import MeltingTemp as mt
from RNA import fold_compound

# Set up PRIDICT2 path
pridict2_dir = os.path.join(PROJECT_ROOT, "PRIDICT2")
if pridict2_dir not in sys.path:
    sys.path.insert(0, pridict2_dir)
from src.models.pe6_module_dp_only import PE6DeepPrimeModule
from src.models.components.dprime import GeneInteractionModelVanilla
from src.utils.predict_pridict2 import load_pridict_model, deeppridict, compute_average_predictions, extract_single_pegRNA_features

logger = logging.getLogger("score_all_pegrnas")
FALLBACK_LOG_PATH = "logs/score_all_pegrnas_fallbacks.log"  # To log any unintentional cases

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
        tm1 = mt.Tm_NN(transcribe(pbs), nn_table=mt.R_DNA_NN1)
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
        tm5 = mt.Tm_NN(transcribe(rtt), nn_table=mt.R_DNA_NN1)
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
        'nick_site': nick_site,
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

# Pipeline feature name -> DeepPrime-original feature name (also accepted as input column names)
FEATURE_RENAME_MAP = {
    "PBS_len": "PBSlen", "RTT_len": "RTlen", "RT-PBS_len": "RT-PBSlen",
    "Tm1_PBS": "Tm1", "Tm2_RTT_cTarget_sameLength": "Tm2", "Tm3_RTT_cTarget_replaced": "Tm2new",
    "Tm4_cDNA_PAM-oppositeTarget": "Tm3", "Tm5_RTT_cDNA": "Tm4", "deltaTm_Tm4-Tm2": "TmD",
    "GC_count_PBS": "nGCcnt1", "GC_count_RTT": "nGCcnt2", "GC_count_RT-PBS": "nGCcnt3",
    "GC_contents_PBS": "fGCcont1", "GC_contents_RTT": "fGCcont2", "GC_contents_RT-PBS": "fGCcont3",
    "MFE_RT-PBS-polyT": "MFE3", "MFE_Spacer": "MFE4",
}
TYPE_COLS = ["type_sub", "type_ins", "type_del"]
# Features that are *assumed* (not derivable from the sequence) when no input column exists
ASSUMED_FEATURES = ["Edit_pos", "Edit_len", "RHA_len"]
EDIT_TYPE_ALIASES = {
    "sub": "Sub", "substitution": "Sub", "snv": "Sub", "replacement": "Sub",
    "ins": "Ins", "insertion": "Ins",
    "del": "Del", "deletion": "Del",
}

def setup_logging():
    """Log to console and to FALLBACK_LOG_PATH so every fallback is recorded."""
    os.makedirs(os.path.dirname(FALLBACK_LOG_PATH), exist_ok=True)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        for h in (logging.StreamHandler(), logging.FileHandler(FALLBACK_LOG_PATH, mode="a")):
            h.setFormatter(fmt)
            logger.addHandler(h)

# NOTE: Legacy summary tables with inconsistent Edit_type/Edit_len annotations (e.g. 'c.1646del'
# labelled 'Sub' with Edit_len=0) are intentionally excluded; clean Edit_type columns before use.
def resolve_edit_type(df):
    """Per-row edit type ('Sub'/'Ins'/'Del') from input columns; falls back to 'Sub' with error logs."""
    n = len(df)
    if all(c in df.columns for c in TYPE_COLS):
        tmat = df[TYPE_COLS].fillna(0).to_numpy()
        bad = tmat.sum(axis=1) != 1
        if bad.any():
            logger.error("%d/%d rows have invalid one-hot %s; falling back to 'Sub' for those rows", int(bad.sum()), n, TYPE_COLS)
        idx = np.where(bad, 0, tmat.argmax(axis=1))
        return pd.Series(np.array(["Sub", "Ins", "Del"])[idx], index=df.index)
    col = next((c for c in df.columns if c.lower() in ("edit_type", "edittype")), None)
    if col is None:
        logger.error("No edit-type column (Edit_type / %s) in input; falling back to 'Sub' for all %d rows", TYPE_COLS, n)
        return pd.Series("Sub", index=df.index)
    norm = df[col].astype(str).str.strip().str.lower().map(EDIT_TYPE_ALIASES)
    bad = norm.isna()
    if bad.any():
        logger.error("%d/%d rows have unrecognized '%s' values (e.g. %s); falling back to 'Sub' for those rows",
                     int(bad.sum()), n, col, df.loc[bad, col].unique()[:5].tolist())
        norm = norm.fillna("Sub")
    return norm

def resolve_input_features(feat_df, df):
    """Override sequence-derived/assumed features with input-data columns when available.

    Every fallback to an assumed value is reported via logger.error. Returns the per-row edit type.
    """
    assert (feat_df["idx"].to_numpy() == df.index.to_numpy()).all(), "feat_df rows are not aligned with input df"
    edit_types = resolve_edit_type(df).reset_index(drop=True)
    feat_df["type_sub"] = (edit_types == "Sub").astype(int).to_numpy()
    feat_df["type_ins"] = (edit_types == "Ins").astype(int).to_numpy()
    feat_df["type_del"] = (edit_types == "Del").astype(int).to_numpy()

    for f in FEATURE_COLS:
        if f in TYPE_COLS:
            continue
        cands = [f] + ([FEATURE_RENAME_MAP[f]] if f in FEATURE_RENAME_MAP else [])
        src = next((c for c in cands if c in df.columns), None)
        if f not in feat_df.columns:
            feat_df[f] = np.nan
        if src is None:
            if f in ASSUMED_FEATURES:
                logger.error("No input column for '%s' (tried %s); falling back to the assumed value for all %d rows", f, cands, len(df))
            elif f == "DeepSpCas9_score":
                logger.info("No input column for 'DeepSpCas9_score'; will compute with DeepSpCas9")
            else:
                logger.info("No input column for '%s'; using value computed from sequence", f)
            continue
        vals = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=float)
        ok = ~np.isnan(vals)
        feat_df.loc[ok, f] = vals[ok]
        if not ok.all():
            logger.error("Input column '%s' missing/non-numeric for %d/%d rows; using computed/assumed value for those rows",
                         src, int((~ok).sum()), len(df))
    return edit_types

def load_deepprime_base(weights_dir, device):
    """Load the original DeepPrime ensemble (20 models) with the original GENET architecture.

    strict=True guarantees that every parameter/buffer comes from the GENET files only; no PE6
    checkpoint (weights or BatchNorm statistics) is involved.
    """
    files = sorted(glob.glob(os.path.join(weights_dir, "final_model_*.pt")))
    if len(files) != 20:
        raise FileNotFoundError(f"Expected 20 DeepPrime base models in {weights_dir}, found {len(files)}")
    models = []
    for f in files:
        m = GeneInteractionModelVanilla()
        m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True)
        models.append(m.to(device).eval())
    return models

def predict_deepprime_base(models, g, b):
    return torch.stack([m(g, b) for m in models], dim=0).mean(dim=0)

def main():
    setup_logging()
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
    
    # Prefer input-data columns (edit type, edit pos/len, Tm/GC/MFE, DeepSpCas9, ...); log every fallback
    edit_types = resolve_input_features(feat_df, df)
    non_sub = edit_types != "Sub"
    if non_sub.any():
        # Wild-type context (Target_74) is reconstructed assuming a 1-bp substitution
        raise ValueError(
            f"{int(non_sub.sum())} rows have non-Sub edit type ({edit_types[non_sub].unique().tolist()}); "
            "WT/edited context reconstruction in this script supports substitutions only."
        )
    
    # Run SpCas9 prediction only for rows without an input DeepSpCas9_score
    sp_missing = feat_df['DeepSpCas9_score'].isna().to_numpy()
    if sp_missing.any():
        print(f"Running DeepSpCas9 predictions for {int(sp_missing.sum())} rows...")
        t_sp = time.time()
        sp_scores = run_deep_spcas9_inference(feat_df.loc[sp_missing, 'seq30'].tolist(), device)
        feat_df.loc[sp_missing, 'DeepSpCas9_score'] = sp_scores
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
        
    # DeepPrime base: original GENET architecture + original weights only (no PE6 checkpoint involved)
    base_models = load_deepprime_base("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428", device)
    print(f"Loaded DeepPrime Base model ({len(base_models)}-model ensemble, strict GENET weights).")
    
    # Prepare batch execution inputs for DeepPrime
    print("Running DeepPrime and DeepPrime6 predictions...")
    t_dp = time.time()
    
    # Compute mean/std on the full dataset for standardization
    biofeatures_raw = feat_df[FEATURE_COLS].copy()
    
    # Rename columns to match DeepPrime original feature names
    biofeatures_renamed = biofeatures_raw.rename(columns=FEATURE_RENAME_MAP)
    # Order to match standard feature indexing
    standard_features = [
        "PBSlen", "RTlen", "RT-PBSlen", "Edit_pos", "Edit_len", "RHA_len",
        "type_sub", "type_ins", "type_del", "Tm1", "Tm2", "Tm2new", "Tm3", "Tm4", "TmD",
        "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
        "MFE3", "MFE4", "DeepSpCas9_score"
    ]
    biofeatures_ordered = biofeatures_renamed[standard_features]
    
    # Canonical statistics for dual normalization
    # 1. Base statistics for DeepPrime-base
    base_norm_dir = "src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428"
    norm_mean_base = pd.read_csv(os.path.join(base_norm_dir, "mean.csv"), header=None, index_col=0).squeeze()
    norm_std_base = pd.read_csv(os.path.join(base_norm_dir, "std.csv"), header=None, index_col=0).squeeze()

    # 2. Canonical PE6 statistics for DeepPrime6 checkpoints
    pe6_norm_dir = "src/models/weights/DeepPrime6-weights"
    norm_mean_pe6 = pd.read_csv(os.path.join(pe6_norm_dir, "pe6_mean.csv"), header=None, index_col=0).squeeze()
    norm_std_pe6 = pd.read_csv(os.path.join(pe6_norm_dir, "pe6_std.csv"), header=None, index_col=0).squeeze()

    norm_biofeatures_base = ((biofeatures_ordered - norm_mean_base) / norm_std_base).fillna(0).values
    norm_biofeatures_pe6 = ((biofeatures_ordered - norm_mean_pe6) / norm_std_pe6).fillna(0).values
    
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
            b_batch_base = torch.from_numpy(norm_biofeatures_base[start_idx:end_idx].copy()).float().to(device)
            b_batch_pe6 = torch.from_numpy(norm_biofeatures_pe6[start_idx:end_idx].copy()).float().to(device)
            
            # Predict Base with canonical Base normalization
            preds_dp_base[start_idx:end_idx] = predict_deepprime_base(base_models, g_batch, b_batch_base).cpu().numpy().flatten()
            # Predict DP6 variants with canonical PE6 normalization
            preds_dp6_pe6a[start_idx:end_idx] = dp6_models["pred_dp6_pe6a"]((g_batch, b_batch_pe6)).cpu().numpy().flatten()
            preds_dp6_pe6b[start_idx:end_idx] = dp6_models["pred_dp6_pe6b"]((g_batch, b_batch_pe6)).cpu().numpy().flatten()
            preds_dp6_pe6c[start_idx:end_idx] = dp6_models["pred_dp6_pe6c"]((g_batch, b_batch_pe6)).cpu().numpy().flatten()
            preds_dp6_drnaseh[start_idx:end_idx] = dp6_models["pred_dp6_drnaseh"]((g_batch, b_batch_pe6)).cpu().numpy().flatten()
            
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
    pr_input_df['Edit_type'] = edit_types.to_numpy()  # resolved from input columns (fallback logged)
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
