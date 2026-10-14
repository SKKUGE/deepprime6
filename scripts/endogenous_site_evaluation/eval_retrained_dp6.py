import os, sys, torch, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

def find_project_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".project-root")) or (
            os.path.exists(os.path.join(curr, "PRIDICT2")) and os.path.exists(os.path.join(curr, "src"))
        ):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.getcwd())

PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.inference import (
    char_to_onehot_tensor,
    prepare_biofeatures,
    load_deepprime6_models,
    predict_deepprime6,
)

excel_df = pd.read_excel('data/endogenous_ground_truth/PE_efficiency_cleanonly_NCcorrected_n32.xlsx', sheet_name='Final_n32_NCcorrected')
pq_df = pd.read_parquet('data/full_output/final_candidates_validated.parquet')
pq_df['target_num'] = range(1, len(pq_df) + 1)
merged_cand = pd.merge(excel_df, pq_df[['target_num', 'design_id', 'Disagreement_Subcategory']], on='target_num')

bio_df = pd.read_parquet('data/full_output/pegrna_designs_biofeatures.parquet')
bio_matched = pd.merge(merged_cand, bio_df, on='design_id')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Sequence and biofeature preparation via canonical engine
g_tensor = char_to_onehot_tensor(
    bio_matched['Target'].tolist(),
    bio_matched['Masked_EditSeq'].tolist(),
    device
)
_, b_tensor = prepare_biofeatures(bio_matched, 'pe6', device)

dp6_models = load_deepprime6_models(device=device, is_scratch=False)
dp6_raw_preds = predict_deepprime6(dp6_models, g_tensor, b_tensor)

new_preds = {
    '6a': dp6_raw_preds['pe6a'],
    '6b': dp6_raw_preds['pe6b'],
    '6c': dp6_raw_preds['pe6c'],
    'dRN': dp6_raw_preds['drnaseh']
}

print('=== Retrained DeepPrime6 (Canonical codebase engine + model_versions.yaml) ===')
for sys_name in ['6a', '6b', '6c', 'dRN']:
    out = new_preds[sys_name]
    r_new, p_new = spearmanr(out, bio_matched[f'{sys_name}_NCcorr_%'])
    r_old, _ = spearmanr(bio_matched['pred_dp6'], bio_matched[f'{sys_name}_NCcorr_%'])
    r_pri, _ = spearmanr(bio_matched['pred_pridict2'], bio_matched[f'{sys_name}_NCcorr_%'])
    print(f'System {sys_name:4s}: Retrained DP6 R={r_new:.4f} (p={p_new:.4f}) | July pred_dp6 R={r_old:.4f} | PRIDICT2 R={r_pri:.4f}')

avg_new = (new_preds['6a'] + new_preds['6b'] + new_preds['6c'] + new_preds['dRN']) / 4.0
r_avg_new, p_avg_new = spearmanr(avg_new, bio_matched['overall_avg_NCcorr_%'])
r_avg_old, _ = spearmanr(bio_matched['pred_dp6'], bio_matched['overall_avg_NCcorr_%'])
r_avg_pri, _ = spearmanr(bio_matched['pred_pridict2'], bio_matched['overall_avg_NCcorr_%'])
print(f'Overall Avg: Retrained DP6 R={r_avg_new:.4f} (p={p_avg_new:.4f}) | July pred_dp6 R={r_avg_old:.4f} | PRIDICT2 R={r_avg_pri:.4f}')

# Ensemble comparison
r_ens_new, p_ens_new = spearmanr(bio_matched['pred_dp_base'].rank() + pd.Series(avg_new).rank() + bio_matched['pred_pridict2'].rank(), bio_matched['overall_avg_NCcorr_%'])
print(f'New DP6 Ensemble (DP + new DP6 + PRIDICT2): Spearman R={r_ens_new:.4f} (p={p_ens_new:.6f})')

# Disagreement subset
dis_mask = bio_matched['Disagreement_Subcategory'] == 'Disagreement'
r_dis_new, _ = spearmanr(avg_new[dis_mask], bio_matched.loc[dis_mask, 'overall_avg_NCcorr_%'])
r_dis_old, _ = spearmanr(bio_matched.loc[dis_mask, 'pred_dp6'], bio_matched.loc[dis_mask, 'overall_avg_NCcorr_%'])
r_dis_pri, _ = spearmanr(bio_matched.loc[dis_mask, 'pred_pridict2'], bio_matched.loc[dis_mask, 'overall_avg_NCcorr_%'])
r_dis_dp, _ = spearmanr(bio_matched.loc[dis_mask, 'pred_dp_base'], bio_matched.loc[dis_mask, 'overall_avg_NCcorr_%'])
print(f'\nDisagreement Subset (N={dis_mask.sum()}):')
print(f'  DP-Base: R={r_dis_dp:.4f}')
print(f'  PRIDICT2: R={r_dis_pri:.4f}')
print(f'  July DP6: R={r_dis_old:.4f}')
print(f'  Retrained DP6: R={r_dis_new:.4f}')
