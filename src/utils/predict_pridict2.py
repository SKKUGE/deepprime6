import os
import sys
import re
import pandas as pd
import numpy as np
import time

# Set up python paths to import from cloned PRIDICT2
pridict2_dir = os.path.abspath("PRIDICT2")
sys.path.append(pridict2_dir)

from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

import torch
import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pridict.pridictv2.utilities import *
from pridict.pridictv2.dataset import *
from pridict.pridictv2.predict_outcomedistrib import *
from trained_models.DeepCas9_TestCode import runprediction

def melting_temperature(protospacer, extension, RT, RToverhang, PBS, original_base, edited_base):
    protospacermt = mt.Tm_Wallace(Seq(protospacer))
    extensionmt = mt.Tm_Wallace(Seq(extension))
    RTmt = mt.Tm_Wallace(Seq(RT))
    RToverhangmt = mt.Tm_Wallace(Seq(RToverhang))
    PBSmt = mt.Tm_Wallace(Seq(PBS))

    if original_base == '-':
        original_base_mt = 0
        original_base_mt_nan = 1
    else:
        original_base_mt = mt.Tm_Wallace(Seq(original_base))
        original_base_mt_nan = 0

    if edited_base == '-':
        edited_base_mt = 0
        edited_base_mt_nan = 1
    else:
        edited_base_mt = mt.Tm_Wallace(Seq(edited_base))
        edited_base_mt_nan = 0

    return protospacermt, extensionmt, RTmt, RToverhangmt, PBSmt, original_base_mt, edited_base_mt, original_base_mt_nan, edited_base_mt_nan

def RToverhangmatches(RToverhang, edited_seq, RToverhangstartposition, RTlengthoverhang):
    def occurrences(string, sub):
        count = start = 0
        while True:
            start = string.find(sub, start) + 1
            if start > 0:
                count += 1
            else:
                return count
    RToverhangmatchcount = occurrences(
        edited_seq[RToverhangstartposition:RToverhangstartposition + RTlengthoverhang + 15], RToverhang)
    return RToverhangmatchcount

def multideepeditpositionfunc(originalbases, editedbases, deepeditposition):
    multideepeditlist = []
    for i in range(len(originalbases)):
        if originalbases[i] != editedbases[i]:
            multideepeditlist.append(deepeditposition+i)
    return multideepeditlist

def editorcharacteristics(editor):
    if editor == 'PE2-NGG':
        PAM = '(?=GG)'
        numberN = 1
        PAM_length = 3
        variant = 'PE2-NGG'
        protospacerlength = 19
        PAM_side = 'right'
        primescaffoldseq = 'GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC'
    return PAM, numberN, variant, protospacerlength, PAM_side, primescaffoldseq, PAM_length

def get_prieml_model_template():
    device = get_device(True, 0)
    wsize = 20
    normalize_opt = 'max'
    prieml_model = PRIEML_Model(device, wsize=wsize, normalize=normalize_opt, fdtype=torch.float32)
    return prieml_model

def load_pridict_model(run_ids=[0]):
    models_lst_dict = {}
    modellist = [
        ('PRIDICT1_1', 'pe_rnn_distribution_multidata', 'exp_2023-08-25_20-55-53'),
        ('PRIDICT1_2', 'pe_rnn_distribution_multidata', 'exp_2023-08-28_22-22-26')
    ]
    prieml_model = get_prieml_model_template()
    for model_desc_tup in modellist:
        models_lst = []
        model_id, __, mfolder = model_desc_tup
        for run_num in run_ids:
            model_dir = os.path.join(pridict2_dir, 'trained_models', model_id.lower(), mfolder, 'train_val', f'run_{run_num}')
            loaded_model = prieml_model.build_retrieve_models(model_dir)
            models_lst.append((loaded_model, model_dir))
        models_lst_dict[model_id] = models_lst
    return models_lst_dict

def deeppridict(pegdataframe, models_lst_dict):
    deepdfcols = ['wide_initial_target', 'wide_mutated_target', 'deepeditposition', 
                  'deepeditposition_lst', 'Correction_Type', 'Correction_Length', 
                  'protospacerlocation_only_initial', 'PBSlocation',
                  'RT_initial_location', 'RT_mutated_location',
                  'RToverhangmatches', 'RToverhanglength', 
                  'RTlength', 'PBSlength', 'RTmt', 'RToverhangmt','PBSmt','protospacermt',
                  'extensionmt','original_base_mt','edited_base_mt','original_base_mt_nan',
                  'edited_base_mt_nan']

    deepdf = pegdataframe[deepdfcols].copy()
    deepdf.insert(1, 'seq_id', list(range(len(deepdf))))
    deepdf['protospacerlocation_only_initial'] = deepdf['protospacerlocation_only_initial'].apply(lambda x: str(x))
    deepdf['PBSlocation'] = deepdf['PBSlocation'].apply(lambda x: str(x))
    deepdf['RT_initial_location'] = deepdf['RT_initial_location'].apply(lambda x: str(x))
    deepdf['RT_mutated_location'] = deepdf['RT_mutated_location'].apply(lambda x: str(x))
    deepdf['deepeditposition_lst'] = deepdf['deepeditposition_lst'].apply(lambda x: str(x))

    deepdf['edited_base_mt'] = deepdf.apply(lambda x: 0 if x.Correction_Type == 'Deletion' else x.edited_base_mt, axis=1)
    deepdf['original_base_mt'] = deepdf.apply(lambda x: 0 if x.Correction_Type == 'Insertion' else x.original_base_mt, axis=1)
    
    plain_tcols = ['averageedited', 'averageunedited', 'averageindel']
    cell_types = ['HEK', 'K562']
    batch_size = int(1500/len(cell_types))

    prieml_model = get_prieml_model_template()
    dloader = prieml_model.prepare_data(deepdf, None, cell_types=cell_types, y_ref=[], batch_size=batch_size)
    all_avg_preds = {} 

    for model_id, model_runs_lst in models_lst_dict.items():
        pred_dfs = []
        runs_c = 0
        for loaded_model_lst, model_dir in model_runs_lst:
            pred_df = prieml_model.predict_from_dloader_using_loaded_models(dloader, loaded_model_lst, y_ref=plain_tcols)
            pred_df['run_num'] = runs_c
            pred_dfs.append(pred_df)
            runs_c += 1
        pred_df_allruns = pd.concat(pred_dfs, axis=0, ignore_index=True)
        avg_preds = prieml_model.compute_avg_predictions(pred_df_allruns)
        avg_preds['model'] = model_id
        all_avg_preds[model_id] = avg_preds
    return all_avg_preds

def compute_average_predictions(df, grp_cols=['seq_id', 'dataset_name']):
    tcols = ['pred_averageedited', 'pred_averageunedited', 'pred_averageindel']
    agg_df = df.groupby(by=grp_cols)[tcols].mean()
    agg_df.reset_index(inplace=True)
    return agg_df

def find_exact_edit_info(wt, edited, edit_type):
    wt, edited = wt.upper(), edited.upper()
    if edit_type == 'Sub':
        for i in range(min(len(wt), len(edited))):
            if wt[i] != edited[i]:
                mismatch_start = i
                mismatch_end_wt = len(wt)
                mismatch_end_ed = len(edited)
                while mismatch_end_wt > mismatch_start and mismatch_end_ed > mismatch_start:
                    if wt[mismatch_end_wt-1] != edited[mismatch_end_ed-1]:
                        break
                    mismatch_end_wt -= 1
                    mismatch_end_ed -= 1
                orig_b = wt[mismatch_start:mismatch_end_wt]
                edit_b = edited[mismatch_start:mismatch_end_ed]
                return mismatch_start, orig_b, edit_b
    elif edit_type == 'Del':
        for i in range(min(len(wt), len(edited))):
            if wt[i] != edited[i]:
                del_len = len(wt) - len(edited)
                return i, wt[i:i+del_len], '-'
    elif edit_type == 'Ins':
        for i in range(min(len(wt), len(edited))):
            if wt[i] != edited[i]:
                ins_len = len(edited) - len(wt)
                return i, '-', edited[i:i+ins_len]
    return 0, '', ''

def get_pridict_input_seq(row):
    wt_seq = row['WildTypeSequence']
    edited_seq_raw = row['PrimeEditedSequence']
    edit_type = row['Edit_type']
    
    real_pos, orig_b, edit_b = find_exact_edit_info(wt_seq, edited_seq_raw, edit_type)
    
    left_seq = wt_seq[:real_pos]
    if edit_type == 'Sub':
        bracket = f'({orig_b}/{edit_b})'
        right_seq = wt_seq[real_pos+len(orig_b):]
    elif edit_type == 'Del':
        bracket = f'(-{orig_b})'
        right_seq = wt_seq[real_pos+len(orig_b):]
    elif edit_type == 'Ins':
        bracket = f'(+{edit_b})'
        right_seq = wt_seq[real_pos:]
    else:
        raise ValueError(f"Unknown edit type: {edit_type}")
        
    target_seq_with_bracket = left_seq + bracket + right_seq
    
    left_pad = ""
    if len(left_seq) < 100:
        left_pad = "A" * (100 - len(left_seq))
    right_pad = ""
    if len(right_seq) < 100:
        right_pad = "A" * (100 - len(right_seq))
        
    return left_pad + target_seq_with_bracket + right_pad

def primesequenceparsing(sequence: str) -> object:
    sequence = sequence.replace('\n','').replace(' ','').upper()
    five_prime_seq = sequence.split('(')[0]
    three_prime_seq = sequence.split(')')[1]

    if '/' in sequence:
        original_base = sequence.split('/')[0].split('(')[1]
        edited_base = sequence.split('/')[1].split(')')[0]
    elif '+' in sequence:
        original_base = '-'
        edited_base = sequence.split('+')[1].split(')')[0]
    elif '-' in sequence:
        original_base = sequence.split('-')[1].split(')')[0]
        edited_base = '-'

    if original_base == '-':
        original_seq = five_prime_seq + three_prime_seq
        mutation_type = 'Insertion'
        correction_length = len(edited_base)
    else:
        original_seq = five_prime_seq + original_base + three_prime_seq
        if edited_base == '-':
            mutation_type = 'Deletion'
            correction_length = len(original_base)
        elif len(original_base) == 1 and len(edited_base) == 1:
            mutation_type = '1bpReplacement'
            correction_length = len(original_base)
        else:
            mutation_type = 'MultibpReplacement'
            correction_length = len(original_base)

    if edited_base == '-':
        edited_seq = five_prime_seq + three_prime_seq
    else:
        edited_seq = five_prime_seq + edited_base.lower() + three_prime_seq

    basebefore_temp = five_prime_seq[-1:]
    baseafter_temp = three_prime_seq[:1]
    editposition_left = len(five_prime_seq)
    editposition_right = len(three_prime_seq)
    return original_base, edited_base, original_seq, edited_seq, editposition_left, editposition_right, mutation_type, correction_length, basebefore_temp, baseafter_temp

def extract_single_pegRNA_features(dfrow, editor='PE2-NGG'):
    sequence = get_pridict_input_seq(dfrow)
    name = dfrow['REF_ID']
    target_guide = dfrow['Guide']
    target_pbs = dfrow['PBS']
    target_rtt = dfrow['RTT']
    
    original_base, edited_base, original_seq, edited_seq, editposition_left, editposition_right, mutation_type, correction_length, basebefore_temp, baseafter_temp = primesequenceparsing(sequence)
    PAM, numberN, variant, protospacerlength, PAM_side, primescaffoldseq, PAM_length = editorcharacteristics(editor)

    if mutation_type == 'Deletion':
        correction_type = 'Deletion'
    elif mutation_type == 'Insertion':
        correction_type = 'Insertion'
    else:
        correction_type = 'Replacement'

    # Determine strand by checking where guide matches in the target sequences
    # We find if Guide matches original_seq or edited_seq (case-insensitive)
    # Check if target_guide[-19:] matches original_seq
    target_strand = 'Fw'
    local_spacer_start = original_seq.lower().find(target_guide[-19:].lower())
    if local_spacer_start == -1:
        # Check RC
        rc_guide = str(Seq(target_guide).reverse_complement())
        local_spacer_start = original_seq.lower().find(rc_guide[-19:].lower())
        if local_spacer_start != -1:
            target_strand = 'Rv'

    # If Rv, we work on reverse complement strand
    if target_strand == 'Fw':
        editposition = editposition_left
        curr_original_base = original_base
        curr_edited_base = edited_base
        curr_original_seq = original_seq
        curr_edited_seq = edited_seq
    else:
        editposition = editposition_right
        curr_original_base = str(Seq(original_base).reverse_complement()) if original_base != '-' else '-'
        curr_edited_base = str(Seq(edited_base).reverse_complement()) if edited_base != '-' else '-'
        curr_original_seq = str(Seq(original_seq).reverse_complement())
        curr_edited_seq = str(Seq(edited_seq).reverse_complement())

    # Re-evaluate local spacer start on the selected strand
    local_spacer_start = curr_original_seq.lower().find(target_guide[-19:].lower())
    if local_spacer_start == -1:
        rc_guide = str(Seq(target_guide).reverse_complement())
        local_spacer_start = curr_original_seq.lower().find(rc_guide[-19:].lower())
        if local_spacer_start == -1:
            # Fallback search - find most common sequence match
            for shift in range(len(curr_original_seq) - 19):
                cand = curr_original_seq[shift:shift+19]
                if cand.lower() == target_guide[-19:].lower() or cand.lower() == rc_guide[-19:].lower():
                    local_spacer_start = shift
                    break

    if local_spacer_start == -1:
        # Ultimate fallback: spacer is usually positioned around editposition-25 to editposition
        local_spacer_start = max(0, editposition - 22)

    # protospacerseq
    protospacerseq = 'G' + curr_original_seq[local_spacer_start + 19 - protospacerlength : local_spacer_start + 19]

    # XPAM is PAM index, typically 3bp downstream of spacer end
    XPAM = local_spacer_start + 19
    start = XPAM + (len(PAM) - 7) - 3  # Nick position

    PBSlength = len(target_pbs)
    
    # Robust string matching for PBS to avoid indexing shifts
    pbs_target_seq = str(Seq(target_pbs).reverse_complement())
    pbs_start = curr_original_seq.lower().find(pbs_target_seq.lower())
    if pbs_start != -1:
        PBS = curr_original_seq[pbs_start:pbs_start + len(target_pbs)]
    else:
        PBS = curr_original_seq[XPAM + (len(PAM) - 7) - protospacerlength + (protospacerlength - PBSlength) - 3:XPAM + (len(PAM) - 7) - 3]
    PBSrevcomp = str(Seq(PBS).reverse_complement())

    # Robust string matching for RTT to determine matched_rtoverhang
    rtt_target_seq = str(Seq(target_rtt).reverse_complement())
    rtt_start = curr_edited_seq.lower().find(rtt_target_seq.lower())
    if rtt_start != -1:
        RTseq = curr_edited_seq[rtt_start:rtt_start + len(target_rtt)]
        stop = rtt_start + len(target_rtt)
        if curr_edited_base == '-':
            matched_rtoverhang = stop - editposition
        else:
            matched_rtoverhang = stop - editposition - len(curr_edited_base)
    else:
        matched_rtoverhang = 10
        for RTlengthoverhang in range(1, 100):
            stop = editposition + len(curr_edited_base) + RTlengthoverhang
            if curr_edited_base == '-':
                stop -= 1
            RTseq = curr_edited_seq[start:stop]
            RTseqrevcomp = str(Seq(RTseq).reverse_complement())
            if RTseqrevcomp.lower() == target_rtt.lower():
                matched_rtoverhang = RTlengthoverhang
                break
        stop = editposition + len(curr_edited_base) + matched_rtoverhang
        if curr_edited_base == '-':
            stop -= 1
        RTseq = curr_edited_seq[start:stop]

    RTlengthoverhang = matched_rtoverhang
    RTseqrevcomp = str(Seq(RTseq).reverse_complement())
    RTseqoverhang = curr_edited_seq[stop - RTlengthoverhang:stop]
    RTseqoverhangrevcomp = str(Seq(RTseqoverhang).reverse_complement())

    # Features
    startposition = 10
    # WideTarget coordinates (offset anchored strictly at local_spacer_start)
    wide_start = local_spacer_start - 10
    wide_initial_target = curr_original_seq[wide_start : wide_start + 99]
    wide_mutated_target = curr_edited_seq[wide_start : wide_start + 99]

    # Ensure 99bp lengths
    if len(wide_initial_target) < 99:
        wide_initial_target = wide_initial_target + "A" * (99 - len(wide_initial_target))
    if len(wide_mutated_target) < 99:
        wide_mutated_target = wide_mutated_target + "A" * (99 - len(wide_mutated_target))

    # Edit position relative to wide target
    deepeditposition = startposition + (editposition - local_spacer_start)
    if mutation_type == 'MultibpReplacement':
        multideepeditpositions = multideepeditpositionfunc(curr_original_base, curr_edited_base, deepeditposition)
        if not multideepeditpositions:
            multideepeditpositions = [deepeditposition]
    else:
        multideepeditpositions = [deepeditposition]

    protospacerlocation_only_initial = [startposition, startposition + protospacerlength]
    PBSlocation = [startposition + protospacerlength - 3 - PBSlength, startposition + protospacerlength - 3]

    if correction_type == 'Replacement':
        RT_initial_location = [startposition + protospacerlength - 3, startposition + protospacerlength - 3 + len(RTseq)]
        RT_mutated_location = [startposition + protospacerlength - 3, startposition + protospacerlength - 3 + len(RTseq)]
    elif correction_type == 'Deletion':
        RT_initial_location = [startposition + protospacerlength - 3, startposition + protospacerlength - 3 + len(RTseq) + correction_length]
        RT_mutated_location = [startposition + protospacerlength - 3, startposition + protospacerlength - 3 + len(RTseq)]
    elif correction_type == 'Insertion':
        RT_initial_location = [startposition + protospacerlength - 3, startposition + protospacerlength - 3 + len(RTseq) - correction_length]
        RT_mutated_location = [startposition + protospacerlength - 3, startposition + protospacerlength - 3 + len(RTseq)]

    protospacermt, extensionmt, RTmt, RToverhangmt, PBSmt, original_base_mt, edited_base_mt, original_base_mt_nan, edited_base_mt_nan = melting_temperature(
        protospacerseq, RTseqrevcomp + PBSrevcomp, RTseqrevcomp, RTseqoverhangrevcomp, PBSrevcomp, curr_original_base, curr_edited_base
    )
    rtoverhangmatch = RToverhangmatches(RTseqoverhang, curr_edited_seq, stop - RTlengthoverhang, RTlengthoverhang)

    matched_features = {
        'PRIDICT2_Format': sequence,
        'Original_Sequence': curr_original_seq,
        'Edited_Sequence': curr_edited_seq,
        'Target-Strand': target_strand,
        'Mutation_Type': mutation_type,
        'Correction_Type': correction_type,
        'Correction_Length': correction_length,
        'Editing_Position': editposition - start,
        'PBSlength': PBSlength,
        'RToverhanglength': RTlengthoverhang,
        'RTlength': len(RTseq),
        'EditedAllele': curr_edited_base,
        'OriginalAllele': curr_original_base,
        'Spacer-Sequence': protospacerseq,
        'PBSrevcomp': PBSrevcomp,
        'RTseqoverhangrevcomp': RTseqoverhangrevcomp,
        'RTrevcomp': RTseqrevcomp,
        'Scaffold_Optimized': primescaffoldseq,
        'pegRNA': protospacerseq + primescaffoldseq + RTseqrevcomp + PBSrevcomp,
        'Editor_Variant': variant,
        'protospacermt': protospacermt,
        'extensionmt': extensionmt,
        'RTmt': RTmt,
        'RToverhangmt': RToverhangmt,
        'PBSmt': PBSmt,
        'original_base_mt': original_base_mt,
        'edited_base_mt': edited_base_mt,
        'original_base_mt_nan': original_base_mt_nan,
        'edited_base_mt_nan': edited_base_mt_nan,
        'RToverhangmatches': rtoverhangmatch,
        'wide_initial_target': wide_initial_target,
        'wide_mutated_target': wide_mutated_target,
        'protospacerlocation_only_initial': protospacerlocation_only_initial,
        'PBSlocation': PBSlocation,
        'RT_initial_location': RT_initial_location,
        'RT_mutated_location': RT_mutated_location,
        'deepeditposition': deepeditposition,
        'deepeditposition_lst': multideepeditpositions,
        'sequence_name': name
    }
    return matched_features

def main():
    print("Loading test library...")
    df = pd.read_parquet("data/rq3_test_library.parquet")
    print(f"Loaded {len(df)} sequences.")

    print("Loading PRIDICT2.0 trained models...")
    models_list = load_pridict_model(run_ids=[0])

    features_list = []
    skipped_indices = []

    print("Extracting features for target pegRNAs...")
    t0 = time.time()
    for idx, row in df.iterrows():
        try:
            feat = extract_single_pegRNA_features(row)
            if feat is not None:
                features_list.append(feat)
            else:
                skipped_indices.append(idx)
        except Exception as e:
            skipped_indices.append(idx)
            pass
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(df)} sequences...")

    print(f"Feature extraction completed in {time.time() - t0:.2f}s.")
    print(f"Successfully matched: {len(features_list)} / {len(df)}")
    if skipped_indices:
        print(f"Warning: {len(skipped_indices)} sequences could not be matched.")

    pegdataframe = pd.DataFrame(features_list)

    print("Running PRIDICT2.0 deep model inference...")
    t1 = time.time()
    all_avg_preds = deeppridict(pegdataframe, models_list)
    print(f"Deep model inference completed in {time.time() - t1:.2f}s.")

    tmp = [all_avg_preds[model_id] for model_id in all_avg_preds]
    tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
    agg_df = compute_average_predictions(tmp_df, grp_cols=['seq_id', 'dataset_name'])

    cell_types = ['HEK', 'K562']
    for cell_type in cell_types:
        cond = agg_df['dataset_name'] == cell_type
        avg_edited_eff = agg_df.loc[cond, 'pred_averageedited'].values * 100
        pegdataframe[f'PRIDICT2_0_editing_Score_deep_{cell_type}'] = avg_edited_eff

    score_df = pegdataframe[['sequence_name', 'PRIDICT2_0_editing_Score_deep_HEK', 'PRIDICT2_0_editing_Score_deep_K562']]
    output_df = df.merge(score_df, left_on='REF_ID', right_on='sequence_name', how='left')

    if 'sequence_name' in output_df.columns:
        output_df = output_df.drop(columns=['sequence_name'])

    os.makedirs("data", exist_ok=True)
    output_df.to_parquet("data/rq3_test_library_with_pridict2.parquet")
    output_df.to_csv("data/rq3_predictions_pridict2.csv", index=False)
    print("Saved PRIDICT2.0 predictions to data/rq3_predictions_pridict2.csv")

if __name__ == "__main__":
    main()
