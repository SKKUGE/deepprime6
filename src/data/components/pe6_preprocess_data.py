from dataclasses import asdict
from typing import List, Tuple

import pandas as pd
from Bio.Seq import Seq, reverse_complement, transcribe
from Bio.SeqUtils import MeltingTemp as mt

from src.utils.dataprep import RENAME_MAP, RENAME_MAP_FOR_VIS

from .data_model.biofeatures import (
    EditTypeClass,
    MFEData,
    PegRNAExtensionData,
    TargetSequenceData,
    TmData,
    TmSequences,
)

UPSTREAM_LENGTH_TO_NICK = 21  # 74nt start to pegRNA nicking site
DOWNSTREAM_IDX_TO_NICK = 53  # 74nt end to pegRNA nicking site


def preprocess_data(
    data: pd.DataFrame = None,
    data_dir: str = None,
    output_file: str = None,
) -> pd.DataFrame:
    """Preprocesses the data for further analysis.

    Args:
        data (pd.DataFrame, optional): Input DataFrame to process. If None, loads from data_dir.
        data_dir (str, optional): The directory path where the data is located. Used if data is None.
        output_file (str, optional): The name of the output file. If None, returns DataFrame without saving.

    Returns:
        pd.DataFrame: The preprocessed data.

    Raises:
        ValueError: If both data and data_dir are None.
    """
    import pathlib

    from genet.predict import SpCas9

    # Load data if not provided
    if data is None:
        if data_dir is None:
            raise ValueError("Either 'data' or 'data_dir' must be provided")
        source = load_data(data_dir)
    else:
        source = data.copy()

    # Standardize column names
    column_mapping = {
        "Edit_len": "Edit length",
        "Edit_pos": "Edit position",
        "PBS_len": "PBS_length",
        "RTT_len": "RT_length",
        "WideTargetSequence": "WideTargetSequence",  # No change
        "Guide": "Guide",  # No change
    }
    source = source.rename(columns=column_mapping)
    
    if "PBS_length" not in source.columns and "PBS" in source.columns:
        source["PBS_length"] = source["PBS"].apply(len)
    if "RT_length" not in source.columns and "RTT" in source.columns:
        source["RT_length"] = source["RTT"].apply(len)

    

    # Process data
    data = calculate_guide_features(source)

    # Extend data properties related to sequences
    data = pd.concat(
        [
            data,
            data.apply(
                lambda x: determine_seqs(
                    alt_type=x["Edit_type"],
                    alt_len=x["Edit length"],
                    wt_seq=x[x["ContextSeqUsed"]], # Use the same context as Nicking
                    pbs_seq=x["PBS"],
                    rt_seq=x["RTT"],
                    nick_index=x["Nicking"],
                ),
                axis=1,
            ).rename("TmSequences"),
        ],
        axis=1,
    )
    data = determine_secondary_structure(data)
    data = make_output_df(data)

    # Filter out invalid sequences for SpCas9 prediction
    # Ensure deepspcas9_guide_30 only contains A, C, G, T (case insensitive)
    valid_seq_mask = data["deepspcas9_guide_30"].str.fullmatch(r"^[ACGTacgt]+$")
    if not valid_seq_mask.all():
        print(
            f"Warning: Dropping {sum(~valid_seq_mask)} rows with invalid deepspcas9_guide_30 sequences."
        )
        data = data[valid_seq_mask].copy()

    try:
        unique_guides = data["deepspcas9_guide_30"].unique().tolist()
        pred_res = SpCas9().predict(unique_guides)
        score_map = dict(zip(pred_res["Target"], pred_res["SpCas9"]))
        data["DeepSpCas9_score"] = data["deepspcas9_guide_30"].map(score_map)
    except Exception as e:
        print(f"Warning: DeepSpCas9 prediction failed with error: {e}")
        print("Filling DeepSpCas9_score with dummy value (0.0).")
        data["DeepSpCas9_score"] = 0.0

    # Preprocess: melting data
    data = data.rename(columns=RENAME_MAP)
    data = data.reset_index(drop=False)

    # Remove duplicate columns if any
    data = data.loc[:, ~data.columns.duplicated()]

    # Fill in missing and erroneous prime editing efficiencies
    data = data.fillna(0)

    # Rename columns for interpretability
    data = data.rename(columns=RENAME_MAP_FOR_VIS)

    # Save only if output_file is specified
    if output_file is not None:
        if data_dir is not None:
            save_path = f"{pathlib.Path(data_dir).parent}/{output_file}"
        else:
            save_path = output_file
        data.to_parquet(save_path)

    return data


def load_data(data_dir: str = "data/") -> pd.DataFrame:
    """Load the PE6 dataset from a parquet file or csv file.

    Args:
        data_dir (str): The directory containing the PE6 dataset file.

    Returns:
        pd.DataFrame: The PE6 dataset.
    """
    if data_dir.endswith(".csv"):
        return pd.read_csv(data_dir)
    return pd.read_parquet(data_dir)


def find_all_indices(main_string: str, query_string: str) -> tuple[int, int]:
    """Find all start and end indices of query_string in main_string.

    Args:
    main_string (str): The string to search in
    query_string (str): The substring to search for

    Returns:
    list of tuples: Each tuple contains the start and end indices of an occurrence
    """
    import re

    query_result = [
        (m.start(), m.end() - 1) for m in re.finditer(re.escape(query_string), main_string)
    ]
    return query_result


def calculate_guide_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates guide features based on the given DataFrame.
    It prefers 'OligoSequence_fixed_length' for context if available, 
    otherwise falls back to 'WideTargetSequence'.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with additional columns "GuideStart", "GuideEnd", and "Nicking".
    """
    context_col = "OligoSequence_fixed_length" if "OligoSequence_fixed_length" in df.columns else "WideTargetSequence"
    
    def select_best_index(main_seq, guide_seq, indices):
        if len(indices) == 0:
            return (None, None)
        if len(indices) == 1:
            return indices[0]
        
        # Try to find occurrence followed by NGG PAM
        pam_scores = []
        for start, end in indices:
            pam_start = end + 1
            pam = main_seq[pam_start:pam_start+3]
            if len(pam) == 3 and pam[1:3] == "GG":
                pam_scores.append((start, end, True))
            else:
                pam_scores.append((start, end, False))
        
        pam_matches = [idx for idx in pam_scores if idx[2]]
        if len(pam_matches) == 1:
            return pam_matches[0][0], pam_matches[0][1]
        
        # If still ambiguous or no PAM, return the first one
        return indices[0]

    query: pd.Series = df[[context_col, "Guide"]].apply(
        lambda x: select_best_index(x[context_col], x["Guide"], find_all_indices(x[context_col], x["Guide"])),
        axis=1,
    )
    start_idx = query.str[0]
    end_idx = query.str[1]

    df = df.assign(GuideStart=start_idx, GuideEnd=end_idx, ContextSeqUsed=context_col)
    df["Nicking"] = df["GuideEnd"] - 3
    return df


# TODO: Write test for this function. Check whether it can calculate input params as expected (Use DeepPrime dataset).
def determine_seqs(
    alt_type: str,
    alt_len: int,
    wt_seq: str,
    pbs_seq: str,
    rt_seq: str,
    nick_index: int,
) -> TmSequences:
    """Determine various sequences for melting temperature calculations.

    This function takes in the PAM nick position, alteration type, alteration length,
    wild-type sequence, PBS sequence, and RT sequence as input. It calculates and returns
    a dictionary containing sequences required for melting temperature calculations.

    Args:
        pam_nick (int): The PAM nick position.
        alt_type (str): The type of alteration (sub, ins, del).
        alt_len (int): The length of the alteration.
        wt_seq (str): The wild-type sequence.
        pbs_seq (str): The PBS sequence.
        rt_seq (str): The RT sequence.

    Returns:
        TmData: A named tuple containing the following sequences:
            - Tm1_PBS: The reverse complement of the PBS sequence.
            - Tm2_RTT_cTarget_sameLength: The portion of the wild-type sequence that corresponds
              to the RT sequence, starting from the PAM nick position.
            - Tm3_RTT_cTarget_replaced: The same portion of the wild-type sequence as
              Tm2_RTT_cTarget_sameLength, but modified based on the alteration type and length.
            - Tm4_cDNA_PAM_oppositeTarget: A tuple containing the RT sequence and its reverse
              complement, which is used for calculating the melting temperature.
            - Tm5_RTT_cDNA: The reverse complement of the RT sequence.

    Raises:
        ValueError: If the alteration type is invalid.
    """
    # pam_nick: int,    # To be inferred from the data

    tm1_pbs = transcribe(pbs_seq)  # genet: transcribe(pbs) → T→U conversion
    tm2_rtt_ctarget = wt_seq[
        nick_index : nick_index + len(rt_seq)
    ]  # ASSUMPTION: The input is + strand

    if alt_type.lower().startswith("sub"):
        tm2new_rtt_ctarget = tm2_rtt_ctarget
        tm3_antiSeq = reverse_complement(tm2_rtt_ctarget)
    elif alt_type.lower().startswith("ins"):
        tm2new_rtt_ctarget = wt_seq[nick_index : nick_index + len(rt_seq) - alt_len]
        tm3_antiSeq = reverse_complement(tm2new_rtt_ctarget)
    elif alt_type.lower().startswith("del"):
        tm2new_rtt_ctarget = wt_seq[nick_index : nick_index + len(rt_seq) + alt_len]
        tm3_antiSeq = reverse_complement(tm2new_rtt_ctarget)
    else:
        raise ValueError(f"Invalid alteration type: {alt_type.lower()}")

    # genet: seq_Tm4 = [back_transcribe(reverse_complement(rtt)), sTm4antiSeq]
    # For DNA input, back_transcribe(reverse_complement(x)) == reverse_complement(x)
    tm4_seq1 = reverse_complement(rt_seq)
    tm4_seq2 = tm3_antiSeq
    # genet: seq_Tm5 = transcribe(rtt)
    tm5_rtt_cdna = transcribe(rt_seq)
    tmdata = TmSequences(
        Tm1_PBS_seq=tm1_pbs,
        Tm2_RTT_cTarget_sameLength_seq=tm2_rtt_ctarget,
        Tm3_RTT_cTarget_replaced_seq=tm2new_rtt_ctarget,
        Tm4_cDNA_PAM_oppositeTarget_seq=(tm4_seq1, tm4_seq2),
        Tm5_RTT_cDNA_seq=tm5_rtt_cdna,
    )
    return tmdata


def _process_secondary_structure_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    chunk_df["TmData"] = chunk_df["TmSequences"].apply(lambda x: determine_tm(x))
    chunk_df["PegRNAExtensionData"] = chunk_df[["PBS", "RTT"]].apply(
        lambda x: determine_GC(x["PBS"], x["RTT"]), axis=1
    )
    chunk_df["MFEData"] = chunk_df[["PBS", "RTT", "Guide"]].apply(
        lambda x: determine_MFE(x["PBS"], x["RTT"], x["Guide"]), axis=1
    )
    return chunk_df


def determine_secondary_structure(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    from multiprocessing import Pool

    num_cores = min(32, len(df))
    df_split = np.array_split(df, num_cores)

    with Pool(num_cores) as p:
        processed_chunks = p.map(_process_secondary_structure_chunk, df_split)

    return pd.concat(processed_chunks)


def determine_tm(sequence_data: TmSequences) -> TmData:
    """Determines the melting temperatures (Tm) for different sequences in the given
    `sequence_data`.

    Args:
        sequence_data (TmSequences): The input TmSequences object containing the sequences for which Tm needs to be determined.

    Returns:
        TmData: The TmData object containing the calculated Tm values.
    """

    def _calculate_tm(seq: str, nn_table: dict) -> float:
        if len(seq) < 2:
            return 0.0
        try:
            return mt.Tm_NN(seq=Seq(seq), nn_table=nn_table)
        except Exception:
            return 0.0

    def _calculate_tm4(seq_pairs: Tuple[str, str]) -> float:
        # genet's Tm4 iterates char-by-char, keeping only the LAST pair's result.
        # This is an acknowledged bug in genet, but the model was trained with it.
        seq1, seq2 = seq_pairs
        fTm4 = 0.0
        for s1, s2 in zip(str(seq1), str(seq2)):
            try:
                fTm4 = mt.Tm_NN(seq=Seq(s1), c_seq=Seq(s2), nn_table=mt.DNA_NN3)
            except ValueError:
                fTm4 = 0
        return fTm4

    tm1 = _calculate_tm(sequence_data.Tm1_PBS_seq, mt.R_DNA_NN1)
    tm2 = _calculate_tm(sequence_data.Tm2_RTT_cTarget_sameLength_seq, mt.DNA_NN3)
    tm3 = _calculate_tm(sequence_data.Tm3_RTT_cTarget_replaced_seq, mt.DNA_NN3)
    try:
        tm4 = (  # BUG: The else case is applied to all; found 2025-11-11
            _calculate_tm4(sequence_data.Tm4_cDNA_PAM_oppositeTarget_seq)
            if sequence_data.Tm4_cDNA_PAM_oppositeTarget_seq
            else 0.0
        )
    except ValueError:
        tm4 = 0.0
    tm5 = _calculate_tm(sequence_data.Tm5_RTT_cDNA_seq, mt.R_DNA_NN1)

    return TmData(
        Tm1_PBS=tm1,
        Tm2_RTT_cTarget_sameLength=tm2,
        Tm3_RTT_cTarget_replaced=tm3,
        Tm4_cDNA_PAM_oppositeTarget=tm4,
        Tm5_RTT_cDNA=tm5,
        deltaTm_Tm4_Tm2=tm4 - tm2,
    )


def determine_GC(
    sPBSSeq: str,
    sRTSeqAlt: str,
) -> PegRNAExtensionData:
    """Determines the GC content and count for the given PBSSeq and RTSeqAlt sequences.

    Args:
        sPBSSeq (str): The PBS sequence.
        sRTSeqAlt (str): The RT sequence alternative.

    Returns:
        PegRNAExtensionData: An object containing the GC count and content for the PBS, RT, and combined sequences.
    """

    from Bio.SeqUtils import gc_fraction as gc

    nGCcnt1 = sPBSSeq.count("G") + sPBSSeq.count("C")
    nGCcnt2 = sRTSeqAlt.count("G") + sRTSeqAlt.count("C")
    nGCcnt3 = (sPBSSeq + sRTSeqAlt).count("G") + (sPBSSeq + sRTSeqAlt).count("C")
    fGCcont1 = 100 * gc(sPBSSeq)
    fGCcont2 = 100 * gc(sRTSeqAlt)
    fGCcont3 = 100 * gc(sPBSSeq + sRTSeqAlt)

    return PegRNAExtensionData(
        GC_count_PBS=nGCcnt1,
        GC_count_RTT=nGCcnt2,
        GC_count_RT_PBS=nGCcnt3,
        GC_contents_PBS=fGCcont1,
        GC_contents_RTT=fGCcont2,
        GC_contents_RT_PBS=fGCcont3,
    )


def determine_MFE(
    sPBSSeq: str,
    sRTSeq: str,
    sGuideSeq: str,
) -> MFEData:
    """Determines the minimum free energy (MFE) for a given set of sequences.

    Args:
        sPBSSeq (str): The PBS (Primer Binding Site) sequence.
        sRTSeq (str): The RT (Reverse Transcriptase) sequence.
        sGuideSeq (str): The guide sequence.

    Returns:
        MFEData: An object containing the MFE values for RT + PBS + PolyT and Spacer.

    Raises:
        ValueError: If there is an error in the MFE calculation.
    """
    from RNA import fold_compound

    try:
        # MFE_3 - RT + PBS + PolyT
        # Transcribe T to U for correct RNA folding context
        sInputSeq = (reverse_complement(sPBSSeq + sRTSeq) + "TTTTTT").replace("T", "U")
        _, fMFE3 = fold_compound(sInputSeq).mfe()

        # MFE_4 - spacer only
        sInputSeq = sGuideSeq.replace("T", "U")
        _, fMFE4 = fold_compound(sInputSeq).mfe()

        return MFEData(
            MFE_RT_PBS_polyT=round(fMFE3, 1),
            MFE_Spacer=round(fMFE4, 1),
        )
    except Exception as e:
        raise ValueError(f"Error in MFE calculation: {e}")


def make_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process the input DataFrame and return a modified DataFrame with calculated features.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        pd.DataFrame: The modified DataFrame with calculated features.
    """

    # TODO: add test

    def calculate_RHA_len(sRTTSeq: str, nEditPos: int, nAltLen: int, sAltType: str) -> int:
        # TODO: add test
        """Calculate the length of the RHA (Right Homology Arm) based on the given parameters.

        Args:
            sRTTSeq (str): The RTT sequence.
            nEditPos (int): The position of the edit.
            nAltLen (int): The length of the alternative sequence.
            sAltType (str): The type of the alternative sequence.

        Returns:
            int: The length of the RHA.
        """
        if sAltType.lower().startswith("del"):
            return len(sRTTSeq) - nEditPos + 1
        else:
            return len(sRTTSeq) - nEditPos - nAltLen + 1

    def calculate_74nt_target_sequence(
        sWTSeq: str,
        nNickIndex: int,
        sPBS_RTSeq: str,
        PBSlen: int,
        RTlen: int,
        nEditPos: int,
    ) -> TargetSequenceData:
        # TODO: add test
        """Calculates the 74nt target sequence based on the given inputs.

        Args:
            sWTSeq (str): The wild-type sequence (e.g. WideTargetSequence).
            nNickIndex (int): The nick index in the sWTSeq.
            sPBS_RTSeq (str): The PBS-RT sequence.
            PBSlen (int): The length of the PBS sequence.
            RTlen (int): The length of the RT sequence.
            nEditPos (int): The edit position.

        Returns:
            TargetSequenceData: An object containing the calculated target sequence data, including the wild type sequence,
            prime edited sequence, and edit position.
        """
        # Define the target window
        start_idx = nNickIndex - UPSTREAM_LENGTH_TO_NICK
        end_idx = nNickIndex + DOWNSTREAM_IDX_TO_NICK
        
        # Robust extraction with padding if context is insufficient
        # This allows using shorter WideTargetSequence if necessary
        sWTSeq_padded = sWTSeq
        offset = 0
        
        if start_idx < 0:
            pad_len = abs(start_idx)
            sWTSeq_padded = ("N" * pad_len) + sWTSeq_padded
            offset = pad_len
            start_idx = 0
            end_idx += offset
        
        if end_idx > len(sWTSeq_padded):
            pad_len = end_idx - len(sWTSeq_padded)
            sWTSeq_padded = sWTSeq_padded + ("N" * pad_len)

        sWTSeq74: str = sWTSeq_padded[start_idx:end_idx]
        assert len(sWTSeq74) == 74, f"Length of sWTSeq74 is not 74: {len(sWTSeq74)}"

        # Similarly for Seq30 (DeepSpCas9 input)
        seq30_start = (nNickIndex + offset) - UPSTREAM_LENGTH_TO_NICK
        seq30_end = (nNickIndex + offset) + 9
        sSeq30: str = sWTSeq_padded[seq30_start:seq30_end]
        
        assert (
            len(sSeq30) == 30
        ), f"Length of sSeq30 (30-nt window for SpCas9) is not 30: {len(sSeq30)}"

        s5Bufferlen = UPSTREAM_LENGTH_TO_NICK - PBSlen
        s3Bufferlen = DOWNSTREAM_IDX_TO_NICK - RTlen
        sEDSeq74 = "x" * s5Bufferlen + sPBS_RTSeq + "x" * s3Bufferlen
        assert len(sEDSeq74) == 74, f"Length of sEDSeq74 is not 74: {len(sEDSeq74)}"

        return TargetSequenceData(
            wild_type_sequence=sWTSeq74.upper(),
            deepspcas9_guide_30=sSeq30.upper(),
            prime_edited_sequence=sEDSeq74.upper(),
            edit_position=nEditPos,
        )

    def determine_edit_type(sAltType: str) -> EditTypeClass:
        # TODO: add test
        """Determines the edit type based on the given alternative type.

        Args:
            sAltType (str): The alternative type.

        Returns:
            EditType: The determined edit type.

        Raises:
           AssertionError: If the alternative type is not a valid edit type.
        """
        bSub = int(sAltType.lower().startswith("sub"))
        bDel = int(sAltType.lower().startswith("del"))
        bIns = int(sAltType.lower().startswith("ins"))

        assert (
            sum([bSub, bIns, bDel]) == 1
        ), f"Invalid edit type: {sAltType.lower(), bSub, bIns, bDel}"

        return EditTypeClass(type_sub=bSub, type_ins=bIns, type_del=bDel)

    def unpack_dataclass_columns(dataclass_cols: List[str], df: pd.DataFrame) -> pd.DataFrame:
        for dataclass_col in dataclass_cols:
            # Expand the dataclass column into a DataFrame
            expanded_df = pd.DataFrame.from_records(df[dataclass_col].map(lambda x: asdict(x)))
            # Alignment: Ensure the index matches the original DataFrame
            expanded_df.index = df.index

            # Drop overlapping columns from the original DataFrame specific to the expanded data
            # to avoid duplicate columns after concatenation
            cols_to_drop = [col for col in expanded_df.columns if col in df.columns]
            if cols_to_drop:
                # Only drop if they are strictly not the dataclass column itself (which is dropped later)
                # But here we want to drop collisions from the *result*
                df = df.drop(columns=[c for c in cols_to_drop if c != dataclass_col])

            df = pd.concat(
                [
                    df.drop(columns=dataclass_col),
                    expanded_df,
                ],
                axis=1,
            )
        return df

    # Calculate features
    df["RT_PBS"] = df["RTT"] + df["PBS"]
    df["TS_PBS_RT"] = df["RT_PBS"].map(reverse_complement)
    df["RT_PBS_len"] = df["RT_PBS"].apply(len)
    df["RHA_len"] = df[["RTT", "Edit position", "Edit length", "Edit_type"]].apply(
        lambda x: calculate_RHA_len(*x), axis=1
    )
    # Use the same context sequence as used for guide finding
    context_col = df["ContextSeqUsed"].iloc[0] if "ContextSeqUsed" in df.columns else "WideTargetSequence"
    
    df["TargetSequenceData"] = df[
        [
            context_col,
            "Nicking",
            "TS_PBS_RT",
            "PBS_length",
            "RT_length",
            "Edit position",
        ]
    ].apply(
        lambda x: calculate_74nt_target_sequence(*x), axis=1
    )  # dataclass
    df["EditTypeClass"] = df["Edit_type"].apply(lambda x: determine_edit_type(x))  # dataclass
    df["Spacer"] = df["leading G"] + df["Guide"]
    # Unpack dataclasses into columns

    df = unpack_dataclass_columns(
        dataclass_cols=[
            "EditTypeClass",
            "MFEData",
            "PegRNAExtensionData",
            "TargetSequenceData",
            "TmData",
            "TmSequences",
        ],
        df=df,
    )

    hder_features = {
        # Sample ID
        "REF_ID": "ID",
        # pegRNA sequence features
        "RT_PBS": "RT-PBS",
        # pegRNA edit and length features
        "PBS_length": "PBS_len",
        "RT_length": "RTT_len",
        "RT_PBS_len": "RT-PBS_len",
        "Edit position": "Edit_pos",  # TODO: Add test for data integrity
        "Edit length": "Edit_len",
        "RHA_len": "RHA_len",
        # Target sequences
        "wild_type_sequence": "Target",
        "prime_edited_sequence": "Masked_EditSeq",
        # Edit types
        "type_sub": "type_sub",
        "type_ins": "type_ins",
        "type_del": "type_del",
        # Tm features: TmData
        "Tm1_PBS": "Tm1_PBS",
        "Tm2_RTT_cTarget_sameLength": "Tm2_RTT_cTarget_sameLength",
        "Tm3_RTT_cTarget_replaced": "Tm3_RTT_cTarget_replaced",
        "Tm4_cDNA_PAM_oppositeTarget": "Tm4_cDNA_PAM-oppositeTarget",
        "Tm5_RTT_cDNA": "Tm5_RTT_cDNA",
        "deltaTm_Tm4_Tm2": "deltaTm_Tm4-Tm2",
        # GC counts and contents: PegRNAExtensionData
        "GC_count_PBS": "GC_count_PBS",
        "GC_count_RTT": "GC_count_RTT",
        "GC_count_RT_PBS": "GC_count_RT-PBS",
        "GC_contents_PBS": "GC_contents_PBS",
        "GC_contents_RTT": "GC_contents_RTT",
        "GC_contents_RT_PBS": "GC_contents_RT-PBS",
        # RNA 2ndary structure features: MFEData
        "MFE_RT_PBS_polyT": "MFE_RT-PBS-polyT",
        "MFE_Spacer": "MFE_Spacer",
    }
    df_out = df.rename(columns=hder_features)

    return df_out
