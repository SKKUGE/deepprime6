import torch

RENAME_MAP = {
    #  Combined samples
    "Normalized+3rep_HEK-M-1-7D+pe_ratio_%": "comb_vis+HEK-M-1-7D%",
    "Normalized+3rep_HEK-M-2-7D+pe_ratio_%": "comb_vis+HEK-M-2-7D%",
    "Normalized+3rep_HEK-M-3-7D+pe_ratio_%": "comb_vis+HEK-M-3-7D%",
    "Normalized+3rep_HEK-M-4-7D+pe_ratio_%": "comb_vis+HEK-M-4-7D%",
    "Normalized+3rep_HEK-M-5-7D+pe_ratio_%": "comb_vis+HEK-M-5-7D%",
    "Normalized+3rep_HEK-M-6-7D+pe_ratio_%": "comb_vis+HEK-M-6-7D%",
    "Normalized+3rep_HEK-M-7-7D+pe_ratio_%": "comb_vis+HEK-M-7-7D%",
    "Normalized+3rep_HEK-M-8-7D+pe_ratio_%": "comb_vis+HEK-M-8-7D%",
    "Normalized+3rep_HEK-M-9-7D+pe_ratio_%": "comb_vis+HEK-M-9-7D%",
}


RENAME_MAP_FOR_VIS = {
    # Separated samples
    "indiv_vis+HEK-M-1-7D-B2%": "PEmax-B2",
    "indiv_vis+HEK-M-2-7D-B2%": "PEmaxdRNaseH-B2",
    "indiv_vis+HEK-M-3-7D-B2%": "PE6a(+PEmaxCas9)-B2",
    "indiv_vis+HEK-M-4-7D-B2%": "PE6b(+PEmaxCas9)-B2",
    "indiv_vis+HEK-M-5-7D-B2%": "PE6c(+PEmaxCas9)-B2",
    "indiv_vis+HEK-M-6-7D-B2%": "PE6d(+PEmaxCas9)-B2",
    "indiv_vis+HEK-M-7-7D-B2%": "PE6e(+dRNaseH)-B2",
    "indiv_vis+HEK-M-8-7D-B2%": "PE6f(+dRNaseH)-B2",
    "indiv_vis+HEK-M-9-7D-B2%": "PE6g(+dRNaseH)-B2",
    "indiv_vis+HEK-M-1-7D-B1%": "PEmax-B1",
    "indiv_vis+HEK-M-2-7D-B1%": "PEmaxdRNaseH-B1",
    "indiv_vis+HEK-M-3-7D-B1%": "PE6a(+PEmaxCas9)-B1",
    "indiv_vis+HEK-M-4-7D-B1%": "PE6b(+PEmaxCas9)-B1",
    "indiv_vis+HEK-M-5-7D-B1%": "PE6c(+PEmaxCas9)-B1",
    "indiv_vis+HEK-M-6-7D-B1%": "PE6d(+PEmaxCas9)-B1",
    "indiv_vis+HEK-M-7-7D-B1%": "PE6e(+dRNaseH)-B1",
    "indiv_vis+HEK-M-8-7D-B1%": "PE6f(+dRNaseH)-B1",
    "indiv_vis+HEK-M-9-7D-B1%": "PE6g(+dRNaseH)-B1",
    "indiv_vis+HEK-S-1-14D-B2%": "PEmax-B2",
    "indiv_vis+HEK-S-2-14D-B2%": "PEmaxdRNaseH-B2",
    "indiv_vis+HEK-S-3-14D-B2%": "PE6a(+PEmaxCas9)-B2",
    "indiv_vis+HEK-S-4-14D-B2%": "PE6b(+PEmaxCas9)-B2",
    "indiv_vis+HEK-S-5-14D-B2%": "PE6c(+PEmaxCas9)-B2",
    "indiv_vis+HEK-S-6-14D-B2%": "PE6d(+PEmaxCas9)-B2",
    "indiv_vis+HEK-S-10-14D-B2%": "PE6ec-B2",
    "indiv_vis+HEK-S-11-14D-B2%": "PE6fc-B2",
    "indiv_vis+HEK-S-12-14D-B2%": "PE6gc-B2",
    "indiv_vis+HEK-S-13-14D-B2%": "PE6ed-B2",
    "indiv_vis+HEK-S-14-14D-B2%": "PE6fd-B2",
    "indiv_vis+HEK-S-15-14D-B2%": "PE6gd-B2",
    "indiv_vis+HEK-S-1-14D-B1%": "PEmax-B1",
    "indiv_vis+HEK-S-2-14D-B1%": "PEmaxdRNaseH-B1",
    "indiv_vis+HEK-S-3-14D-B1%": "PE6a(+PEmaxCas9)-B1",
    "indiv_vis+HEK-S-4-14D-B1%": "PE6b(+PEmaxCas9)-B1",
    "indiv_vis+HEK-S-5-14D-B1%": "PE6c(+PEmaxCas9)-B1",
    "indiv_vis+HEK-S-6-14D-B1%": "PE6d(+PEmaxCas9)-B1",
    "indiv_vis+HEK-S-10-14D-B1%": "PE6ec-B1",
    "indiv_vis+HEK-S-11-14D-B1%": "PE6fc-B1",
    "indiv_vis+HEK-S-12-14D-B1%": "PE6gc-B1",
    "indiv_vis+HEK-S-13-14D-B1%": "PE6ed-B1",
    "indiv_vis+HEK-S-14-14D-B1%": "PE6fd-B1",
    "indiv_vis+HEK-S-15-14D-B1%": "PE6gd-B1",
    #  Combined samples
    "comb_vis+HEK-M-1-7D%": "PEmax",
    "comb_vis+HEK-M-2-7D%": "PEmaxdRNaseH",
    "comb_vis+HEK-M-3-7D%": "PE6a(+PEmaxCas9)",
    "comb_vis+HEK-M-4-7D%": "PE6b(+PEmaxCas9)",
    "comb_vis+HEK-M-5-7D%": "PE6c(+PEmaxCas9)",
    "comb_vis+HEK-M-6-7D%": "PE6d(+PEmaxCas9)",
    "comb_vis+HEK-M-7-7D%": "PE6e(+dRNaseH)",
    "comb_vis+HEK-M-8-7D%": "PE6f(+dRNaseH)",
    "comb_vis+HEK-M-9-7D%": "PE6g(+dRNaseH)",
    "comb_vis+HEK-S-1-14D%": "PEmax",
    "comb_vis+HEK-S-2-14D%": "PEmaxdRNaseH",
    "comb_vis+HEK-S-3-14D%": "PE6a(+PEmaxCas9)",
    "comb_vis+HEK-S-4-14D%": "PE6b(+PEmaxCas9)",
    "comb_vis+HEK-S-5-14D%": "PE6c(+PEmaxCas9)",
    "comb_vis+HEK-S-6-14D%": "PE6d(+PEmaxCas9)",
    "comb_vis+HEK-S-10-14D%": "PE6ec",
    "comb_vis+HEK-S-11-14D%": "PE6fc",
    "comb_vis+HEK-S-12-14D%": "PE6gc",
    "comb_vis+HEK-S-13-14D%": "PE6ed",
    "comb_vis+HEK-S-14-14D%": "PE6fd",
    "comb_vis+HEK-S-15-14D%": "PE6gd",
    # Samples added for PE7 comparison
    "PE6-2rep1": "PEmaxdRNaseH-#1",
    "PE6-2rep2": "PEmaxdRNaseH-#2",
    "PE6-2rep3": "PEmaxdRNaseH-#3",
    "PE6-3rep1": "PE6b-#1",
    "PE6-3rep2": "PE6b-#2",
    "PE6-3rep3": "PE6b-#3",
    "PE6-4rep1": "PE6c-#1",
    "PE6-4rep2": "PE6c-#2",
    "PE6-4rep3": "PE6c-#3",
    "PE7-2rep1": "PE7maxdRNaseH-#1",
    "PE7-2rep2": "PE7maxdRNaseH-#2",
    "PE7-2rep3": "PE7maxdRNaseH-#3",
    "PE7-3rep1": "PE7b-#1",
    "PE7-3rep2": "PE7b-#2",
    "PE7-3rep3": "PE7b-#3",
    "PE7-4rep1": "PE7c-#1",
    "PE7-4rep2": "PE7c-#2",
    "PE7-4rep3": "PE7c-#3",
    # Averaged PE 6 and 7 replicate results
    "PE6-2avg": "PEmaxdRNaseH-avg",
    "PE6-3avg": "PE6b-avg",
    "PE6-4avg": "PE6c-avg",
    "PE7-2avg": "PE7maxdRNaseH-avg",
    "PE7-3avg": "PE7b-avg",
    "PE7-4avg": "PE7c-avg",
}


AVG_PAIR = {
    "PEmax": (
        "HEK-M-1-7D-B1+total_read_counts",
        "HEK-M2-1-7D-B1+total_read_counts",
    ),
    "PEmaxdRNaseH": (
        "HEK-M-2-7D-B1+total_read_counts",
        "HEK-M2-2-7D-B1+total_read_counts",
    ),
    "PE6a(+PEmaxCas9)": (
        "HEK-M-3-7D-B1+total_read_counts",
        "HEK-M2-3-7D-B1+total_read_counts",
    ),
    "PE6b(+PEmaxCas9)": (
        "HEK-M-4-7D-B1+total_read_counts",
        "HEK-M2-4-7D-B1+total_read_counts",
    ),
    "PE6c(+PEmaxCas9)": (
        "HEK-M-5-7D-B1+total_read_counts",
        "HEK-M2-5-7D-B1+total_read_counts",
    ),
    "PE6d(+PEmaxCas9)": (
        "HEK-M-6-7D-B1+total_read_counts",
        "HEK-M2-6-7D-B1+total_read_counts",
    ),
    "PE6e(+dRNaseH)": (
        "HEK-M-7-7D-B2+total_read_counts",
        "HEK-M-7-7D-B1+total_read_counts",
    ),
    "PE6f(+dRNaseH)": (
        "HEK-M-8-7D-B2+total_read_counts",
        "HEK-M-8-7D-B1+total_read_counts",
    ),
    "PE6g(+dRNaseH)": (
        "HEK-M-9-7D-B2+total_read_counts",
        "HEK-M-9-7D-B1+total_read_counts",
    ),
}

# Derived normalized editing efficiency column pairs corresponding strictly to AVG_PAIR
EFF_PAIR = {
    k: tuple(
        c.replace("+total_read_counts", "+pe_ratio_%")
        if c.startswith("Normalized+")
        else f"Normalized+{c.replace('+total_read_counts', '+pe_ratio_%')}"
        for c in v
    )
    for k, v in AVG_PAIR.items()
}

IDX_TO_PE_TYPE = {
    1: "PEmax",
    2: "PEmaxdRNaseH",
    3: "PE6a(+PEmaxCas9)",
    4: "PE6b(+PEmaxCas9)",
    5: "PE6c(+PEmaxCas9)",
    6: "PE6d(+PEmaxCas9)",
    7: "PE6e(+dRNaseH)",
    8: "PE6f(+dRNaseH)",
    9: "PE6g(+dRNaseH)",
}

PE_TYPE_ALIASES = {
    "PE2max": "PEmax",
    "PEmax": "PEmax",
    "PE2max-dRNaseH": "PEmaxdRNaseH",
    "PEmax-dRNaseH": "PEmaxdRNaseH",
    "PEmaxdRNaseH": "PEmaxdRNaseH",
    "PE6a": "PE6a(+PEmaxCas9)",
    "PE6a(+PEmaxCas9)": "PE6a(+PEmaxCas9)",
    "PE6b": "PE6b(+PEmaxCas9)",
    "PE6b(+PEmaxCas9)": "PE6b(+PEmaxCas9)",
    "PE6c": "PE6c(+PEmaxCas9)",
    "PE6c(+PEmaxCas9)": "PE6c(+PEmaxCas9)",
    "PE6d": "PE6d(+PEmaxCas9)",
    "PE6d(+PEmaxCas9)": "PE6d(+PEmaxCas9)",
    "PE6e": "PE6e(+dRNaseH)",
    "PE6e(+dRNaseH)": "PE6e(+dRNaseH)",
    "PE6f": "PE6f(+dRNaseH)",
    "PE6f(+dRNaseH)": "PE6f(+dRNaseH)",
    "PE6g": "PE6g(+dRNaseH)",
    "PE6g(+dRNaseH)": "PE6g(+dRNaseH)",
}


def get_pe6_qc_mask(
    data,
    pe_type: str,
    read_count_filter: int = 200,
    max_ctrl_bg: float = 5.0,
):
    """Computes the canonical PE6 quality-control boolean mask for a given PE system.

    Mirrors PE6DeepPrimeDataset filtering:
    1. Total read counts >= read_count_filter for all replicates in AVG_PAIR[pe_type].
    2. Control background editing efficiency <= max_ctrl_bg (5.0%) for matched control batches.
    3. Target efficiency column (if present in data) is non-null.
    """
    import numpy as np
    import pandas as pd

    mask = pd.Series(True, index=data.index)
    if read_count_filter <= 0:
        return mask

    t = PE_TYPE_ALIASES.get(pe_type, pe_type)

    # 1. Total read counts >= read_count_filter for both replicates in AVG_PAIR
    for r in AVG_PAIR.get(t, []):
        if r in data.columns:
            mask = mask & (data[r] >= read_count_filter)

    # 2. Background noise filter <= max_ctrl_bg for the control batches relevant to this PE system
    batches_in_t = set()
    for r in AVG_PAIR.get(t, []):
        if "B2" in r:
            batches_in_t.add("HEK-M-CTRL-7D-B2")
        elif "M2" in r:
            batches_in_t.add("HEK-M2-CTRL-7D-B1")
        elif "B1" in r:
            batches_in_t.add("HEK-M-CTRL-7D-B1")

    for ctrl_prefix in batches_in_t:
        ed_col = f"{ctrl_prefix}+edited_read_counts"
        un_col = f"{ctrl_prefix}+unedited_read_counts"
        if ed_col in data.columns and un_col in data.columns:
            ctrl_tot = data[ed_col] + data[un_col]
            bg = (data[ed_col] / ctrl_tot.replace(0, np.nan) * 100).fillna(0)
            mask = mask & (bg <= max_ctrl_bg)

    if t in data.columns:
        mask = mask & data[t].notna()

    return mask


def exp_transform(x: torch.Tensor) -> torch.Tensor:
    """Applies exponential transformation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transformed tensor after applying exponential transformation.
    """
    return torch.exp(x) - 1

