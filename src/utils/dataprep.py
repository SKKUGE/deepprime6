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
    "indiv_vis+HEK-M-1-7D-KMC%": "PEmax-KMC",
    "indiv_vis+HEK-M-2-7D-KMC%": "PEmaxdRNaseH-KMC",
    "indiv_vis+HEK-M-3-7D-KMC%": "PE6a(+PEmaxCas9)-KMC",
    "indiv_vis+HEK-M-4-7D-KMC%": "PE6b(+PEmaxCas9)-KMC",
    "indiv_vis+HEK-M-5-7D-KMC%": "PE6c(+PEmaxCas9)-KMC",
    "indiv_vis+HEK-M-6-7D-KMC%": "PE6d(+PEmaxCas9)-KMC",
    "indiv_vis+HEK-M-7-7D-KMC%": "PE6e(+dRNaseH)-KMC",
    "indiv_vis+HEK-M-8-7D-KMC%": "PE6f(+dRNaseH)-KMC",
    "indiv_vis+HEK-M-9-7D-KMC%": "PE6g(+dRNaseH)-KMC",
    "indiv_vis+HEK-M-1-7D-UAR%": "PEmax-UAR",
    "indiv_vis+HEK-M-2-7D-UAR%": "PEmaxdRNaseH-UAR",
    "indiv_vis+HEK-M-3-7D-UAR%": "PE6a(+PEmaxCas9)-UAR",
    "indiv_vis+HEK-M-4-7D-UAR%": "PE6b(+PEmaxCas9)-UAR",
    "indiv_vis+HEK-M-5-7D-UAR%": "PE6c(+PEmaxCas9)-UAR",
    "indiv_vis+HEK-M-6-7D-UAR%": "PE6d(+PEmaxCas9)-UAR",
    "indiv_vis+HEK-M-7-7D-UAR%": "PE6e(+dRNaseH)-UAR",
    "indiv_vis+HEK-M-8-7D-UAR%": "PE6f(+dRNaseH)-UAR",
    "indiv_vis+HEK-M-9-7D-UAR%": "PE6g(+dRNaseH)-UAR",
    "indiv_vis+HEK-S-1-14D-KMC%": "PEmax-KMC",
    "indiv_vis+HEK-S-2-14D-KMC%": "PEmaxdRNaseH-KMC",
    "indiv_vis+HEK-S-3-14D-KMC%": "PE6a(+PEmaxCas9)-KMC",
    "indiv_vis+HEK-S-4-14D-KMC%": "PEb(+PEmaxCas9)-KMC",
    "indiv_vis+HEK-S-5-14D-KMC%": "PE6c(+PEmaxCas9)-KMC",
    "indiv_vis+HEK-S-6-14D-KMC%": "PE6d(+PEmaxCas9)-KMC",
    "indiv_vis+HEK-S-10-14D-KMC%": "PE6ec-KMC",
    "indiv_vis+HEK-S-11-14D-KMC%": "PE6fc-KMC",
    "indiv_vis+HEK-S-12-14D-KMC%": "PE6gc-KMC",
    "indiv_vis+HEK-S-13-14D-KMC%": "PE6ed-KMC",
    "indiv_vis+HEK-S-14-14D-KMC%": "PE6fd-KMC",
    "indiv_vis+HEK-S-15-14D-KMC%": "PE6gd-KMC",
    "indiv_vis+HEK-S-1-14D-UAR%": "PEmax-UAR",
    "indiv_vis+HEK-S-2-14D-UAR%": "PEmaxdRNaseH-UAR",
    "indiv_vis+HEK-S-3-14D-UAR%": "PE6a(+PEmaxCas9)-UAR",
    "indiv_vis+HEK-S-4-14D-UAR%": "PEb(+PEmaxCas9)-UAR",
    "indiv_vis+HEK-S-5-14D-UAR%": "PE6c(+PEmaxCas9)-UAR",
    "indiv_vis+HEK-S-6-14D-UAR%": "PE6d(+PEmaxCas9)-UAR",
    "indiv_vis+HEK-S-10-14D-UAR%": "PE6ec-UAR",
    "indiv_vis+HEK-S-11-14D-UAR%": "PE6fc-UAR",
    "indiv_vis+HEK-S-12-14D-UAR%": "PE6gc-UAR",
    "indiv_vis+HEK-S-13-14D-UAR%": "PE6ed-UAR",
    "indiv_vis+HEK-S-14-14D-UAR%": "PE6fd-UAR",
    "indiv_vis+HEK-S-15-14D-UAR%": "PE6gd-UAR",
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
    "comb_vis+HEK-S-4-14D%": "PEb(+PEmaxCas9)",
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
    # Averaged PE 6 and 7 results from SKW
    "PE6-2avg": "PEmaxdRNaseH-avg",
    "PE6-3avg": "PE6b-avg",
    "PE6-4avg": "PE6c-avg",
    "PE7-2avg": "PE7maxdRNaseH-avg",
    "PE7-3avg": "PE7b-avg",
    "PE7-4avg": "PE7c-avg",
}


AVG_PAIR = {
    "PEmax": (
        # "HEK-M-1-7D-KMC+total_read_counts",
        "HEK-M-1-7D-UAR+total_read_counts",
        "HEK-M2-1-7D-UAR+total_read_counts",
    ),
    "PEmaxdRNaseH": (
        # "HEK-M-2-7D-KMC+total_read_counts",
        "HEK-M-2-7D-UAR+total_read_counts",
        "HEK-M2-2-7D-UAR+total_read_counts",
    ),
    "PE6a(+PEmaxCas9)": (
        # "HEK-M-3-7D-KMC+total_read_counts",
        "HEK-M-3-7D-UAR+total_read_counts",
        "HEK-M2-3-7D-UAR+total_read_counts",
    ),
    "PE6b(+PEmaxCas9)": (
        # "HEK-M-4-7D-KMC+total_read_counts",
        "HEK-M-4-7D-UAR+total_read_counts",
        "HEK-M2-4-7D-UAR+total_read_counts",
    ),
    "PE6c(+PEmaxCas9)": (
        # "HEK-M-5-7D-KMC+total_read_counts",
        "HEK-M-5-7D-UAR+total_read_counts",
        "HEK-M2-5-7D-UAR+total_read_counts",
    ),
    "PE6d(+PEmaxCas9)": (
        # "HEK-M-6-7D-KMC+total_read_counts",
        "HEK-M-6-7D-UAR+total_read_counts",
        "HEK-M2-6-7D-UAR+total_read_counts",
    ),
    "PE6e(+dRNaseH)": (
        # "HEK-M-7-7D-KMC+total_read_counts",
        "HEK-M-7-7D-UAR+total_read_counts",
    ),
    "PE6f(+dRNaseH)": (
        # "HEK-M-8-7D-KMC+total_read_counts",
        "HEK-M-8-7D-UAR+total_read_counts",
    ),
    "PE6g(+dRNaseH)": (
        # "HEK-M-9-7D-KMC+total_read_counts",
        "HEK-M-9-7D-UAR+total_read_counts",
    ),
    # "PEmaxdRNaseH-avg": (
    #     "PEmaxdRNaseH-#1",
    #     "PEmaxdRNaseH-#2",
    #     "PEmaxdRNaseH-#3",
    # ),
    # "PE6b-avg": (
    #     "PE6b-#1",
    #     "PE6b-#2",
    #     "PE6b-#3",
    # ),
    # "PE6c-avg": (
    #     "PE6c-#1",
    #     "PE6c-#2",
    #     "PE6c-#3",
    # ),
    # "PE7maxdRNaseH-avg": (
    #     "PE7maxdRNaseH-#1",
    #     "PE7maxdRNaseH-#2",
    #     "PE7maxdRNaseH-#3",
    # ),
    # "PE7b-avg": (
    #     "PE7b-#1",
    #     "PE7b-#2",
    #     "PE7b-#3",
    # ),
    # "PE7c-avg": (
    #     "PE7c-#1",
    #     "PE7c-#2",
    #     "PE7c-#3",
    # ),
}


def exp_transform(x: torch.Tensor) -> torch.Tensor:
    """Applies exponential transformation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transformed tensor after applying exponential transformation.
    """
    return torch.exp(x) - 1
