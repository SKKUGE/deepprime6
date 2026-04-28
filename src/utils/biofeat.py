import gc
from itertools import product

import pandas as pd

NUCLEOTIDES = ["A", "C", "G", "T"]


def _annotating_ref_id(row, feature_annotation):
    feature_annotation.update({"gene_name": row["gene_name"]})
    feature_annotation.update({"alias": row["alias"]})


def _annotating_guide_target_sequence_info(
    row, label, guide_sequence, read_count=None, feature_annotation=None
):
    if feature_annotation is None:
        feature_annotation = {}
    feature_annotation.update({"Wide target sequence": row["Wide target sequence"]})
    feature_annotation.update({"guide_sequence": guide_sequence})
    feature_annotation.update({"label": label})

    if read_count is not None:  # In case of endogenous dataset
        feature_annotation.update({"read_count": read_count})


def _calculating_mfe(guide_sequence, feature_annotation):
    import RNA

    feature_annotation.update({"MFE": RNA.fold(guide_sequence)[1]})


def _calculating_gc_counts(target_sequence, feature_annotation):
    feature_annotation.update(
        {"GC_count": target_sequence.count("G") + target_sequence.count("C")}
    )
    feature_annotation.update({"GC_count_lte_9": int(feature_annotation["GC_count"] <= 9)})
    feature_annotation.update({"GC_count_gt_9": int(feature_annotation["GC_count"] > 9)})


def _calculating_tm_wallace(guide_sequence, feature_annotation):
    from Bio.SeqUtils import MeltingTemp as Tm

    feature_annotation.update({"Tm": Tm.Tm_Wallace(guide_sequence)})


def _extract_pos_di_nucleotides(target_sequence, feature_annotation):
    for pos in range(1, len(target_sequence)):  # Counting from 1, two nucleotides at a time
        for nucleotide in ["".join(comb) for comb in product(NUCLEOTIDES, NUCLEOTIDES)]:
            feature_annotation.update(
                {f"pos_{pos}_{nucleotide}": int(target_sequence[pos - 1 : pos + 1] == nucleotide)}
            )


def _extract_pos_mono_nucleotides(target_sequence, feature_annotation):
    for pos in range(1, len(target_sequence) + 1):  # Counting from position 1
        for nucleotide in NUCLEOTIDES:
            feature_annotation.update(
                {f"pos_{pos}_{nucleotide}": int(target_sequence[pos - 1] == nucleotide)}
            )


def _extract_di_nucleotides(target_sequence, feature_annotation):
    for nucleotide in ["".join(comb) for comb in product(NUCLEOTIDES, NUCLEOTIDES)]:
        feature_annotation.update(
            {f"pos_independent_{nucleotide}": target_sequence.count(nucleotide)}
        )


def _extract_mono_nucleotides(target_sequence, feature_annotation):
    for nucleotide in NUCLEOTIDES:
        feature_annotation.update(
            {f"pos_independent_{nucleotide}": target_sequence.count(nucleotide)}
        )


def _feature_extraction(
    row,
    wide_target_sequence_col: str = "Wide target sequence",
    y_label_col: str = "label",
    read_count_col: str = "read_count",
    pbs_col: str = "PBS",
    rtt_col: str = "RTT",
    pam_length=3,
    upstream_flanking_length: int = 4,
    downstream_flanking_length: int = 4,
):
    """
    a total of 2,956 features were used for training of conventional machine learning models
    - Development of conventional machine learning-based models

    Extracts features from the target sequence and returns a dictionary containing the features.

    Args:
        row (pandas.Series): The row containing the target sequence and label.
        pam_length (int, optional): The length of the target adjacent motif (PAM). Defaults to 3 (NGG).
        upstream_flanking_length (int, optional): The number of nucleotides to include upstream of the PAM. Defaults to 4.
        downstream_flanking_length (int, optional): The number of nucleotides to include downstream of the PAM. Defaults to 4.

    Returns:
        dict: A dictionary containing the extracted features.
    """

    target_sequence = row[wide_target_sequence_col]
    label = row[y_label_col]
    read_count = row[read_count_col] if read_count_col in row.index else None
    target_sequence = target_sequence.upper()
    guide_sequence = target_sequence[upstream_flanking_length:-downstream_flanking_length][
        : pam_length * -1
    ]
    feature_annotation = {}  # dict to store all the features

    pbs = row[pbs_col] if pbs_col in row.index else ""
    rtt = row[rtt_col] if rtt_col in row.index else ""

    # From the Wide Target sequences
    # (i) Position-independent mono-nucleotides and di-nucleotides
    # Position independent Mono nucleotides (4 features = A  C  G T)
    _extract_mono_nucleotides(target_sequence, feature_annotation)
    # Position independent Di nucleotides (16 features = AA AG AC AT  GA GG GC GT  CA CG CC CT  TA TG TC TT)
    _extract_di_nucleotides(target_sequence, feature_annotation)
    # (ii) Position-dependent nucleotides and dinucleotides
    # Position dependent Mono nucleotides (4 * 74 = 296 features)
    _extract_pos_mono_nucleotides(target_sequence, feature_annotation)
    # Position dependent Di nucleotides (4^2 * 73 = 1168 features)
    _extract_pos_di_nucleotides(target_sequence, feature_annotation)

    # From the PBS sequence: position independent and dependent mono and di nucleotides

    # From the RTT sequence: position independent and dependent mono and di nucleotides

    # Z-normalized features: should be calculated out of this function
    # (iii) Tm  (1)
    _calculating_tm_wallace(guide_sequence, feature_annotation)
    # (iv) GC count (3)
    _calculating_gc_counts(target_sequence, feature_annotation)
    # (v) Free energy (1)
    _calculating_mfe(guide_sequence, feature_annotation)

    # 20 + 644 + 1 + 3 + 1 = 669 features
    _annotating_guide_target_sequence_info(
        row, label, guide_sequence, read_count, feature_annotation
    )

    # unique id appendation
    _annotating_ref_id(row, feature_annotation)

    return feature_annotation


def feature_transform_dataframe(
    df, pam_length, upstream_flanking_length, downstream_flanking_length
):
    """Transforms a dataframe by applying a feature extraction function to each row.

    Args:
        df (pd.DataFrame): The input dataframe.
        pam_length (int): The length of the target site.
        upstream_flanking (int): The length of the upstream flanking region.
        downstream_flanking (int): The length of the downstream flanking region.

    Returns:
        pd.DataFrame: The transformed dataframe.
    """
    storage = []
    for idx, r in df.iterrows():
        storage.append(
            _feature_extraction(
                r, pam_length, upstream_flanking_length, downstream_flanking_length
            )
        )
    transformed = pd.DataFrame(storage)

    del storage
    gc.collect()
    return transformed
