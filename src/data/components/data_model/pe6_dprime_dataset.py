from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.dataprep import AVG_PAIR


class PE6DeepPrimeDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        datafilter: Dict[str, List[str]],
        read_count_filter: int = 0,
        norm_mean: pd.Series = None,
        norm_std: pd.Series = None,
    ):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        assert (
            len(datafilter["PE_types"]) < 2
        ), "The current implementation only supports one PE type"
        t = datafilter["PE_types"][0]
        filtering_results = []
        for r in AVG_PAIR.get(t, []):
            if r in data.columns:
                filtering_results.append(data[r] > read_count_filter)

        # It returns False if all replicates are 0.
        # from functools import reduce
        # filtered_data = data.loc[reduce(np.logical_or, filtering_results)]
        filtered_data = data  # AB Test: No filtering if all replicates are 0

        # Labels
        self._label_col: List[str] = datafilter["PE_types"]
        self.labels = torch.from_numpy(
            filtered_data[
                [*self._label_col, "type_sub", "type_ins", "type_del"]
            ].values  # TODO: Move feature tensor into separate object
        )

        # TODO: Filter specific PE types
        self.data = filtered_data.drop(columns=self._label_col)

        # Annotations for
        self.annotations = filtered_data["ID"]

        # One-hot encoding for sequence data
        self.data["Target_encoded"] = self.data["Target"].apply(
            lambda x: self.preprocess_sequence_data(x)
        )
        self.data["Masked_EditSeq_encoded"] = self.data["Masked_EditSeq"].apply(
            lambda x: self.preprocess_sequence_data(x)
        )
        self.genetic_features = np.stack(
            self.data[["Target_encoded", "Masked_EditSeq_encoded"]]
            .apply(
                lambda x: np.stack([x["Target_encoded"], x["Masked_EditSeq_encoded"]], axis=0),
                axis=1,  # Writing verbose code for readability
            )
            .to_numpy(),
            axis=0,
        )

        # Input data
        self.genetic_features = (
            2 * self.genetic_features - 1
        )  # Normalize tnesor value range to [-1, 1]

        self.biofeatures = self.preprocess_biofeatures(
            self.data[
                [
                    "PBS_len",
                    "RTT_len",
                    "RT-PBS_len",
                    "Edit_pos",
                    "Edit_len",
                    "RHA_len",
                    "type_sub",
                    "type_ins",
                    "type_del",
                    "Tm1_PBS",
                    "Tm2_RTT_cTarget_sameLength",
                    "Tm3_RTT_cTarget_replaced",
                    "Tm4_cDNA_PAM-oppositeTarget",
                    "Tm5_RTT_cDNA",
                    "deltaTm_Tm4-Tm2",
                    "GC_count_PBS",
                    "GC_count_RTT",
                    "GC_count_RT-PBS",
                    "GC_contents_PBS",
                    "GC_contents_RTT",
                    "GC_contents_RT-PBS",
                    "MFE_RT-PBS-polyT",
                    "MFE_Spacer",
                    "DeepSpCas9_score",
                ]
            ]
        )

        # Release the full DataFrame; all needed data has been extracted into
        # self.genetic_features, self.biofeatures, self.labels, and self.annotations.
        self._len = len(self.data)
        del self.data

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        #  Tensor preparation
        g = torch.from_numpy(self.genetic_features[idx].copy())
        g = g.permute(2, 0, 1)  # Rearrange the tensor to (batches, 4, len(sequence): H, 2: W)
        b = torch.from_numpy(self.biofeatures[idx].copy())

        label = self.labels[idx].clone().detach()
        annot = self.annotations.iloc[idx]

        assert g is not None and b is not None and label is not None, "Data is missing!"

        return ((g, b), label), annot

    @property
    def label_col(self):
        return self._label_col

    def preprocess_sequence_data(self, sequence: str):
        """Preprocesses a masked DNA sequence by converting it into a one-hot encoded tensor.

        Args:
            sequence (str): The DNA sequence to preprocess.

        Returns:
            torch.Tensor: The preprocessed sequence as a one-hot encoded 2D tensor.
                The shape of the tensor is (len(sequence), 4), where:
                - len(sequence) is the length of the input sequence
                - 4 is the number of nucleotide bases (A, C, G, T)
        """
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "X": 4, "N": 4}
        map_seq = [mapping[i] for i in sequence.upper()]
        arr_seq = np.eye(5)[map_seq]
        return np.delete(arr_seq, -1, axis=1)

    def ohe_pe_types(self, pe_type: str) -> int:
        """One-hot encodes the given PE type.

        Args:
            pe_type (str): The PE type to be one-hot encoded.

        Returns:
            torch.Tensor: The one-hot encoded tensor representing the PE type.
        """

        pe_dict = {
            "PEmax": 0,
            "PEmaxdRNaseH": 1,
            "PE6a(+PEmaxCas9)": 2,
            "PE6b(+PEmaxCas9)": 3,
            "PE6c(+PEmaxCas9)": 4,
            "PE6d(+PEmaxCas9)": 5,
            "PE6e(+dRNaseH)": 6,
            "PE6f(+dRNaseH)": 7,
            "PE6g(+dRNaseH)": 8,
        }
        return pe_dict[pe_type]

    def preprocess_biofeatures(self, biofeatures: pd.DataFrame) -> torch.Tensor:
        """Preprocesses the given biofeatures by converting them into a tensor.

        Args:
            biofeatures (pd.DataFrame): The biofeatures to preprocess.

        Returns:
            torch.Tensor: The preprocessed biofeatures as a tensor.
        """
        if self.norm_mean is not None:
            # Check if norm_mean is a scalar or single-value Series
            # Legacy
            is_scalar = False
            if isinstance(self.norm_mean, (float, int)):
                is_scalar = True
            elif isinstance(self.norm_mean, pd.Series) and len(self.norm_mean) == 1:
                # If it's a single value series, treat as scalar for broadcasting
                self.norm_mean = self.norm_mean.iloc[0]
                if self.norm_std is not None and isinstance(self.norm_std, pd.Series):
                    self.norm_std = self.norm_std.iloc[0]
                is_scalar = True

            if not is_scalar:
                # Rename columns to match DeepPrime original feature names
                # Derived from manual mapping based on feature definitions
                feature_rename_map = {
                    "PBS_len": "PBSlen",
                    "RTT_len": "RTlen",
                    "RT-PBS_len": "RT-PBSlen",
                    "Tm1_PBS": "Tm1",
                    "Tm2_RTT_cTarget_sameLength": "Tm2",
                    "Tm3_RTT_cTarget_replaced": "Tm2new",
                    "Tm5_RTT_cDNA": "Tm3",
                    "Tm4_cDNA_PAM-oppositeTarget": "Tm4",
                    "deltaTm_Tm4-Tm2": "TmD",
                    "GC_count_PBS": "nGCcnt1",
                    "GC_count_RTT": "nGCcnt2",
                    "GC_count_RT-PBS": "nGCcnt3",
                    "GC_contents_PBS": "fGCcont1",
                    "GC_contents_RTT": "fGCcont2",
                    "GC_contents_RT-PBS": "fGCcont3",
                    "MFE_RT-PBS-polyT": "MFE3",
                    "MFE_Spacer": "MFE4",
                }
                biofeatures = biofeatures.rename(columns=feature_rename_map)

                # Filter and reorder biofeatures to match norm_mean's index (feature list)
                # Only if norm_mean has an index (is not scalar)
                if isinstance(self.norm_mean, (pd.Series, pd.DataFrame)):
                    biofeatures = biofeatures[self.norm_mean.index]

        norm_mean = self.norm_mean if self.norm_mean is not None else biofeatures.mean()
        norm_std = self.norm_std if self.norm_std is not None else biofeatures.std()

        norm_biofeatures = (biofeatures - norm_mean) / norm_std
        norm_biofeatures = norm_biofeatures.fillna(0)  # Divide by zero fix

        return norm_biofeatures.values
