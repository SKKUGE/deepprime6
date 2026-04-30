"""Tests for dataset preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.components.data_model.pe6_dprime_dataset import PE6DeepPrimeDataset
from src.data.components.data_model.collate_func import custom_collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEQ_LEN = 74
BIOFEATURE_COLS = [
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


def _random_dna(length: int) -> str:
    return "".join(np.random.choice(list("ACGT"), size=length))


def _make_dataframe(n: int = 10, pe_type: str = "PEmaxdRNaseH") -> pd.DataFrame:
    """Create a minimal synthetic DataFrame compatible with PE6DeepPrimeDataset."""
    rng = np.random.default_rng(seed=0)
    rows = []
    for i in range(n):
        row = {
            "ID": f"sample_{i:04d}",
            "Target": _random_dna(SEQ_LEN),
            "Masked_EditSeq": _random_dna(SEQ_LEN),
            pe_type: rng.uniform(0, 20),
            "type_sub": int(i % 3 == 0),
            "type_ins": int(i % 3 == 1),
            "type_del": int(i % 3 == 2),
        }
        for col in BIOFEATURE_COLS:
            if col not in row:
                row[col] = rng.uniform(0, 10)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Ensure integer types match expected behaviour
    df["type_sub"] = df["type_sub"].astype(float)
    df["type_ins"] = df["type_ins"].astype(float)
    df["type_del"] = df["type_del"].astype(float)
    return df


def _make_dataset(n: int = 10, pe_type: str = "PEmaxdRNaseH") -> PE6DeepPrimeDataset:
    df = _make_dataframe(n=n, pe_type=pe_type)
    datafilter = {"PE_types": [pe_type]}
    return PE6DeepPrimeDataset(df, datafilter)


# ---------------------------------------------------------------------------
# Sequence preprocessing (one-hot encoding)
# ---------------------------------------------------------------------------


class TestSequencePreprocessing:
    def test_output_shape(self):
        dataset = _make_dataset()
        seq = "ACGT" * (SEQ_LEN // 4)
        arr = dataset.preprocess_sequence_data(seq)
        assert arr.shape == (len(seq), 4)

    def test_one_hot_values(self):
        dataset = _make_dataset()
        arr = dataset.preprocess_sequence_data("ACGT")
        # Each row must be a valid one-hot vector
        assert ((arr == 0) | (arr == 1)).all()
        assert (arr.sum(axis=1) == 1).all()

    def test_mask_character(self):
        dataset = _make_dataset()
        arr = dataset.preprocess_sequence_data("X")
        # 'X' should map to a zero vector (all bases masked out)
        assert (arr == 0).all()

    def test_nucleotide_ordering(self):
        dataset = _make_dataset()
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        for base, idx in mapping.items():
            arr = dataset.preprocess_sequence_data(base)
            assert arr[0, idx] == 1
            assert arr[0].sum() == 1


# ---------------------------------------------------------------------------
# Dataset __len__ and __getitem__
# ---------------------------------------------------------------------------


class TestPE6DeepPrimeDataset:
    def test_length(self):
        n = 12
        dataset = _make_dataset(n=n)
        assert len(dataset) == n

    def test_item_structure(self):
        dataset = _make_dataset()
        item, annot = dataset[0]
        (g, b), label = item
        assert isinstance(g, torch.Tensor)
        assert isinstance(b, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_genetic_feature_shape(self):
        dataset = _make_dataset()
        item, _ = dataset[0]
        (g, _), _ = item
        # g should be (4, SEQ_LEN, 2) after permute
        assert g.shape == (4, SEQ_LEN, 2)

    def test_genetic_feature_range(self):
        dataset = _make_dataset()
        item, _ = dataset[0]
        (g, _), _ = item
        # Values are in [-1, 1] after normalisation 2*x - 1
        assert g.min().item() >= -1.0 - 1e-6
        assert g.max().item() <= 1.0 + 1e-6

    def test_biofeature_shape(self):
        dataset = _make_dataset()
        item, _ = dataset[0]
        (_, b), _ = item
        assert b.shape == (len(BIOFEATURE_COLS),)

    def test_label_dtype(self):
        dataset = _make_dataset()
        item, _ = dataset[0]
        (_, _), label = item
        assert label.dtype in (torch.float32, torch.float64)

    def test_normalization_with_provided_stats(self):
        df = _make_dataframe(n=20)
        pe_type = "PEmaxdRNaseH"
        datafilter = {"PE_types": [pe_type]}
        ds1 = PE6DeepPrimeDataset(df, datafilter)
        norm_mean = ds1.norm_mean
        norm_std = ds1.norm_std

        # Create a second dataset using the stats from the first
        ds2 = PE6DeepPrimeDataset(df, datafilter, norm_mean=norm_mean, norm_std=norm_std)
        assert len(ds2) == len(ds1)

    def test_read_count_filter_keeps_all_rows(self):
        """With filter=0 all rows should be retained (no filtering)."""
        n = 15
        dataset = _make_dataset(n=n)
        assert len(dataset) == n


# ---------------------------------------------------------------------------
# custom_collate_fn
# ---------------------------------------------------------------------------


class TestCustomCollateFn:
    def test_collate_batch_size(self):
        dataset = _make_dataset(n=6)
        items = [dataset[i] for i in range(4)]
        inputs, annotations = custom_collate_fn(items)
        (g_batch, b_batch), label_batch = inputs
        assert g_batch.shape[0] == 4

    def test_collate_annotation_count(self):
        dataset = _make_dataset(n=6)
        items = [dataset[i] for i in range(3)]
        inputs, annotations = custom_collate_fn(items)
        assert len(annotations) == 3

    def test_collate_tensor_shapes(self):
        n_items = 5
        dataset = _make_dataset(n=10)
        items = [dataset[i] for i in range(n_items)]
        inputs, _ = custom_collate_fn(items)
        (g_batch, b_batch), label_batch = inputs
        assert g_batch.shape == (n_items, 4, SEQ_LEN, 2)
        assert b_batch.shape == (n_items, len(BIOFEATURE_COLS))
