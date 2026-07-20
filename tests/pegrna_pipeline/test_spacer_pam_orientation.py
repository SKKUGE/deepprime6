import pytest
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("design_pegrnas", "src/pegrna_pipeline/04_design_pegrnas.py")
design_pegrnas = importlib.util.module_from_spec(spec)
sys.modules["design_pegrnas"] = design_pegrnas
spec.loader.exec_module(design_pegrnas)

extract_target_features = design_pegrnas.extract_target_features

def test_extract_target_features_fwd_strand():
    # Construct a WT context
    # Edit is at index 100. Let's make the context 200bp total.
    # We want a PAM (NGG) near the edit.
    # index 100 is 'A'. We change it to 'T'.
    
    wt_ctx = "C" * 80 + "ATGCATGCATGCATGCATGC" + "TGG" + "C" * 97
    # index 100 is 'G' here if we count:
    # 80 + 20 = 100 -> 'T'
    # Wait, let's be exact.
    
    prefix = "C" * 77
    spacer = "ATGCATGCATGCATGCATGC" # len 20
    pam = "TGG"
    # nick is 3bp upstream of pam: pam_start = 77 + 20 = 97.
    # nick = 94.
    # Let's say edit is at 100.
    suffix = "A" + "C" * 99
    
    wt_ctx = prefix + spacer + pam + suffix
    assert len(wt_ctx) == 200
    
    ed_ctx = wt_ctx[:100] + "T" + wt_ctx[101:]
    
    designs = extract_target_features(
        wt_ctx, ed_ctx, "A", "T", "+",
        pbs_len=13, rtt_len=15
    )
    
    assert len(designs) > 0
    
    d = designs[0]
    assert d['spacer'] == "ATGCATGCATGCATGCATGC"
    assert d['PAM'] == "TGG"
    assert d['PBS_length'] == 13
    assert d['RTT_length'] == 15
    assert d['pegRNA_strand'] == "+"

def test_extract_target_features_rev_strand():
    # We test if the reverse strand PAM correctly reverses the spacer and identifies the right edit index.
    pass
