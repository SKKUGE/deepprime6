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
    # Construct a WT context with a PAM on the reverse strand (i.e. CCN on forward strand)
    # Target length 200bp, edit at index 100 ('A' -> 'T').
    from Bio.Seq import reverse_complement
    import pandas as pd
    
    # Target spacer on reverse complement
    rc_spacer = "GCTAGCTAGCTAGCTAGCTA"
    rc_pam = "AGG"
    
    # Construct reverse complement target context first
    prefix_rc = "A" * 77
    suffix_rc = "T" + "A" * 99  # edit at index 100 in rc
    rc_wt_ctx = prefix_rc + rc_spacer + rc_pam + suffix_rc
    assert len(rc_wt_ctx) == 200
    
    # Forward strand is RC of rc_wt_ctx
    wt_ctx = reverse_complement(rc_wt_ctx)
    ed_ctx = wt_ctx[:100] + "T" + wt_ctx[101:]
    
    designs = extract_target_features(
        wt_ctx, ed_ctx, "A", "T", "-",
        pbs_len=13, rtt_len=15
    )
    
    rev_designs = [d for d in designs if d['pegRNA_strand'] == '-']
    assert len(rev_designs) > 0
    d = rev_designs[0]
    assert d['spacer'] == rc_spacer
    assert d['PAM'] == rc_pam
    assert d['pegRNA_strand'] == "-"
    
    # Now verify map_to_legacy_features produces correct reverse-complemented WideTargetSequence
    score_spec = importlib.util.spec_from_file_location("score_deepprime", "src/pegrna_pipeline/06_score_deepprime.py")
    score_mod = importlib.util.module_from_spec(score_spec)
    score_spec.loader.exec_module(score_mod)
    
    row = dict(d)
    row.update({'WT_context': wt_ctx, 'REF': 'A', 'ALT': 'T'})
    ldf = score_mod.map_to_legacy_features(pd.DataFrame([row]))
    
    # WideTargetSequence must be reverse complement of WT_context for negative strand
    assert ldf['WideTargetSequence'].iloc[0] == reverse_complement(wt_ctx)
    # The spacer (Guide) must be present in the WideTargetSequence
    assert d['spacer'] in ldf['WideTargetSequence'].iloc[0]

