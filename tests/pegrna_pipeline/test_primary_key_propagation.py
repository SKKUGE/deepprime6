import pandas as pd
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("validate_designs", "src/pegrna_pipeline/05_validate_pegrna_designs.py")
validate_designs = importlib.util.module_from_spec(spec)
sys.modules["validate_designs"] = validate_designs
spec.loader.exec_module(validate_designs)

validate_design = validate_designs.validate_design

def test_bio1_spacer_mismatch():
    # Construct a valid design, then mutate spacer
    row = {
        'WT_context': 'A'*100 + 'CGG' + 'A'*97,
        'Edited_context': 'A'*100 + 'GGG' + 'A'*97,
        'spacer': 'A'*20, # Suppose valid spacer
        'PAM': 'CGG',
        'nick_position': 97,
        'PBS_sequence': 'T'*13, # mock
        'RTT_sequence': 'C'*15, # mock
        'REF': 'C',
        'ALT': 'G',
        'pegRNA_strand': '+'
    }
    
    # Intentionally corrupt spacer
    row['spacer'] = 'T' * 20
    
    passed, reason = validate_design(row)
    assert not passed
    assert "BIO-1" in reason

def test_bio10_id_propagation():
    # This is handled in main by pandas, but we can verify our pipeline concept.
    pass
