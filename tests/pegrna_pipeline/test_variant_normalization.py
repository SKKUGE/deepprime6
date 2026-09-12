import pytest
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("normalize_variants", "src/pegrna_pipeline/01_normalize_variants.py")
normalize_variants = importlib.util.module_from_spec(spec)
sys.modules["normalize_variants"] = normalize_variants
spec.loader.exec_module(normalize_variants)

parse_ref_alt = normalize_variants.parse_ref_alt
get_edit_type_and_length = normalize_variants.get_edit_type_and_length

def test_parse_ref_alt():
    # Substitution
    ref, alt = parse_ref_alt("ATCG", "ATTG")
    assert ref == "C"
    assert alt == "T"
    
    # Insertion
    ref, alt = parse_ref_alt("ATG", "ATCTG")
    assert ref == "G"
    assert alt == "CTG"
    
    # Deletion
    ref, alt = parse_ref_alt("ATCTG", "ATG")
    assert ref == "CTG"
    assert alt == "G"

def test_get_edit_type_and_length():
    # SNV
    t, l = get_edit_type_and_length("A", "T")
    assert t == "Sub"
    assert l == 1
    
    # Insertion
    t, l = get_edit_type_and_length("A", "ACT")
    assert t == "Ins"
    assert l == 3
    
    # Deletion
    t, l = get_edit_type_and_length("ACT", "A")
    assert t == "Del"
    assert l == 3
