import importlib.util
import sys

spec = importlib.util.spec_from_file_location("build_contexts", "src/pegrna_pipeline/02_build_contexts_and_designability.py")
build_contexts = importlib.util.module_from_spec(spec)
sys.modules["build_contexts"] = build_contexts
spec.loader.exec_module(build_contexts)

reverse_complement = build_contexts.reverse_complement

find_pams = build_contexts.find_pams
assess_designability = build_contexts.assess_designability

def test_find_pams():
    # Sequence: AGGCTGG
    # Index:    0123456
    # GG at 1,2 -> PAM at 0
    # GG at 5,6 -> PAM at 4
    pams = find_pams("AGGCTGG")
    assert pams == [0, 4]

def test_assess_designability():
    config = {
        'design_parameters': {
            'pam': 'NGG',
            'rtt_length_max': 30
        }
    }
    
    # 200bp sequence, edit at 100
    # Create a 200bp sequence with NGG pam near the middle
    sequence = "N" * 90 + "GGCCA" + "A" + "CCAAGGG" + "N" * 90
    # Let's ensure length is 200: 90 + 5 + 1 + 7 + 90 = 193? 
    # Let's make it exactly 200:
    sequence = "N" * 95 + "CC" + "A" + "CCAAGGG" + "N" * 95 # 95 + 2 + 1 + 7 + 95 = 200
    # Edit is at 97 (REF: A, ALT: T)
    
    # NGG PAM on forward strand: GGG at index 98-100? No, let's just make it simple:
    # wt_fwd has GG at some position
    # Let's mock a sequence where assess_designability returns True
    wt_fwd = "N" * 95 + "GG" + "A" + "CC" + "N" * 100 # len = 200
    # Edit at 97 (A). GG is at 95,96. nick at 92. dist_to_edit = 97 - 92 = 5 (within 30)
    has_pam = assess_designability(wt_fwd, "A", "T", config)
    assert has_pam is True
