import os
import sys
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import time
from Bio.Seq import Seq

def fetch_genomic_sequence(chrom, start, stop, cache):
    key = f"{chrom}:{start}-{stop}"
    if key in cache:
        return cache[key]
        
    url = f'https://genome.ucsc.edu/cgi-bin/das/hg38/dna?segment={chrom}:{start},{stop}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        xml_data = res.read().decode()
        
        root = ET.fromstring(xml_data)
        dna_element = root.find('.//DNA')
        if dna_element is not None and dna_element.text:
            dna_seq = ''.join(dna_element.text.split()).upper()
            cache[key] = dna_seq
            return dna_seq
    except Exception as e:
        print(f"Error fetching {key}: {e}")
    return None

def align_and_reconstruct(row, meta, genomic_seq):
    # rc_PBS
    pbs = row['PBS_pegRNA_DNA'].upper()
    rc_pbs = str(Seq(pbs).reverse_complement())
    
    # RTT on the genomic strand (which is the reverse complement of the pegRNA RTT template)
    rtt_peg = row['RTT_template_DNA'].upper()
    rtt = str(Seq(rtt_peg).reverse_complement())
    
    # Coordinates of the variant
    v_start = meta['start']
    
    # We find where active_seq matches genomic_seq.
    # Since active_seq contains the mutation, we match by dividing it into two wild-type flanking parts:
    # 5' flank = rc_pbs (ends at nick site)
    # 3' flank = last 12nt of rtt
    
    # Check forward strand
    fwd_match = False
    fwd_strand = True
    
    # Search for rc_pbs
    idx_pbs = genomic_seq.find(rc_pbs)
    # Search for the 3' flank of RTT (length >= 10)
    rtt_flank = rtt[-12:]
    idx_flank = genomic_seq.find(rtt_flank)
    
    if idx_pbs != -1 and idx_flank != -1 and idx_flank > idx_pbs:
        fwd_match = True
        fwd_strand = True
    else:
        # Check reverse complement strand
        rc_genomic = str(Seq(genomic_seq).reverse_complement())
        idx_pbs_rc = rc_genomic.find(rc_pbs)
        idx_flank_rc = rc_genomic.find(rtt_flank)
        if idx_pbs_rc != -1 and idx_flank_rc != -1 and idx_flank_rc > idx_pbs_rc:
            fwd_match = True
            fwd_strand = False
            genomic_seq = rc_genomic # use reverse complement genomic sequence
            idx_pbs = idx_pbs_rc
            idx_flank = idx_flank_rc
            
    if not fwd_match:
        return None # Could not align uniquely
        
    # Reconstruct coordinate details on the matched genomic strand
    # Nick site index in genomic_seq is idx_pbs + len(rc_pbs)
    nick_idx = idx_pbs + len(rc_pbs)
    
    # 1. 200bp flanking sequence for PRIDICT2.0
    # 100bp upstream and 100bp downstream of the nick site
    pr_wt_start = nick_idx - 100
    pr_wt_end = nick_idx + 100
    if pr_wt_start < 0 or pr_wt_end > len(genomic_seq):
        return None
        
    pr_wt_seq = genomic_seq[pr_wt_start:pr_wt_end]
    
    # Reconstruct the edited 200bp sequence for PRIDICT2.0:
    # We replace the RTT portion in the wild-type sequence with the RTT sequence from the pegRNA
    # Wild-type RTT starts at nick_idx. What is its length?
    # Since rtt_flank matched at idx_flank, the wild-type RTT ends at idx_flank + len(rtt_flank)
    wt_rtt_end = idx_flank + len(rtt_flank)
    
    pr_ed_seq = pr_wt_seq[:100] + rtt + pr_wt_seq[100 + (wt_rtt_end - nick_idx):]
    
    # 2. 74nt Target sequence for DeepPrime:
    # Nick index in the 74nt target sequence must be exactly 21 (so 21 bases upstream of nick, 53 bases downstream)
    dp_wt_start = nick_idx - 21
    dp_wt_end = nick_idx + 53
    if dp_wt_start < 0 or dp_wt_end > len(genomic_seq):
        return None
        
    dp_wt_seq = genomic_seq[dp_wt_start:dp_wt_end]
    
    # Determine exact edit details by comparing wild-type RTT and mutant RTT
    wt_rtt = genomic_seq[nick_idx : wt_rtt_end]
    
    # Compare wt_rtt and rtt to find the edit
    # E.g. Substitution, Insertion, Deletion
    edit_type = "Sub"
    edit_len = 1
    edit_pos = 1
    
    # Check if they have the same length (Substitution)
    if len(wt_rtt) == len(rtt):
        edit_type = "Sub"
        # Find first difference position
        diffs = [i for i in range(len(rtt)) if rtt[i] != wt_rtt[i]]
        if diffs:
            edit_pos = diffs[0] + 1
            edit_len = len(diffs)
        else:
            edit_pos = 1
            edit_len = 0
    elif len(rtt) > len(wt_rtt):
        edit_type = "Ins"
        edit_len = len(rtt) - len(wt_rtt)
        # Find insertion position
        min_len = min(len(rtt), len(wt_rtt))
        diverge = min_len
        for i in range(min_len):
            if rtt[i] != wt_rtt[i]:
                diverge = i
                break
        edit_pos = diverge + 1
    else:
        edit_type = "Del"
        edit_len = len(wt_rtt) - len(rtt)
        # Find deletion position
        min_len = min(len(rtt), len(wt_rtt))
        diverge = min_len
        for i in range(min_len):
            if rtt[i] != wt_rtt[i]:
                diverge = i
                break
        edit_pos = diverge + 1
        
    return {
        'wt_target_74': dp_wt_seq,
        'wt_pridict_200': pr_wt_seq,
        'ed_pridict_200': pr_ed_seq,
        'edit_type': edit_type,
        'edit_len': edit_len,
        'edit_pos': edit_pos,
        'gene': meta['gene'],
        'strand': '+' if fwd_strand else '-'
    }

def main():
    print("Loading resolved ClinVar mapping...")
    with open("data/clinvar_hgvs_mapping.json") as f:
        mapping = json.load(f)
        
    print("Loading pegRNA CSV dataset...")
    df = pd.read_csv("data/MFE_randompeg_RHA30_0.71M_result.csv")
    
    # Load or initialize genomic sequence cache
    cache_file = "data/genomic_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} sequences from genomic cache.")
    else:
        cache = {}
        
    resolved_details = {}
    
    # Process each unique resolved variant
    total_resolved = len(mapping)
    processed = 0
    success = 0
    
    t0 = time.time()
    for aid, meta in mapping.items():
        processed += 1
        chrom = meta['chr']
        v_start = meta['start']
        
        if not chrom or not v_start:
            continue
            
        # Retrieve genomic sequence centered at variant
        # Fetch 150bp flanking on each side
        start_coord = v_start - 150
        stop_coord = v_start + 150
        
        genomic_seq = fetch_genomic_sequence(chrom, start_coord, stop_coord, cache)
        if not genomic_seq:
            continue
            
        # Find a representative pegRNA row for this ID in the dataset
        subset = df[df['ID'].astype(str).str.replace('clinic_', '').str.strip() == str(aid)]
        if subset.empty:
            continue
            
        rep_row = subset.iloc[0]
        
        # Align and reconstruct sequences
        res = align_and_reconstruct(rep_row, meta, genomic_seq)
        if res:
            success += 1
            resolved_details[aid] = res
            
        if processed % 50 == 0:
            print(f"Processed {processed}/{total_resolved} variants. Success rate: {success}/{processed} ({success/processed*100:.1f}%)")
            # Save cache incrementally
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=2)
                
    # Final save
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)
        
    print(f"Successfully aligned and reconstructed genomic sequences for {success}/{total_resolved} variants.")
    
    with open("data/resolved_pegrna_genomic_details.json", "w") as f:
        json.dump(resolved_details, f, indent=2)
    print("Saved resolved pegRNA genomic details to data/resolved_pegrna_genomic_details.json")
    print(f"Total time elapsed: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
