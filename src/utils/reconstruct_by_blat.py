import os
import sys
import json
import urllib.request
import urllib.parse
import pandas as pd
import time
from Bio.Seq import Seq

# Set project root path
sys.path.append("/home/work/workdir/deepprime6-genomebiol-revision")

def query_ucsc_blat(seq):
    # Add rate-limiting delay
    time.sleep(1.5)
    url = 'https://genome.ucsc.edu/cgi-bin/hgBlat'
    data = urllib.parse.urlencode({
        'db': 'hg38',
        'type': 'DNA',
        'userSeq': seq,
        'output': 'json'
    }).encode('utf-8')
    
    try:
        req = urllib.request.Request(url, data=data, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=15)
        response_data = json.loads(res.read().decode())
        blat_results = response_data.get('blat', [])
        if blat_results:
            best = blat_results[0]
            return {
                'chr': best[13],
                'strand': best[8],
                'matches': best[0],
                'tStart': best[15],
                'tEnd': best[16]
            }
    except Exception as e:
        print(f"Error querying BLAT for {seq}: {e}")
    return None

def fetch_genomic_sequence(chrom, start, stop, cache):
    key = f"{chrom}:{start}-{stop}"
    if key in cache:
        return cache[key]
        
    # Add rate-limiting delay
    time.sleep(1.5)
    url = f'https://genome.ucsc.edu/cgi-bin/das/hg38/dna?segment={chrom}:{start},{stop}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=15)
        xml_data = res.read().decode()
        
        # Simple parsing of DNA element from XML
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_data)
        dna_element = root.find('.//DNA')
        if dna_element is not None and dna_element.text:
            dna_seq = ''.join(dna_element.text.split()).upper()
            cache[key] = dna_seq
            return dna_seq
    except Exception as e:
        print(f"Error fetching DAS {key}: {e}")
    return None

def align_and_reconstruct(row, blat_res, genomic_seq):
    pbs = row['PBS_pegRNA_DNA'].upper()
    rc_pbs = str(Seq(pbs).reverse_complement())
    
    rtt_peg = row['RTT_template_DNA'].upper()
    rtt = str(Seq(rtt_peg).reverse_complement())
    
    fwd_match = False
    fwd_strand = True
    
    rtt_flank = rtt[-12:]
    idx_pbs = genomic_seq.find(rc_pbs)
    idx_flank = genomic_seq.find(rtt_flank)
    
    if idx_pbs != -1 and idx_flank != -1 and idx_flank > idx_pbs:
        fwd_match = True
        fwd_strand = True
    else:
        # Check reverse complement strand of genomic sequence
        rc_genomic = str(Seq(genomic_seq).reverse_complement())
        idx_pbs_rc = rc_genomic.find(rc_pbs)
        idx_flank_rc = rc_genomic.find(rtt_flank)
        if idx_pbs_rc != -1 and idx_flank_rc != -1 and idx_flank_rc > idx_pbs_rc:
            fwd_match = True
            fwd_strand = False
            genomic_seq = rc_genomic
            idx_pbs = idx_pbs_rc
            idx_flank = idx_flank_rc
            
    if not fwd_match:
        return None
        
    nick_idx = idx_pbs + len(rc_pbs)
    wt_rtt_end = idx_flank + len(rtt_flank)
    
    # 1. 200bp flanking sequence for PRIDICT2.0 (centered on nick)
    pr_wt_start = nick_idx - 100
    pr_wt_end = nick_idx + 100
    if pr_wt_start < 0 or pr_wt_end > len(genomic_seq):
        return None
        
    pr_wt_seq = genomic_seq[pr_wt_start:pr_wt_end]
    pr_ed_seq = pr_wt_seq[:100] + rtt + pr_wt_seq[100 + (wt_rtt_end - nick_idx):]
    
    # 2. 74nt Target sequence for DeepPrime (nick at index 21)
    dp_wt_start = nick_idx - 21
    dp_wt_end = nick_idx + 53
    if dp_wt_start < 0 or dp_wt_end > len(genomic_seq):
        return None
        
    dp_wt_seq = genomic_seq[dp_wt_start:dp_wt_end]
    
    # 3. Determine exact edit details by comparing wild-type RTT and mutant RTT
    wt_rtt = genomic_seq[nick_idx : wt_rtt_end]
    
    edit_type = "Sub"
    edit_len = 1
    edit_pos = 1
    
    if len(wt_rtt) == len(rtt):
        edit_type = "Sub"
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
        'chr': blat_res['chr'],
        'strand': '+' if (blat_res['strand'] == '+' if fwd_strand else blat_res['strand'] == '-') else '-'
    }

def main():
    print("Loading pegRNA CSV dataset...")
    df = pd.read_csv("data/MFE_randompeg_RHA30_0.71M_result.csv")
    df['clean_id'] = df['ID'].astype(str).str.replace('clinic_', '').str.strip()
    
    unique_ids = df['clean_id'].unique()
    print(f"Total unique variant IDs: {len(unique_ids)}")
    
    cache_file = "data/genomic_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} sequences from genomic cache.")
    else:
        cache = {}
        
    resolved_details_file = "data/resolved_pegrna_genomic_details.json"
    if os.path.exists(resolved_details_file):
        with open(resolved_details_file) as f:
            resolved_details = json.load(f)
        print(f"Loaded {len(resolved_details)} resolved details from cache.")
    else:
        resolved_details = {}
        
    unresolved_ids = [aid for aid in unique_ids if aid not in resolved_details]
    print(f"Variants left to resolve: {len(unresolved_ids)}")
    
    success = 0
    total_to_resolve = len(unresolved_ids)
    
    for idx, aid in enumerate(unresolved_ids):
        subset = df[df['clean_id'] == aid]
        rep_row = subset.iloc[0]
        
        pbs = rep_row['PBS_pegRNA_DNA'].upper()
        rc_pbs = str(Seq(pbs).reverse_complement())
        rtt_peg = rep_row['RTT_template_DNA'].upper()
        rtt = str(Seq(rtt_peg).reverse_complement())
        active_seq = rc_pbs + rtt
        
        # Query BLAT
        blat_res = query_ucsc_blat(active_seq)
        if not blat_res:
            blat_res = query_ucsc_blat(str(Seq(active_seq).reverse_complement()))
            if not blat_res:
                print(f"[{idx+1}/{total_to_resolve}] ID {aid} - BLAT alignment failed.")
                continue
                
        # Fetch genomic sequence
        chrom = blat_res['chr']
        start_coord = blat_res['tStart'] - 150
        stop_coord = blat_res['tEnd'] + 150
        
        genomic_seq = fetch_genomic_sequence(chrom, start_coord, stop_coord, cache)
        if not genomic_seq:
            print(f"[{idx+1}/{total_to_resolve}] ID {aid} - DAS sequence fetch failed.")
            continue
            
        res = align_and_reconstruct(rep_row, blat_res, genomic_seq)
        if res:
            success += 1
            resolved_details[aid] = res
            print(f"[{idx+1}/{total_to_resolve}] ID {aid} - Successfully resolved.")
        else:
            print(f"[{idx+1}/{total_to_resolve}] ID {aid} - Alignment mapping failed.")
            
        # Incremental save every success
        if res:
            with open(resolved_details_file, "w") as f:
                json.dump(resolved_details, f, indent=2)
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=2)
                
    print(f"Reconstruction finished. Total resolved variants: {len(resolved_details)} / {len(unique_ids)}")

if __name__ == "__main__":
    main()
