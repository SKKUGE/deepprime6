import pandas as pd
import json
import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
from Bio.Seq import Seq

def query_ucsc_blat(seq):
    url = 'https://genome.ucsc.edu/cgi-bin/hgBlat'
    data = urllib.parse.urlencode({
        'db': 'hg38',
        'type': 'DNA',
        'userSeq': seq,
        'output': 'json'
    }).encode('utf-8')
    
    for attempt in range(3):
        time.sleep(3.0 * (attempt + 1))
        try:
            req = urllib.request.Request(url, data=data, headers={'User-Agent': 'Mozilla/5.0'})
            res = urllib.request.urlopen(req, timeout=20)
            response_data = json.loads(res.read().decode())
            blat_results = response_data.get('blat', [])
            if blat_results:
                blat_results.sort(key=lambda x: x[0], reverse=True)
                best = blat_results[0]
                return {
                    'chr': best[13],
                    'strand': best[8],
                    'matches': best[0],
                    'tStart': best[15],
                    'tEnd': best[16]
                }
        except Exception as e:
            print(f"Error querying BLAT (attempt {attempt+1}) for {seq}: {e}")
    return None

def fetch_genomic_sequence(chrom, start, stop):
    time.sleep(2.0)
    url = f'https://genome.ucsc.edu/cgi-bin/das/hg38/dna?segment={chrom}:{start},{stop}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=20)
        xml_data = res.read().decode()
        root = ET.fromstring(xml_data)
        dna_element = root.find('.//DNA')
        if dna_element is not None and dna_element.text:
            return ''.join(dna_element.text.split()).upper()
    except Exception as e:
        print(f"Error fetching DAS {chrom}:{start}-{stop}: {e}")
    return None

def align_and_reconstruct(row, blat_res, genomic_seq):
    pbs = str(row['PBS_pegRNA_DNA']).upper()
    rc_pbs = str(Seq(pbs).reverse_complement())
    
    rtt_peg = str(row['RTT_template_DNA']).upper()
    rc_rtt = str(Seq(rtt_peg).reverse_complement())
    
    fwd_match = False
    fwd_strand = True
    
    rtt_flank = rc_rtt[-12:] if len(rc_rtt) >= 12 else rc_rtt
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
    
    wt_rtt_len = wt_rtt_end - nick_idx
    wt_rtt = genomic_seq[nick_idx:wt_rtt_end]
    
    # Construct 74nt wt sequence for DeepPrime (nick at index 21)
    dp_wt_start = nick_idx - 21
    dp_wt_end = nick_idx + 53
    dp_wt_seq = genomic_seq[dp_wt_start:dp_wt_end]
    
    # Construct 200nt wt sequence for PRIDICT (nick at index 100)
    pr_wt_start = nick_idx - 100
    pr_wt_end = nick_idx + 100
    pr_wt_seq = genomic_seq[pr_wt_start:pr_wt_end]
    
    # Construct edited sequence
    pr_ed_seq = pr_wt_seq[:100] + rc_rtt + pr_wt_seq[100 + wt_rtt_len:]
    
    # Determine edit type, length, and starting position relative to nick
    if len(rc_rtt) == len(wt_rtt):
        edit_type = "Sub"
        diffs = [i for i in range(len(rc_rtt)) if rc_rtt[i] != wt_rtt[i]]
        if diffs:
            edit_pos = diffs[0] + 1
            edit_len = len(diffs)
        else:
            edit_pos = 1
            edit_len = 0
    elif len(rc_rtt) > len(wt_rtt):
        edit_type = "Ins"
        edit_len = len(rc_rtt) - len(wt_rtt)
        min_len = min(len(rc_rtt), len(wt_rtt))
        diverge = min_len
        for i in range(min_len):
            if rc_rtt[i] != wt_rtt[i]:
                diverge = i
                break
        edit_pos = diverge + 1
    else:
        edit_type = "Del"
        edit_len = len(wt_rtt) - len(rc_rtt)
        min_len = min(len(rc_rtt), len(wt_rtt))
        diverge = min_len
        for i in range(min_len):
            if rc_rtt[i] != wt_rtt[i]:
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
        'transcript': blat_res.get('transcript', 'unknown'),
        'strand': '+' if fwd_strand else '-'
    }

def get_longest_active_row(df, aid):
    subset = df[df['ID_str'] == str(aid).strip()]
    if subset.empty:
        return None
    # Calculate sum of PBSlen and RTlen
    subset = subset.copy()
    subset['total_len'] = subset['PBSlen'] + subset['RTlen']
    sorted_sub = subset.sort_values('total_len', ascending=False)
    return sorted_sub.iloc[0]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=None, help="Path to selected candidates CSV to check and patch specific mismatching candidates")
    parser.add_argument("--data-dir", default="data", help="Directory for input/output data")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    raw_path = os.path.join(data_dir, "MFE_randompeg_RHA30_0.71M_result.csv")  # Source of the Supp3 analysis
    resolved_path = os.path.join(data_dir, "ncbi_cache", "resolved_pegrna_genomic_details.json")  # Output of Step 0b 
    
    df = pd.read_csv(raw_path)
    df['ID_str'] = df['ID'].astype(str).str.strip()
    
    if not os.path.exists(resolved_path):
        print(f"[*] Resolved details JSON {resolved_path} not found. Cannot patch.")
        return
        
    with open(resolved_path) as f:
        resolved = json.load(f)
        
    # Write empty cache_patched flag file first
    flag_file = os.path.join(data_dir, "ncbi_cache", "cache_patched.txt")
    with open(flag_file, "w") as f_flag:
        f_flag.write("0")
        
    if args.candidates:
        if not os.path.exists(args.candidates):
            print(f"Candidates file {args.candidates} does not exist. Skipping candidate-specific check.")
            return
            
        print(f"Loading candidates from {args.candidates} for specific genomic alignment verification...")
        cand_df = pd.read_csv(args.candidates)
        
        # Check mismatch for each candidate
        to_repair_rows = []
        for idx, row in cand_df.iterrows():
            aid = str(row['ID']).strip()
            pbs = str(row['PBS_pegRNA_DNA']).upper()
            rc_pbs = str(Seq(pbs).reverse_complement())
            
            if aid not in resolved:
                print(f"[*] Candidate ID {aid} not in resolved cache. Will attempt to resolve.")
                to_repair_rows.append(row)
                continue
                
            wt_200 = resolved[aid].get('wt_pridict_200', '').upper()
            pbs_found = (wt_200.find(pbs) != -1) or (wt_200.find(rc_pbs) != -1)
            
            if not pbs_found:
                print(f"[*] Selected candidate {aid} PBS '{pbs}' / rcPBS '{rc_pbs}' not found in cached wt_200. Re-aligning cache for this candidate...")
                to_repair_rows.append(row)
                
        if not to_repair_rows:
            print("[+] All selected candidates have correct genomic alignment in cache.")
            return
            
        print(f"[*] Attempting to repair cache alignment for {len(to_repair_rows)} mismatched candidates...")
        fixed_count = 0
        for idx, cand_row in enumerate(to_repair_rows):
            aid = str(cand_row['ID']).strip()
            # We want to match the exact row in the raw CSV
            orig_idx = int(cand_row['orig_index']) if 'orig_index' in cand_row else None
            if orig_idx is not None:
                rep_row = df.iloc[orig_idx]
            else:
                matches = df[df['ID_str'] == aid]
                if matches.empty:
                    continue
                rep_row = matches.iloc[0]
                
            pbs = str(rep_row['PBS_pegRNA_DNA']).upper()
            rc_pbs = str(Seq(pbs).reverse_complement())
            rtt_peg = rep_row['RTT_template_DNA'].upper()
            rc_rtt = str(Seq(rtt_peg).reverse_complement())
            active_seq = rc_pbs + rc_rtt
            
            blat_res = query_ucsc_blat(active_seq)
            if not blat_res:
                blat_res = query_ucsc_blat(str(Seq(active_seq).reverse_complement()))
                
            chrom = None
            if blat_res and blat_res['matches'] >= 20:
                chrom = blat_res['chr']
                start_coord = blat_res['tStart'] - 200
                stop_coord = blat_res['tEnd'] + 200
                strand_val = blat_res['strand']
            else:
                # Fallback to HGVS mapping coordinates
                with open(os.path.join(data_dir, 'ncbi_cache', 'clinvar_hgvs_mapping.json')) as f_map:
                    mapping = json.load(f_map)
                if aid in mapping:
                    m_val = mapping[aid]
                    chrom = "chr" + str(m_val['chr'])
                    start_coord = int(m_val['start']) - 200
                    stop_coord = int(m_val['stop']) + 200
                    strand_val = '+'
                    blat_res = {
                        'chr': chrom,
                        'strand': strand_val,
                        'tStart': int(m_val['start']),
                        'tEnd': int(m_val['stop']),
                        'transcript': m_val.get('title', 'unknown').split('(')[0] if '(' in m_val.get('title', '') else 'unknown'
                    }
                    
            if not chrom:
                print(f"[-] Candidate ID {aid}: UCSC BLAT / Fallback resolution failed.")
                continue
                
            genomic_seq = fetch_genomic_sequence(chrom, start_coord, stop_coord)
            if not genomic_seq:
                continue
                
            res = align_and_reconstruct(rep_row, blat_res, genomic_seq)
            if res:
                res['chr'] = chrom
                with open(os.path.join(data_dir, 'ncbi_cache', 'clinvar_hgvs_mapping.json')) as f_map:
                    mapping = json.load(f_map)
                if aid in mapping:
                    title = mapping[aid]['title']
                    tx = title.split('(')[0] if '(' in title else 'unknown'
                    res['transcript'] = tx
                    res['gene'] = mapping[aid]['gene']
                    
                resolved[aid] = res
                fixed_count += 1
                print(f"[+] Candidate ID {aid} cache updated. Gene={res.get('gene')}, Transcript={res.get('transcript')}")
                
                # Save cache incrementally
                with open(resolved_path, "w") as f_save:
                    json.dump(resolved, f_save, indent=2)
            else:
                print(f"[-] Candidate ID {aid} local reconstruction alignment failed.")
                
        print(f"Candidate-specific cache patch complete. Patched {fixed_count} candidates.")
        with open(flag_file, "w") as f_flag:
            f_flag.write(str(fixed_count))
        return

    print("Loaded resolved details. Scanning for coordinate shifts...")
    
    # Collect IDs that have shifted/mismatched genomic coordinates
    to_repair = []
    for aid in list(resolved.keys()):
        rep_row = get_longest_active_row(df, aid)
        if rep_row is None:
            continue
        pbs = str(rep_row['PBS_pegRNA_DNA']).upper()
        rc_pbs = str(Seq(pbs).reverse_complement())
        wt_200 = resolved[aid].get('wt_pridict_200', '').upper()
        
        pbs_found = (wt_200.find(pbs) != -1) or (wt_200.find(rc_pbs) != -1)
        if not pbs_found:
            to_repair.append(aid)
            
    # Pre-inference Wide Cache Fetching for Large Indel Variants (>15bp)
    print("Checking if any large indels (>15bp) need wider genomic context cache...")
    large_indels = [k for k, v in resolved.items() if v.get('edit_len', 0) > 15]
    print(f"Total large indel variants to check: {len(large_indels)}")
    
    mapping_path = os.path.join(data_dir, 'ncbi_cache', 'clinvar_hgvs_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path) as f_map:
            mapping = json.load(f_map)
            
        wide_count = 0
        for aid in large_indels:
            if aid in resolved and 'wt_genomic_wide' not in resolved[aid]:
                m_val = mapping.get(aid)
                if not m_val:
                    continue
                chrom = "chr" + str(m_val['chr'])
                # Fetch 2000bp centered on variant (1000bp upstream/downstream)
                start_coord = int(m_val['start']) - 1000
                stop_coord = int(m_val['stop']) + 1000
                print(f"Fetching wide genomic sequence (2000bp) for large indel ID {aid} ({chrom}:{start_coord}-{stop_coord})...")
                genomic_seq = fetch_genomic_sequence(chrom, start_coord, stop_coord)
                if genomic_seq:
                    resolved[aid]['wt_genomic_wide'] = genomic_seq
                    resolved[aid]['wide_start'] = start_coord
                    resolved[aid]['wide_stop'] = stop_coord
                    wide_count += 1
                    
        if wide_count > 0:
            print(f"[+] Successfully loaded and cached {wide_count} wide genomic sequences.")
            with open(resolved_path, "w") as f_save:
                json.dump(resolved, f_save, indent=2)
    else:
        print("Warning: HGVS mapping file not found. Skipping wide cache fetch.")

    if not to_repair:
        print("[+] No coordinate shifts detected in cache. Cache is fully correct!")
        return
        
    print(f"[*] Detected coordinate shifts for {len(to_repair)} variants: {to_repair}")
    
    # Force rebuild of coordinate mismatches
    for aid in to_repair:
        resolved.pop(aid, None)
        
    fixed_count = 0
    for idx, aid in enumerate(to_repair):
        # Find all rows for this variant, sorted by total length descending
        subset = df[df['ID_str'] == str(aid).strip()]
        if subset.empty:
            print(f"[-] Target ID {aid}: Not found in raw CSV.")
            continue
            
        subset = subset.copy()
        subset['total_len'] = subset['PBSlen'] + subset['RTlen']
        sorted_sub = subset.sort_values('total_len', ascending=False)
        
        # Try each design in order of length until one successfully aligns
        res = None
        for _, rep_row in sorted_sub.iterrows():
            pbs = str(rep_row['PBS_pegRNA_DNA']).upper()
            rc_pbs = str(Seq(pbs).reverse_complement())
            rtt_peg = rep_row['RTT_template_DNA'].upper()
            rc_rtt = str(Seq(rtt_peg).reverse_complement())
            active_seq = rc_pbs + rc_rtt
            
            blat_res = query_ucsc_blat(active_seq)
            if not blat_res:
                blat_res = query_ucsc_blat(str(Seq(active_seq).reverse_complement()))
                
            chrom = None
            if blat_res and blat_res['matches'] >= 20:
                chrom = blat_res['chr']
                start_coord = blat_res['tStart'] - 200
                stop_coord = blat_res['tEnd'] + 200
                strand_val = blat_res['strand']
            else:
                # Fallback to coordinate-based lookup from mapping JSON
                with open('data/ncbi_cache/clinvar_hgvs_mapping.json') as f_map:
                    mapping = json.load(f_map)
                if aid in mapping:
                    m_val = mapping[aid]
                    chrom = "chr" + str(m_val['chr'])
                    start_coord = int(m_val['start']) - 200
                    stop_coord = int(m_val['stop']) + 200
                    strand_val = '+' # default guess
                    blat_res = {
                        'chr': chrom,
                        'strand': strand_val,
                        'tStart': int(m_val['start']),
                        'tEnd': int(m_val['stop']),
                        'transcript': m_val.get('title', 'unknown').split('(')[0] if '(' in m_val.get('title', '') else 'unknown'
                    }
                    
            if not chrom:
                continue
                
            genomic_seq = fetch_genomic_sequence(chrom, start_coord, stop_coord)
            if not genomic_seq:
                continue
                
            res = align_and_reconstruct(rep_row, blat_res, genomic_seq)
            if res:
                res['chr'] = chrom
                with open('data/ncbi_cache/clinvar_hgvs_mapping.json') as f_map:
                    mapping = json.load(f_map)
                if aid in mapping:
                    title = mapping[aid]['title']
                    tx = title.split('(')[0] if '(' in title else 'unknown'
                    res['transcript'] = tx
                    res['gene'] = mapping[aid]['gene']
                
                # Success! Break the design trial loop
                break
                
        if res:
            resolved[aid] = res
            fixed_count += 1
            print(f"[+] Target ID {aid}: Successfully patched. Transcript={res.get('transcript')}, Gene={res.get('gene')}")
            
            # Save cache incrementally
            with open(resolved_path, "w") as f_save:
                json.dump(resolved, f_save, indent=2)
        else:
            print(f"[-] Target ID {aid}: Alignment mapping failed for all designs.")
            
    print(f"Cache patch complete. Patched {fixed_count} / {len(to_repair)} variants.")
    


if __name__ == '__main__':
    main()
