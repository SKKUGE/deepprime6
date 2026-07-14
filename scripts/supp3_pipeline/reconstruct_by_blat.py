import os
import json
import pandas as pd
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from Bio.Seq import Seq

def clean_id(x):
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s

def try_align(pbs, rtt_peg, tx_seq):
    rc_pbs = str(Seq(pbs).reverse_complement())
    rtt = str(Seq(rtt_peg).reverse_complement())
    rtt_flank = rtt[-min(12, len(rtt)):]
    for fwd, seq in [(True, tx_seq), (False, str(Seq(tx_seq).reverse_complement()))]:
        ip = seq.find(rc_pbs)
        iff = seq.find(rtt_flank)
        if ip != -1 and iff != -1 and iff > ip:
            return ip + len(rc_pbs), iff + len(rtt_flank), fwd, seq, rtt
    return None

def reconstruct_from_align(row, nick, wt_end, fwd, seq, rtt, gene, tx_id, chrom):
    pr_s, pr_e = nick - 100, nick + 100
    dp_s, dp_e = nick - 21, nick + 53
    if pr_s < 0 or pr_e > len(seq) or dp_s < 0 or dp_e > len(seq):
        return None
    pr_wt = seq[pr_s:pr_e]
    pr_ed = pr_wt[:100] + rtt + pr_wt[100 + (wt_end - nick):]
    dp_wt = seq[dp_s:dp_e]
    wt_rtt = seq[nick:wt_end]
    
    # Extract 2000bp wt genomic wide centered on the nick site
    wide_s = nick - 1000
    wide_e = nick + 1000
    padded_seq = seq
    offset_pad = 0
    if wide_s < 0:
        pad_len = abs(wide_s)
        padded_seq = ("N" * pad_len) + padded_seq
        offset_pad = pad_len
        wide_s = 0
        wide_e += offset_pad
    if wide_e > len(padded_seq):
        pad_len = wide_e - len(padded_seq)
        padded_seq = padded_seq + ("N" * pad_len)
    wt_genomic_wide = padded_seq[wide_s:wide_e]
    
    if len(wt_rtt) == len(rtt):
        etype = "Sub"
        diffs = [i for i in range(len(rtt)) if rtt[i] != wt_rtt[i]]
        epos = diffs[0] + 1 if diffs else 1
        elen = len(diffs)
    elif len(rtt) > len(wt_rtt):
        etype, elen = "Ins", len(rtt) - len(wt_rtt)
        ml = min(len(rtt), len(wt_rtt))
        d = ml
        for i in range(ml):
            if rtt[i] != wt_rtt[i]:
                d = i
                break
        epos = d + 1
    else:
        etype, elen = "Del", len(wt_rtt) - len(rtt)
        ml = min(len(rtt), len(wt_rtt))
        d = ml
        for i in range(ml):
            if rtt[i] != wt_rtt[i]:
                d = i
                break
        epos = d + 1
        
    return {
        'wt_target_74': dp_wt,
        'wt_pridict_200': pr_wt,
        'ed_pridict_200': pr_ed,
        'wt_genomic_wide': wt_genomic_wide,
        'edit_type': etype,
        'edit_len': elen,
        'edit_pos': epos,
        'gene': gene,
        'transcript': tx_id,
        'strand': '+' if fwd else '-',
        'chr': chrom
    }

def query_ucsc_blat(seq):
    time.sleep(1.0)
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
            best = None
            for hit in blat_results:
                chrom_name = hit[13]
                clean_chrom = chrom_name.lower().replace("chr", "")
                if clean_chrom.isdigit() or clean_chrom in ['x', 'y', 'm', 'mt']:
                    best = hit
                    break
            if not best:
                best = blat_results[0]
            return {
                'chr': best[13],
                'strand': best[8],
                'matches': best[0],
                'tStart': best[15],
                'tEnd': best[16]
            }
    except Exception as e:
        print(f"Error querying BLAT for {seq[:20]}...: {e}")
    return None

def fetch_genomic_sequence(chrom, start, stop, cache):
    chrom_clean = chrom.lower().replace("chr", "")
    if chrom_clean == "mt":
        chrom_clean = "MT"
    else:
        chrom_clean = chrom_clean.upper()
        
    key = f"{chrom_clean}:{start}-{stop}"
    if key in cache:
        return cache[key]
        
    # 1. Query Ensembl REST API
    url = f"https://rest.ensembl.org/sequence/region/human/{chrom_clean}:{start}..{stop}?content-type=text/plain"
    try:
        time.sleep(0.2)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=10)
        if res.status == 200:
            dna_seq = res.read().decode().upper()
            cache[key] = dna_seq
            return dna_seq
    except Exception as e:
        print(f"Ensembl REST query failed for {key}: {e}. Trying UCSC DAS...")
        
    # 2. Fallback to UCSC DAS
    chr_name = chrom if str(chrom).startswith('chr') else f"chr{chrom}"
    url_das = f'https://genome.ucsc.edu/cgi-bin/das/hg38/dna?segment={chr_name}:{start},{stop}'
    try:
        time.sleep(1.0)
        req = urllib.request.Request(url_das, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=15)
        xml_data = res.read().decode()
        root = ET.fromstring(xml_data)
        dna_element = root.find('.//DNA')
        if dna_element is not None and dna_element.text:
            dna_seq = ''.join(dna_element.text.split()).upper()
            cache[key] = dna_seq
            return dna_seq
    except Exception as e:
        print(f"Error fetching UCSC DAS {key}: {e}")
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory for input data")
    args = parser.parse_args()
    data_dir = args.data_dir

    t_start = time.time()
    raw_path = os.path.join(data_dir, "MFE_randompeg_RHA30_0.71M_result.csv")
    cache_path = os.path.join(data_dir, "ncbi_cache/genomic_cache.json")
    resolved_path = os.path.join(data_dir, "ncbi_cache/resolved_pegrna_genomic_details.json")
    tx_cache_path = os.path.join(data_dir, "ncbi_cache/transcript_cache.json")
    
    print("Loading transcript cache...")
    if os.path.exists(tx_cache_path):
        with open(tx_cache_path) as f:
            tx_cache = json.load(f)
        print(f"Loaded {len(tx_cache)} transcripts from cache.")
    else:
        tx_cache = {}
        
    print("Loading raw pegRNA dataset...")
    df = pd.read_csv(raw_path)
    df['clean_id'] = df['ID'].apply(clean_id)
    
    # 1. Group by clean_id and find the row with the longest unmasked Edited74_On sequence
    print("Finding representative rows with longest unmasked Edited74_On sequences...")
    df['unmasked_len'] = df['Edited74_On'].astype(str).str.replace('x', '').str.len()
    idx_max = df.groupby('clean_id')['unmasked_len'].idxmax()
    df_rep = df.loc[idx_max].copy()
    
    print(f"Total representative rows (unique raw IDs) to resolve: {len(df_rep)}")
    
    # Load caches
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} sequences from genomic cache.")
    else:
        cache = {}
        
    if os.path.exists(resolved_path):
        with open(resolved_path) as f:
            resolved = json.load(f)
        print(f"Loaded {len(resolved)} resolved variants from cache.")
    else:
        resolved = {}
        
    unresolved_raw_ids = [rid for rid in df_rep['clean_id'] if rid not in resolved]
    print(f"Variants left to resolve: {len(unresolved_raw_ids)}")
    
    success = 0
    total = unresolved_raw_ids
    
    for idx, rid in enumerate(unresolved_raw_ids, 1):
        row = df_rep[df_rep['clean_id'] == rid].iloc[0]
        pbs = str(row['PBS_pegRNA_DNA']).upper()
        rtt_peg = str(row['RTT_template_DNA']).upper()
        
        # 1. Try local offline alignment via transcript cache first
        aligned_offline = False
        for tx_id, tx_seq in tx_cache.items():
            r = try_align(pbs, rtt_peg, tx_seq)
            if r:
                nick, wt_end, fwd, seq, rtt = r
                rec = reconstruct_from_align(row, nick, wt_end, fwd, seq, rtt, 'unknown', tx_id, 'unknown')
                if rec:
                    resolved[rid] = rec
                    aligned_offline = True
                    break
        
        if aligned_offline:
            success += 1
            print(f"[{idx}/{len(total)}] ID {rid} - Resolved offline using transcript cache.")
            with open(resolved_path, "w") as f:
                json.dump(resolved, f, indent=2)
            continue
            
        # 2. Online BLAT/DAS/Ensembl fallback if offline alignment fails
        print(f"[{idx}/{len(total)}] ID {rid} - Offline alignment failed. Running online reconstruction fallback...")
        query_seq = str(row['Edited74_On']).replace('x', '')
        if not query_seq:
            print(f"[{idx}/{len(total)}] ID {rid} - Empty unmasked sequence.")
            continue
            
        blat_res = None
        if len(query_seq) >= 20:
            blat_res = query_ucsc_blat(query_seq)
            if not blat_res:
                rc_query = str(Seq(query_seq).reverse_complement())
                blat_res = query_ucsc_blat(rc_query)
                
        if not blat_res:
            print(f"[{idx}/{len(total)}] ID {rid} - Sequence too short or BLAT failed.")
            continue
                
        # Fetch genomic sequence (±1000 bp with 1000bp rounding coordinates to maximize cache hits)
        chrom = blat_res['chr']
        start_coord = (blat_res['tStart'] // 1000) * 1000 - 1000
        stop_coord = ((blat_res['tEnd'] // 1000) + 1) * 1000 + 1000
        
        genomic_seq = fetch_genomic_sequence(chrom, start_coord, stop_coord, cache)
        if not genomic_seq:
            print(f"[{idx}/{len(total)}] ID {rid} - Fetch genomic sequence failed.")
            continue
            
        # PBS Alignment
        rc_pbs = str(Seq(pbs).reverse_complement())
        fwd_match = False
        fwd_strand = True
        pos = genomic_seq.find(rc_pbs)
        if pos != -1:
            fwd_match = True
            fwd_strand = True
            matched_seq = genomic_seq
        else:
            rc_genomic = str(Seq(genomic_seq).reverse_complement())
            pos = rc_genomic.find(rc_pbs)
            if pos != -1:
                fwd_match = True
                fwd_strand = False
                matched_seq = rc_genomic
                
        if not fwd_match:
            print(f"[{idx}/{len(total)}] ID {rid} - rc_PBS alignment failed.")
            continue
            
        raw_nick = pos + len(rc_pbs)
        best_offset = 999
        best_pam = None
        best_pam_pos = -1
        best_pam_strand = None
        
        for offset in range(-15, 15):
            i = raw_nick + offset
            if i < 0 or i + 3 > len(matched_seq):
                continue
            triplet = matched_seq[i:i+3]
            if triplet[1:3] == "GG":
                cut_pos = i - 3
                dist = abs(cut_pos - raw_nick)
                if dist < best_offset:
                    best_offset = dist
                    best_pam = triplet
                    best_pam_pos = i
                    best_pam_strand = '+'
            if triplet[0:2] == "CC":
                cut_pos = i + 6
                dist = abs(cut_pos - raw_nick)
                if dist < best_offset:
                    best_offset = dist
                    best_pam = triplet
                    best_pam_pos = i
                    best_pam_strand = '-'
                    
        if best_offset <= 6:
            nick_site = raw_nick + (best_pam_pos - raw_nick - 3 if best_pam_strand == '+' else best_pam_pos - raw_nick + 6)
        else:
            nick_site = raw_nick
            
        rc_rtt = str(Seq(rtt_peg).reverse_complement())
        rtt_flank = rc_rtt[-min(12, len(rc_rtt)):]
        idx_flank = matched_seq.find(rtt_flank, nick_site)
        if idx_flank != -1:
            wt_rtt_end = idx_flank + len(rtt_flank)
        else:
            wt_rtt_end = nick_site + len(rc_rtt)
            
        wt_rtt = matched_seq[nick_site : wt_rtt_end]
        
        edit_type = "Sub"
        edit_len = 1
        edit_pos = 1
        
        if len(wt_rtt) == len(rc_rtt):
            edit_type = "Sub"
            diffs = [i for i in range(len(rc_rtt)) if rc_rtt[i] != wt_rtt[i]]
            edit_pos = diffs[0] + 1 if diffs else 1
            edit_len = len(diffs)
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
            
        dp_wt_start = nick_site - 21
        dp_wt_end = nick_site + 53
        if dp_wt_start < 0 or dp_wt_end > len(matched_seq):
            print(f"[{idx}/{len(total)}] ID {rid} - DeepPrime window out of bounds.")
            continue
        dp_wt_seq = matched_seq[dp_wt_start:dp_wt_end]
        
        pr_wt_start = nick_site - 100
        pr_wt_end = nick_site + 100
        if pr_wt_start < 0 or pr_wt_end > len(matched_seq):
            print(f"[{idx}/{len(total)}] ID {rid} - PRIDICT window out of bounds.")
            continue
        pr_wt_seq = matched_seq[pr_wt_start:pr_wt_end]
        pr_ed_seq = pr_wt_seq[:100] + rc_rtt + pr_wt_seq[100 + (wt_rtt_end - nick_site):]
        
        # Construct 2000bp wt genomic wide centered on the nick site
        wide_s = nick_site - 1000
        wide_e = nick_site + 1000
        padded_wide = matched_seq
        offset_pad = 0
        if wide_s < 0:
            pad_len = abs(wide_s)
            padded_wide = ("N" * pad_len) + padded_wide
            offset_pad = pad_len
            wide_s = 0
            wide_e += offset_pad
        if wide_e > len(padded_wide):
            pad_len = wide_e - len(padded_wide)
            padded_wide = padded_wide + ("N" * pad_len)
        wt_genomic_wide = padded_wide[wide_s:wide_e]

        gene = 'unknown'
        transcript = 'unknown'
            
        resolved[rid] = {
            'wt_target_74': dp_wt_seq,
            'wt_pridict_200': pr_wt_seq,
            'ed_pridict_200': pr_ed_seq,
            'wt_genomic_wide': wt_genomic_wide,
            'edit_type': edit_type,
            'edit_len': edit_len,
            'edit_pos': edit_pos,
            'gene': gene,
            'transcript': transcript,
            'strand': '+' if (blat_res['strand'] == '+' if fwd_strand else blat_res['strand'] == '-') else '-',
            'chr': chrom
        }
        
        success += 1
        print(f"[{idx}/{len(total)}] ID {rid} - Successfully resolved via online fallback.")
        
        # Save cache incrementally
        with open(resolved_path, "w") as f:
            json.dump(resolved, f, indent=2)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
            
    print(f"\nDone! Resolved {success} new variants. Total resolved: {len(resolved)} / {len(df_rep['clean_id'].unique())}")
    print(f"Total time: {time.time() - t_start:.2f}s")

if __name__ == '__main__':
    main()
