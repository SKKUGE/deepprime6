import os
import sys
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import time
import re
from Bio.Seq import Seq

def submit_blast_job(fasta_string):
    url = 'https://blast.ncbi.nlm.nih.gov/Blast.cgi'
    params = {
        'CMD': 'Put',
        'PROGRAM': 'blastn',
        'DATABASE': 'refseq_select_rna',
        'QUERY': fasta_string,
        'SHORT_QUERY_ADJUST': 'true',
        'EXPECT': '1000',
        'WORD_SIZE': '7',
        'FORMAT_TYPE': 'XML'
    }
    data = urllib.parse.urlencode(params).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        res = urllib.request.urlopen(req)
        response_text = res.read().decode()
        rid = None
        rtoe = 20
        for line in response_text.split('\n'):
            if 'RID =' in line:
                rid = line.split('RID =')[1].strip()
            if 'RTOE =' in line:
                rtoe = int(line.split('RTOE =')[1].strip())
        return rid, rtoe
    except Exception as e:
        print("Error submitting BLAST:", e)
        return None, None

def check_blast_status(rid):
    url = f'https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Get&FORMAT_OBJECT=SearchInfo&RID={rid}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res = urllib.request.urlopen(req)
        response_text = res.read().decode()
        if 'Status=WAITING' in response_text:
            return 'WAITING'
        elif 'Status=FAILED' in response_text:
            return 'FAILED'
        elif 'Status=UNKNOWN' in response_text:
            return 'UNKNOWN'
        elif 'Status=READY' in response_text:
            return 'READY'
    except Exception as e:
        print("Error checking status:", e)
    return 'ERROR'

def retrieve_blast_results(rid):
    url = f'https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Get&FORMAT_TYPE=XML&RID={rid}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res = urllib.request.urlopen(req)
        return res.read().decode()
    except Exception as e:
        print("Error retrieving results:", e)
        return None

def parse_blast_xml(xml_content):
    xml_content = re.sub(r'<BlastOutput_reference>.*?</BlastOutput_reference>', '', xml_content, flags=re.DOTALL)
    xml_content = xml_content.replace('&amp;auml;', 'a')
    
    root = ET.fromstring(xml_content)
    align_map = {}
    
    for iteration in root.findall('.//Iteration'):
        query_def = iteration.find('Iteration_query-def').text
        aid = query_def.replace('seq_', '')
        
        hits = iteration.find('.//Iteration_hits')
        if hits is not None and len(hits) > 0:
            best_hit = hits.find('Hit')
            hit_id = best_hit.find('Hit_id').text
            
            nm_accession = ''
            for part in hit_id.split('|'):
                if part.startswith('NM_'):
                    nm_accession = part
                    break
            if not nm_accession:
                nm_accession = best_hit.find('Hit_accession').text
                
            hsp = best_hit.find('.//Hsp')
            h_from = int(hsp.find('Hsp_hit-from').text)
            h_to = int(hsp.find('Hsp_hit-to').text)
            
            strand = '+'
            if h_from > h_to:
                strand = '-'
                t_start = h_to
                t_end = h_from
            else:
                strand = '+'
                t_start = h_from
                t_end = h_to
                
            align_map[aid] = {
                'transcript': nm_accession,
                'strand': strand,
                'tStart': t_start,
                'tEnd': t_end
            }
    return align_map

def fetch_single_transcript(nm_id, cache):
    if nm_id in cache:
        return cache[nm_id]
        
    time.sleep(0.4) # Respect NCBI rate limits
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id={nm_id}&rettype=fasta&retmode=text'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=15)
        lines = res.read().decode().split('\n')
        seq = ''.join(line.strip() for line in lines[1:]).upper()
        if seq:
            cache[nm_id] = seq
            return seq
    except Exception as e:
        print(f"Error fetching transcript {nm_id}: {e}")
    return None

def align_and_reconstruct(row, align_meta, transcript_seq):
    pbs = row['PBS_pegRNA_DNA'].upper()
    rc_pbs = str(Seq(pbs).reverse_complement())
    
    rtt_peg = row['RTT_template_DNA'].upper()
    rtt = str(Seq(rtt_peg).reverse_complement())
    
    fwd_match = False
    fwd_strand = True
    
    rtt_flank = rtt[-12:]
    idx_pbs = transcript_seq.find(rc_pbs)
    idx_flank = transcript_seq.find(rtt_flank)
    
    if idx_pbs != -1 and idx_flank != -1 and idx_flank > idx_pbs:
        fwd_match = True
        fwd_strand = True
    else:
        rc_transcript = str(Seq(transcript_seq).reverse_complement())
        idx_pbs_rc = rc_transcript.find(rc_pbs)
        idx_flank_rc = rc_transcript.find(rtt_flank)
        if idx_pbs_rc != -1 and idx_flank_rc != -1 and idx_flank_rc > idx_pbs_rc:
            fwd_match = True
            fwd_strand = False
            transcript_seq = rc_transcript
            idx_pbs = idx_pbs_rc
            idx_flank = idx_flank_rc
            
    if not fwd_match:
        return None
        
    nick_idx = idx_pbs + len(rc_pbs)
    wt_rtt_end = idx_flank + len(rtt_flank)
    
    pr_wt_start = nick_idx - 100
    pr_wt_end = nick_idx + 100
    if pr_wt_start < 0 or pr_wt_end > len(transcript_seq):
        return None
        
    pr_wt_seq = transcript_seq[pr_wt_start:pr_wt_end]
    pr_ed_seq = pr_wt_seq[:100] + rtt + pr_wt_seq[100 + (wt_rtt_end - nick_idx):]
    
    dp_wt_start = nick_idx - 21
    dp_wt_end = nick_idx + 53
    if dp_wt_start < 0 or dp_wt_end > len(transcript_seq):
        return None
        
    dp_wt_seq = transcript_seq[dp_wt_start:dp_wt_end]
    
    wt_rtt = transcript_seq[nick_idx : wt_rtt_end]
    
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
        'transcript': align_meta['transcript'],
        'strand': '+' if (align_meta['strand'] == '+' if fwd_strand else align_meta['strand'] == '-') else '-'
    }

def main():
    print("Loading pegRNA CSV dataset...")
    df = pd.read_csv("data/MFE_randompeg_RHA30_0.71M_result.csv")
    all_raw_ids = set(df['ID'].astype(str).unique())
    print(f"Total raw unique IDs in dataset: {len(all_raw_ids)}")
    
    resolved_details_file = "data/ncbi_cache/resolved_pegrna_genomic_details.json"
    
    # Load existing resolved details
    with open(resolved_details_file) as f:
        resolved = json.load(f)
        
    print(f"Loaded {len(resolved)} keys from JSON.")
    
    # Keep only resolved entries that correspond to existing raw IDs
    renamed_resolved = {k: v for k, v in resolved.items() if k in all_raw_ids}
    print(f"Validated resolved details count: {len(renamed_resolved)}")
    
    # Identify unresolved raw IDs
    unresolved_ids = [rid for rid in all_raw_ids if rid not in renamed_resolved]
    print(f"Unresolved raw IDs left: {len(unresolved_ids)}")
    print(f"List of unresolved IDs: {unresolved_ids}")
    
    if not unresolved_ids:
        print("All raw IDs already resolved. Saving renamed cache.")
        with open(resolved_details_file, "w") as f:
            json.dump(renamed_resolved, f, indent=2)
        return
        
    # Load transcript cache
    tx_cache_file = "data/ncbi_cache/transcript_cache.json"
    if os.path.exists(tx_cache_file):
        with open(tx_cache_file) as f:
            tx_cache = json.load(f)
        print(f"Loaded {len(tx_cache)} transcript sequences from cache.")
    else:
        tx_cache = {}
        
    # Process the remaining unresolved IDs in a single batch
    batch = unresolved_ids
    print(f"\n--- Processing final batch ({len(batch)} variants) ---")
    
    fasta_lines = []
    for aid in batch:
        subset = df[df['ID'].astype(str) == aid]
        row = subset.iloc[0]
        pbs = row['PBS_pegRNA_DNA'].upper()
        rc_pbs = str(Seq(pbs).reverse_complement())
        rtt_peg = row['RTT_template_DNA'].upper()
        rtt = str(Seq(rtt_peg).reverse_complement())
        active_seq = rc_pbs + rtt
        fasta_lines.append(f">seq_{aid}\n{active_seq}")
        
    fasta_string = '\n'.join(fasta_lines)
    rid, rtoe = submit_blast_job(fasta_string)
    if not rid:
        print("BLAST submission failed.")
        return
        
    print(f"Waiting for BLAST RID: {rid} ({rtoe}s estimated)...")
    time.sleep(rtoe)
    
    while True:
        status = check_blast_status(rid)
        print(f"BLAST Job Status: {status}")
        if status == 'READY':
            break
        elif status in ['FAILED', 'UNKNOWN', 'ERROR']:
            break
        time.sleep(5)
        
    if status != 'READY':
        print("BLAST job not ready or failed.")
        return
        
    xml_results = retrieve_blast_results(rid)
    if not xml_results:
        print("Failed to retrieve XML results.")
        return
        
    align_map = parse_blast_xml(xml_results)
    print(f"Mapped {len(align_map)} / {len(batch)} variants to RefSeq Transcripts.")
    
    success = 0
    for aid, align_meta in align_map.items():
        subset = df[df['ID'].astype(str) == aid]
        rep_row = subset.iloc[0]
        tx_id = align_meta['transcript']
        
        tx_seq = fetch_single_transcript(tx_id, tx_cache)
        if not tx_seq:
            continue
            
        res = align_and_reconstruct(rep_row, align_meta, tx_seq)
        if res:
            success += 1
            renamed_resolved[aid] = res
            
    print(f"Completed final batch: {success} successfully resolved.")
    
    # Save results
    with open(resolved_details_file, "w") as f:
        json.dump(renamed_resolved, f, indent=2)
    with open(tx_cache_file, "w") as f:
        json.dump(tx_cache, f, indent=2)
        
    print(f"\nFinal: resolved {len(renamed_resolved)} / {len(all_raw_ids)} variants.")

if __name__ == "__main__":
    main()
