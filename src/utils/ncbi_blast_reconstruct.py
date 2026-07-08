import os
import sys
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import time
from Bio.Seq import Seq
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set project root path
sys.path.append("/home/work/workdir/deepprime6-genomebiol-revision")

def submit_blast_job(fasta_string):
    url = 'https://blast.ncbi.nlm.nih.gov/Blast.cgi'
    params = {
        'CMD': 'Put',
        'PROGRAM': 'blastn',
        'DATABASE': 'human',
        'QUERY': fasta_string,
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
    print("Parsing BLAST XML results...")
    root = ET.fromstring(xml_content)
    align_map = {}
    
    # Map NCBI NC_ accessions to chromosome names
    def get_chrom_name(accession):
        # e.g. NC_000003.12 -> chr3
        if 'NC_0000' in accession:
            try:
                num_part = accession.split('.')[0].split('_')[1]
                num = int(num_part)
                if num == 23:
                    return 'chrX'
                elif num == 24:
                    return 'chrY'
                elif 1 <= num <= 22:
                    return f"chr{num}"
            except Exception:
                pass
        elif 'NC_012920' in accession:
            return 'chrM'
        return 'chr1'
        
    for iteration in root.findall('.//Iteration'):
        query_def = iteration.find('Iteration_query-def').text
        aid = query_def.replace('seq_', '')
        
        hits = iteration.find('.//Iteration_hits')
        if hits is not None and len(hits) > 0:
            best_hit = hits.find('Hit')
            hit_id = best_hit.find('Hit_id').text
            # Accession is inside hit_id, e.g. ref|NC_000003.12|
            accession = ''
            for part in hit_id.split('|'):
                if part.startswith('NC_'):
                    accession = part
                    break
                    
            chrom = get_chrom_name(accession)
            hsp = best_hit.find('.//Hsp')
            h_from = int(hsp.find('Hsp_hit-from').text)
            h_to = int(hsp.find('Hsp_hit-to').text)
            
            strand = '+'
            if h_from > h_to:
                strand = '-'
                t_start = h_to
                t_end = h_from
            else:
                t_start = h_from
                t_end = h_to
                
            align_map[aid] = {
                'chr': chrom,
                'strand': strand,
                'tStart': t_start,
                'tEnd': t_end
            }
    return align_map

def fetch_genomic_sequence(chrom, start, stop, cache):
    key = f"{chrom}:{start}-{stop}"
    if key in cache:
        return cache[key]
        
    # Respectful rate-limiting
    time.sleep(0.5)
    url = f'https://genome.ucsc.edu/cgi-bin/das/hg38/dna?segment={chrom}:{start},{stop}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=15)
        xml_data = res.read().decode()
        
        root = ET.fromstring(xml_data)
        dna_element = root.find('.//DNA')
        if dna_element is not None and dna_element.text:
            dna_seq = ''.join(dna_element.text.split()).upper()
            cache[key] = dna_seq
            return dna_seq
    except Exception as e:
        print(f"Error fetching DAS sequence {key}: {e}")
    return None

def align_and_reconstruct(row, align_meta, genomic_seq):
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
        # Check reverse complement
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
        'chr': align_meta['chr'],
        'strand': '+' if (align_meta['strand'] == '+' if fwd_strand else align_meta['strand'] == '-') else '-'
    }

def main():
    print("Loading pegRNA CSV dataset...")
    df = pd.read_csv("data/MFE_randompeg_RHA30_0.71M_result.csv")
    df['clean_id'] = df['ID'].astype(str).str.replace('clinic_', '').str.strip()
    
    unique_ids = df['clean_id'].unique()
    print(f"Total unique variant IDs: {len(unique_ids)}")
    
    # Load or initialize genomic sequence cache
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
    
    if unresolved_ids:
        # Build FASTA string for BLAST
        fasta_lines = []
        for aid in unresolved_ids:
            subset = df[df['clean_id'] == aid]
            row = subset.iloc[0]
            pbs = row['PBS_pegRNA_DNA'].upper()
            rc_pbs = str(Seq(pbs).reverse_complement())
            rtt_peg = row['RTT_template_DNA'].upper()
            rtt = str(Seq(rtt_peg).reverse_complement())
            active_seq = rc_pbs + rtt
            fasta_lines.append(f">seq_{aid}\n{active_seq}")
            
        fasta_string = '\n'.join(fasta_lines)
        
        # Submit BLAST job
        print(f"Submitting {len(unresolved_ids)} sequences to NCBI BLAST...")
        rid, rtoe = submit_blast_job(fasta_string)
        if not rid:
            print("BLAST submission failed.")
            return
            
        # Wait for job completion
        print(f"Waiting {rtoe}s for BLAST to finish...")
        time.sleep(rtoe)
        
        while True:
            status = check_blast_status(rid)
            print(f"BLAST Job Status: {status}")
            if status == 'READY':
                break
            elif status in ['FAILED', 'UNKNOWN', 'ERROR']:
                print("NCBI BLAST job failed.")
                return
            time.sleep(10)
            
        # Retrieve and parse BLAST XML
        xml_results = retrieve_blast_results(rid)
        if not xml_results:
            print("Failed to retrieve BLAST XML results.")
            return
            
        align_map = parse_blast_xml(xml_results)
        print(f"Successfully mapped {len(align_map)} variants to genomic coordinates.")
        
        # Fetch genomic sequences and align
        print("Fetching flanking genomic sequences and reconstructing variants...")
        # Use a small thread pool of size 3 to respect UCSC rate limits and avoid 429
        with ThreadPoolExecutor(max_workers=3) as executor:
            tasks = []
            for aid, align_meta in align_map.items():
                subset = df[df['clean_id'] == aid]
                rep_row = subset.iloc[0]
                chrom = align_meta['chr']
                start_coord = align_meta['tStart'] - 150
                stop_coord = align_meta['tEnd'] + 150
                tasks.append((aid, rep_row, align_meta, chrom, start_coord, stop_coord))
                
            future_to_aid = {}
            for aid, rep_row, align_meta, chrom, start_coord, stop_coord in tasks:
                # We fetch sequence and align sequentially inside the executor's task
                def task_fn(a=aid, r=rep_row, m=align_meta, c=chrom, s=start_coord, e=stop_coord):
                    seq_str = fetch_genomic_sequence(c, s, e, cache)
                    if not seq_str:
                        return a, None
                    res = align_and_reconstruct(r, m, seq_str)
                    return a, res
                future_to_aid[executor.submit(task_fn)] = aid
                
            completed = 0
            success = 0
            for future in as_completed(future_to_aid):
                completed += 1
                aid, result = future.result()
                if result:
                    success += 1
                    resolved_details[aid] = result
                if completed % 20 == 0 or completed == len(tasks):
                    print(f"Aligned {completed}/{len(tasks)} variants. Success rate: {success}/{completed} ({success/completed*100:.1f}%)")
                    # Save incremental results
                    with open(resolved_details_file, "w") as f:
                        json.dump(resolved_details, f, indent=2)
                    with open(cache_file, "w") as f:
                        json.dump(cache, f, indent=2)
                        
    print(f"Total variants resolved and reconstructed: {len(resolved_details)} / {len(unique_ids)} ({len(resolved_details)/len(unique_ids)*100:.1f}%)")
    with open(resolved_details_file, "w") as f:
        json.dump(resolved_details, f, indent=2)

if __name__ == "__main__":
    main()
