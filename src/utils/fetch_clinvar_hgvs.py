import urllib.request
import urllib.parse
import json
import pandas as pd
import time
import os

def fetch_batch_summaries(allele_ids):
    term = ' OR '.join([f'{aid}[AlleleID]' for aid in allele_ids])
    search_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=clinvar&term={}&retmax=500&retmode=json'.format(urllib.parse.quote(term))
    
    try:
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        search_data = json.loads(res.read().decode())
        id_list = search_data['esearchresult']['idlist']
        if not id_list:
            return {}
            
        summary_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=clinvar&id={}&retmode=json'.format(','.join(id_list))
        req2 = urllib.request.Request(summary_url, headers={'User-Agent': 'Mozilla/5.0'})
        res2 = urllib.request.urlopen(req2)
        summary_data = json.loads(res2.read().decode())
        
        results = summary_data['result']
        batch_mapping = {}
        for uid in id_list:
            if uid in results and uid != 'uids':
                res_dict = results[uid]
                title = res_dict.get('title', '')
                genes = res_dict.get('genes', [])
                gene_symbol = genes[0].get('symbol', '') if genes else ''
                
                # Check variation_set for measure_id (our Allele ID)
                var_set = res_dict.get('variation_set', [])
                if var_set and len(var_set) > 0:
                    measure_id = var_set[0].get('measure_id', '')
                    variant_type = var_set[0].get('variant_type', '')
                    cdna_change = var_set[0].get('cdna_change', '')
                    
                    # Extract genomic location (chr, start, stop) on GRCh38
                    chrom = ''
                    start = 0
                    stop = 0
                    var_locs = var_set[0].get('variation_loc', [])
                    for loc in var_locs:
                        if loc.get('assembly_name') == 'GRCh38':
                            chrom = loc.get('chr', '')
                            try:
                                start = int(loc.get('start', 0))
                                stop = int(loc.get('stop', 0))
                            except ValueError:
                                pass
                            break
                    
                    if measure_id in allele_ids:
                        batch_mapping[measure_id] = {
                            'title': title,
                            'gene': gene_symbol,
                            'variant_type': variant_type,
                            'cdna_change': cdna_change,
                            'chr': chrom,
                            'start': start,
                            'stop': stop
                        }
        return batch_mapping
    except Exception as e:
        print(f"Error fetching batch: {e}")
        import traceback
        traceback.print_exc()
        return {}

def main():
    print("Loading unique IDs from CSV...")
    df = pd.read_csv("data/MFE_randompeg_RHA30_0.71M_result.csv")
    raw_ids = df['ID'].astype(str).str.strip().unique()
    print(f"Total unique raw IDs in CSV: {len(raw_ids)}")
    
    # Create a mapping from clean numerical ID to a list of original raw IDs
    clean_to_raw = {}
    for rid in raw_ids:
        clean_id = rid.replace('clinic_', '').strip()
        if clean_id not in clean_to_raw:
            clean_to_raw[clean_id] = []
        clean_to_raw[clean_id].append(rid)
        
    # Exclude ambiguous IDs where the same numerical ID is shared by multiple raw designs
    # (since they target different loci, ClinVar lookup by ID is guaranteed to be wrong for at least one)
    allele_ids = []
    for clean_id, raw_list in clean_to_raw.items():
        if len(raw_list) == 1:
            allele_ids.append(clean_id)
        else:
            print(f"[*] Excluding ambiguous ID from query/fallback: {clean_id} maps to {raw_list}")
            
    mapping = {}
    
    batch_size = 100
    total_ids = len(allele_ids)
    print(f"Starting NCBI ClinVar queries for {total_ids} clean Allele IDs...")
    
    for i in range(0, total_ids, batch_size):
        batch = allele_ids[i : i + batch_size]
        print(f"Fetching batch {i // batch_size + 1} ({i} to {i + len(batch)})...")
        batch_map = fetch_batch_summaries(batch)
        
        # Merge results, mapping them back to all original raw IDs
        for clean_id, meta in batch_map.items():
            if clean_id in clean_to_raw:
                for raw_id in clean_to_raw[clean_id]:
                    mapping[raw_id] = meta
            
        time.sleep(1.0) # NCBI API rate limit politeness
        
    print(f"Successfully resolved {len(mapping)} / {len(raw_ids)} raw IDs.")
    
    # Check which IDs are missing
    missing_ids = [rid for rid in raw_ids if rid not in mapping]
    print(f"Missing IDs count: {len(missing_ids)}")
    
    os.makedirs("data/ncbi_cache", exist_ok=True)
    with open("data/ncbi_cache/clinvar_hgvs_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print("Saved mapping to data/ncbi_cache/clinvar_hgvs_mapping.json")

if __name__ == "__main__":
    main()
