# Pipeline Process Flowchart

This flowchart visualizes the sequence of operations, data flows, and models invoked by the end-to-end pegRNA prediction and candidate selection pipeline.

```mermaid
%%{init: { 'theme': 'base', 'themeVariables': { 'clusterBkg': '#F4F4F4', 'clusterBorder': '#CCCCCC' }}}%%
graph TD
    %% Define Styles
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    %% Nodes
    A1["MFE_randompeg_RHA30_0.71M_result.csv"]:::input
    A2["resolved_pegrna_genomic_details.json"]:::input
  
    subgraph Step1 ["Step 1: Feature Preprocessing (CPU)"]
        B["generate_prediction_parquet.py"]:::process
        B1["Reconstruct 80nt Target Context"]:::process
        B2["32-Core Parallel Tm/MFE calculations"]:::process
        B3["preprocessed_pegrna_prediction.parquet"]:::input
    end

    subgraph Step2 ["Step 2 & 3: DeepPrime Models (GPU)"]
        C1["DeepPrime (Base) model"]:::model
        C2["DeepPrime6 PE6a model"]:::model
        C3["DeepPrime6 PE6b model"]:::model
        C4["DeepPrime6 PE6c model"]:::model
        C5["DeepPrime6 PEmax-dRNaseH model"]:::model
  
        D1["pegrna_predictions.csv"]:::input
        D2["predictions_pe6a.csv"]:::input
        D3["predictions_pe6b.csv"]:::input
        D4["predictions_pe6c.csv"]:::input
        D5["predictions_pemaxdrnaseh.csv"]:::input
    end

    subgraph Step4 ["Step 4: PRIDICT2.0 Model (GPU)"]
        E["predict_pridict2_library.py"]:::process
        E1["100k Chunk Loop with Checkpoint Caching"]:::process
        E2["predictions_pridict2.csv"]:::input
    end

    subgraph Step5 ["Step 5: Candidate Selection & Validation"]
        F["select_candidates.py"]:::process
        F1["Percentile Ranking (0 to 1) across Library"]:::process
        F2["Categorize into terciles: Inefficient, Modest, Highly Efficient"]:::process
        F3["Compute Disagreement Score (SD of Percentiles)"]:::process
        F4["Enforce ID Locus Uniqueness"]:::process
  
        G1["Pearson Correlation Verification (< 0.9)"]:::process
        H["selected_50_validation_candidates.csv"]:::output
    end

    %% Flow Connections
    A1 --> B
    A2 --> B
    B --> B1
    B1 --> B2
    B2 --> B3
  
    B3 --> C1
    B3 --> C2
    B3 --> C3
    B3 --> C4
    B3 --> C5
  
    C1 --> D1
    C2 --> D2
    C3 --> D3
    C4 --> D4
    C5 --> D5
  
    A1 --> E
    A2 --> E
    E --> E1
    E1 --> E2
  
    D1 --> F
    D2 --> F
    D3 --> F
    D4 --> F
    D5 --> F
    E2 --> F
  
    F --> F1
    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> G1
    G1 --> H

---

## 📝 Technical Implementation Details & Design Rationale

### 1. Flanking Genomic Context Window Resolution (80nt vs. 200nt)
- **Problem**: DeepPrime requires an **80nt target context** (`WideTargetSequence`), while PRIDICT2.0 requires a **200nt target context** (`wt_pridict_200`, `ed_pridict_200`) to extract structural features and compute MFE.
- **Solution**: 
  - The offline BLAST target reconstruction step fetches the full **200nt sequence** around the nick site and caches it in [resolved_pegrna_genomic_details.json](file:///home/work/workdir/deepprime6-genomebiol-revision/data/resolved_pegrna_genomic_details.json).
  - In **Step 1 (DeepPrime Preprocessing)**, [generate_prediction_parquet.py](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/generate_prediction_parquet.py) slices this 200nt sequence to extract the central 80nt window (`WideTargetSequence = wt_pridict_200[75:155]`), keeping the nick site exactly at index 25.
  - In **Step 4 (PRIDICT2.0)**, [predict_pridict2_library.py](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/predict_pridict2_library.py) maps the full **200nt sequence** directly from the JSON cache without slicing, ensuring both models receive their native correct-length flanking contexts.

### 2. Multi-Model Signal Aggregation
- **Problem**: The pipeline generates 6 individual scores (DeepPrime-Base, 4 DeepPrime6 checkpoints, and 2 PRIDICT2.0 cell-line scores). Using 6 raw signals directly for standard deviation calculations would bias the disagreement score towards DeepPrime.
- **Solution**:
  - The 6 raw signals are aggregated into **3 representative model scores** in [select_candidates.py](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/select_candidates.py):
    1. **DeepPrime Base**: `Score_DP_Base`
    2. **DeepPrime6 Avg**: Mean of PE6a, PE6b, PE6c, and PEmax-dRNaseH scores.
    3. **PRIDICT2.0 Avg**: Mean of HEK and K562 prediction scores.
  - These 3 representative scores are converted to percentile ranks (0 to 1) across the library, and the **Disagreement Score** is calculated as the standard deviation of these **3 percentile ranks**.

### 3. Locus Uniqueness Enforcement (Locus Uniqueness)
- **Problem**: Why is locus uniqueness enforced at the very end (Step 5) instead of Step 1?
- **Solution**: 
  - The 678,084 rows in the preprocessed Parquet represent **biologically unique pegRNA designs** (varying in spacer length, PBS length, and RTT length) targeting the 653 unique genomic variants. There are no duplicate rows in the dataset.
  - To find the **best possible pegRNA design** for each variant, we must score and rank all 678,084 designs.
  - Locus uniqueness is enforced during candidate selection to ensure the final 50 validation candidates cover **50 distinct genomic variant loci**, preventing the selection of multiple pegRNA designs for the same variant.

### 4. Tercile Partitioning & Candidate Distribution
- **Problem**: Choice of Tercile-based partitioning vs. absolute score thresholds and candidate allocation ratios.
- **Solution**:
  - Absolute model scores are not calibrated between DeepPrime (scale 0-100) and PRIDICT2.0 (scale 0-100 but different variance). Percentile normalization standardizes the scales to a uniform $[0, 1]$ interval.
  - The library is partitioned into equal **Terciles** (Highly Efficient: pct > 0.667, Modest: 0.333 < pct <= 0.667, Inefficient: pct <= 0.333).
  - The 50 candidates are distributed as **15 / 20 / 15** (Consensus/Disagreement as 10/5, 14/6, 10/5) to balance consensus validation and high-disagreement outlier evaluation.
  - *Note: These thresholds and distribution ratios are pending final PI confirmation upon reviewing the generated candidate list.*

```
