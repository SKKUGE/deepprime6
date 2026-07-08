import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory for input data")
    parser.add_argument("--output-dir", default="data/predictions", help="Directory for output data")
    parser.add_argument("--image-out", default="scripts/supp3_pipeline/predictions_correlation.png", help="Path to save output image")
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    image_out = args.image_out

    print("Loading datasets for plotting...")
    raw = pd.read_csv(os.path.join(data_dir, "MFE_randompeg_RHA30_0.71M_result.csv"))
    dp = pd.read_csv(os.path.join(output_dir, "pegrna_predictions.csv"))
    pridict = pd.read_csv(os.path.join(output_dir, "predictions_pridict2.csv"))
    candidates = pd.read_csv(os.path.join(output_dir, "selected_50_validation_candidates.csv"))

    # Map percentile ranks to raw dataset using matching indices
    print("Computing percentile ranks for the full library...")
    raw['Percentile_DP'] = dp.set_index('index')['Prediction'].rank(pct=True)
    raw['Percentile_PRIDICT'] = pridict.set_index('index')['PRIDICT2_Score_HEK'].rank(pct=True)

    plt.figure(figsize=(10, 8), dpi=150)

    # 1. Plot density hexbin for the 678,084 designs
    print("Plotting library density map...")
    plt.hexbin(
        raw['Percentile_DP'], 
        raw['Percentile_PRIDICT'], 
        gridsize=80, 
        cmap='Blues', 
        mincnt=1, 
        alpha=0.5, 
        edgecolors='none'
    )
    cb = plt.colorbar(label='Density of pegRNA library designs')

    # 2. Overlay the selected 50 candidates colored by category
    print("Overlaying selected 50 validation candidates...")
    colors = {
        'Highly Efficient': '#2ecc71', 
        'Modest': '#f39c12', 
        'Inefficient': '#e74c3c'
    }
    for cat, col in colors.items():
        sub = candidates[candidates['Category'] == cat]
        plt.scatter(
            sub['Percentile_DP6'], 
            sub['Percentile_PRIDICT2'], 
            color=col, 
            s=80, 
            label=f'Selected {cat}', 
            edgecolors='black', 
            linewidths=1.2, 
            zorder=5
        )

    plt.xlabel('DeepPrime6 Percentile Rank', fontsize=12)
    plt.ylabel('PRIDICT2.0 Percentile Rank', fontsize=12)
    plt.title('Correlation Analysis & Validation Candidate Distribution\nDeepPrime6 vs. PRIDICT2.0 (Pearson r = 0.171)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(image_out), exist_ok=True)
    plt.savefig(image_out, bbox_inches='tight')
    print('Plot successfully saved to:', image_out)

if __name__ == "__main__":
    main()
