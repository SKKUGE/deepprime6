import pandas as pd
import torch
import sys
import argparse
from omegaconf import OmegaConf

# Import our project modules
from src.data.pe6_datamodule import PE6DataModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", default="data/predictions/preprocessed_pegrna_prediction.parquet", help="Path to preprocessed Parquet file")
    args = parser.parse_args()
    parquet_path = args.parquet_path

    print(f"Instantiating datamodule using Parquet: {parquet_path}")
    dm = PE6DataModule(
        data_dir="data",
        batch_size=512,
        skip_preprocessing=True,
        csv_path=parquet_path,
        datafilter={"PE_types": ["PE6a(+PEmaxCas9)"]},
        dataset=OmegaConf.create({
            "norm_mean_path": "src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/mean.csv",
            "norm_std_path": "src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/std.csv"
        })
    )
    
    print("Setting up predict stage...")
    dm.setup(stage="predict")
    
    dataset = dm.data_predict
    print("Success! Dataset loaded.")
    print("Total prediction samples:", len(dataset))
    
    # Load first sample
    ((g, b), label), annot = dataset[0]
    print("\nSample 0 details:")
    print("Genetic feature shape (g):", g.shape)
    print("Biofeatures shape (b):", b.shape)
    print("Label shape:", label.shape)
    print("Annotation (ID):", annot)

if __name__ == "__main__":
    main()
