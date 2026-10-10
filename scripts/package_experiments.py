import os
import zipfile

def make_zip(dir_to_zip, zip_path, exclude_exts=('.pt', '.ckpt', '.parquet')):
    if not os.path.isdir(dir_to_zip):
        print(f"Skipping {dir_to_zip}: directory does not exist.")
        return
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dir_to_zip):
            for file in files:
                if any(file.endswith(ext) for ext in exclude_exts) or file.endswith('.zip'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(dir_to_zip))
                zipf.write(file_path, arcname)
    print(f"Created {zip_path} ({os.path.getsize(zip_path):,} bytes)")

if __name__ == "__main__":
    make_zip(
        "experiments/20261009_small_library_pe6to7",
        "experiments/20261009_small_library_pe6to7/small_library_pe6to7_archive.zip"
    )
    make_zip(
        "experiments/20261009_frt_small_library",
        "experiments/20261009_frt_small_library/frt_small_library_results.zip"
    )
    make_zip(
        "experiments/20261008_panelB_base_architecture_revalidation",
        "experiments/20261008_panelB_base_architecture_revalidation/inhouse_orthogonal_benchmark_archive.zip"
    )
    make_zip(
        "experiments/20261009_orthogonal_benchmark_leakage_stratification",
        "experiments/20261009_orthogonal_benchmark_leakage_stratification/orthogonal_benchmark_results.zip"
    )
