import os
import sys
import zipfile

def find_project_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".project-root")) or (
            os.path.exists(os.path.join(curr, "src")) and os.path.exists(os.path.join(curr, "configs"))
        ):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.getcwd())

PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)

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
        "notebooks/plots/small_library_pe6to7",
        "results/small_library_pe6to7_archive.zip"
    )
    make_zip(
        "notebooks/plots/frt_small_library",
        "results/frt_small_library_results.zip"
    )
    make_zip(
        "notebooks/plots",
        "results/inhouse_orthogonal_benchmark_archive.zip"
    )
    make_zip(
        "notebooks/plots/orthogonal_benchmark",
        "results/orthogonal_benchmark_results.zip"
    )
    make_zip(
        "notebooks/plots/endogenous_site_evaluation",
        "results/endogenous_site_evaluation_results.zip"
    )

