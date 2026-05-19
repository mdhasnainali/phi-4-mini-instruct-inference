#!/usr/bin/env python3
"""
Download script for Phi-4-mini-instruct GGUF model from HuggingFace
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface-hub...")
    os.system(f"{sys.executable} -m pip install huggingface-hub")
    from huggingface_hub import hf_hub_download


def download_model(
    repo_id: str = "unsloth/Phi-4-mini-instruct-GGUF",
    filename: str = "Phi-4-mini-instruct-Q4_K_M.gguf",
    local_dir: str = "models"
):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {filename} from {repo_id}...")
    print(f"Saving to: {local_dir}")

    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"\nModel downloaded successfully!")
        print(f"Location: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Phi-4 GGUF model")
    parser.add_argument("--filename", type=str, default="Phi-4-mini-instruct-Q4_K_M.gguf",
                        help="GGUF filename to download")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Output directory for model")
    args = parser.parse_args()

    download_model(filename=args.filename, local_dir=args.output_dir)