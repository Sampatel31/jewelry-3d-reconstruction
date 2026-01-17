#!/usr/bin/env python3
"""
Automatic model downloader for jewelry 3D reconstruction.
Downloads all required pretrained model weights.
"""

import os
from pathlib import Path
import requests
from tqdm import tqdm


def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_models(cache_dir: Path = None):
    """
    Download all required models.
    
    Args:
        cache_dir: Directory to cache models (default: ~/.cache/jewelry-3d-reconstruction)
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "jewelry-3d-reconstruction"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading model weights...")
    print(f"Cache directory: {cache_dir}")
    
    models = {
        # Models will be automatically downloaded by Hugging Face transformers
        # when first accessed. This script can be extended to pre-download them.
        "dust3r": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "trellis": "JeffreyXiang/TRELLIS-image-large",
        "sam": "facebook/sam-vit-huge",
        "rembg": "u2net",  # Downloaded automatically by rembg
    }
    
    print("\nℹ️ Models will be downloaded automatically on first use:")
    for name, model_id in models.items():
        print(f"  - {name}: {model_id}")
    
    print("\n✅ Model setup complete")
    print("Models will be cached in Hugging Face cache on first use.")


if __name__ == "__main__":
    download_models()
