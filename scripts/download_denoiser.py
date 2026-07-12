"""
Download the pretrained DRUNet weights used as the off-the-shelf denoiser prior.

drunet_gray.pth is KAIR's grayscale DRUNet, trained on natural images with noise
levels sampled from [1, 50] in 8-bit units. It is the plug-and-play prior in the
ADMM x-step.

Usage:
    python scripts/download_denoiser.py
"""

import urllib.request
from pathlib import Path

# URL of pretrained DRUNet weights
DRUNET_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth"

# set repo root and destination path
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "results" / "baselines" / "drunet_gray.pth"

# set expected size for error checking
EXPECTED_SIZE_MB = 125 
SIZE_TOLERANCE_MB = 20

def main() -> None:
    """
    Download drunet_gray.pth to results/baselines/, skipping if already present.
    """
    # make destination dir 
    DEST.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip the download if the file is already there
    if DEST.exists():
        
        size_mb = DEST.stat().st_size / 1e6
        print(f"Already present: {DEST} ({size_mb:.1f} MB)")

        return

    print(f"Downloading DRUNet (grayscale) from {DRUNET_URL}")

    # download from URL and save
    urllib.request.urlretrieve(DRUNET_URL, DEST)
    size_mb = DEST.stat().st_size / 1e6
    print(f"Saved to {DEST} ({size_mb:.1f} MB)")

    # warn if unexpected size
    if abs(size_mb - EXPECTED_SIZE_MB) > SIZE_TOLERANCE_MB:
        print(f"WARNING: unexpected size (expected ~{EXPECTED_SIZE_MB} MB)")

if __name__ == "__main__":
    main()