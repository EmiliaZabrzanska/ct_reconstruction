"""
Download DRUNet pretrained weights for grayscale denoising.

DRUNet is used as the off-the-shelf denoiser in TFPnP's plug-and-play ADMM.
"""

import urllib.request
from pathlib import Path

# URL of pretrained DRUNet weights
DRUNET_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth"

# set repo root and destination path
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "results" / "baselines" / "drunet_gray.pth"

# set expected size for error checking
EXPECTED_SIZE_MB = 125  # ~130 MB on disk

def main() -> None:

    # make destination dir 
    DEST.parent.mkdir(parents=True, exist_ok=True)
    
    if DEST.exists():

        # check file size to avoid redownloading
        size_mb = DEST.stat().st_size / 1e6
        
        print(f"Already present: {DEST} ({size_mb:.1f} MB)")
        return

    print(f"Downloading DRUNet (grayscale) from {DRUNET_URL}")

    # download from URL and save
    urllib.request.urlretrieve(DRUNET_URL, DEST)
    size_mb = DEST.stat().st_size / 1e6
    print(f"Saved to {DEST} ({size_mb:.1f} MB)")

    # warn if unexpected size
    if abs(size_mb - EXPECTED_SIZE_MB) > 20:
        print(f"WARNING: unexpected size (expected ~{EXPECTED_SIZE_MB} MB)")

if __name__ == "__main__":
    main()