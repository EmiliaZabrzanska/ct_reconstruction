"""Download DRUNet pretrained weights for grayscale denoising.

DRUNet is used as the off-the-shelf denoiser in TFPnP's plug-and-play ADMM.
Pretrained on BSD400 by Zhang et al. (KAIR repository).

Source: Zhang, K., Zuo, W., Zhang, L. "Plug-and-Play Image Restoration with
Deep Denoiser Prior". TPAMI 2021.
KAIR repo: https://github.com/cszn/KAIR
"""

import urllib.request
from pathlib import Path

DRUNET_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "results" / "baselines" / "drunet_gray.pth"
EXPECTED_SIZE_MB = 125  # ~130 MB on disk

def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        size_mb = DEST.stat().st_size / 1e6
        print(f"Already present: {DEST} ({size_mb:.1f} MB)")
        return

    print(f"Downloading DRUNet (grayscale) from {DRUNET_URL}")
    urllib.request.urlretrieve(DRUNET_URL, DEST)
    size_mb = DEST.stat().st_size / 1e6
    print(f"Saved to {DEST} ({size_mb:.1f} MB)")
    if abs(size_mb - EXPECTED_SIZE_MB) > 20:
        print(f"WARNING: unexpected size (expected ~{EXPECTED_SIZE_MB} MB)")

if __name__ == "__main__":
    main()