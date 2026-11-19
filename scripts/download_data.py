"""Download the US file of the public "Trending YouTube Video Statistics" dataset.

Usage:  python scripts/download_data.py
It saves data/USvideos.csv (about 60 MB) and data/US_category_id.json.
Original dataset: https://www.kaggle.com/datasets/datasnaek/youtube-new (CC0).
This script uses a public GitHub mirror so no Kaggle account is needed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MIRROR = "https://github.com/piyush033/Trending-Youtube-Video-Statistics.git"
FILES = ["USvideos.csv", "US_category_id.json"]


def main() -> None:
    dest = Path(__file__).resolve().parents[1] / "data"
    dest.mkdir(exist_ok=True)
    if all((dest / f).exists() for f in FILES):
        print("Data already downloaded:", dest)
        return
    with tempfile.TemporaryDirectory() as tmp:
        print("Cloning mirror (this can take a minute)...")
        subprocess.run(["git", "clone", "--depth", "1", MIRROR, tmp], check=True)
        for f in FILES:
            src = next(Path(tmp).rglob(f), None)
            if src is None:
                sys.exit(f"Could not find {f} in the mirror. Download it from Kaggle instead.")
            shutil.copy(src, dest / f)
            print("Saved", dest / f)


if __name__ == "__main__":
    main()
