"""Download ATP match CSV files from TML-Database (GitHub)."""

import subprocess
import shutil
from pathlib import Path

from src.config import MATCH_YEARS, RAW_DIR

TML_REPO_URL = "https://github.com/Tennismylife/TML-Database.git"
TML_CLONE_DIR = Path("/tmp/TML-Database")


def download_atp_data(
    years: range = MATCH_YEARS,
    dest: Path = RAW_DIR,
) -> None:
    """Fetch ATP match CSVs from Tennismylife/TML-Database.

    TML-Database uses Sackmann-compatible schema (same columns) and
    is live-updated through the current season including 2026.
    """
    dest.mkdir(parents=True, exist_ok=True)

    # Check if we already have all files
    existing = sum(1 for y in years if (dest / f"atp_matches_{y}.csv").exists())
    if existing == len(list(years)):
        print(f"all {existing} files already cached in {dest}")
        return

    # Clone repo if not already present
    if not TML_CLONE_DIR.exists():
        print(f"cloning TML-Database to {TML_CLONE_DIR}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", TML_REPO_URL, str(TML_CLONE_DIR)],
            check=True,
            capture_output=True,
        )
    else:
        # Pull latest
        print("updating TML-Database...")
        subprocess.run(
            ["git", "-C", str(TML_CLONE_DIR), "pull", "--ff-only"],
            capture_output=True,
        )

    # Copy match files
    copied = 0
    for year in years:
        src_file = TML_CLONE_DIR / f"{year}.csv"
        dst_file = dest / f"atp_matches_{year}.csv"
        if src_file.exists() and not dst_file.exists():
            shutil.copy2(src_file, dst_file)
            copied += 1

    total = len(list(years))
    cached = total - copied
    print(f"done: {copied} copied, {cached} cached ({total} total)")


if __name__ == "__main__":
    print("downloading ATP data from TML-Database...")
    download_atp_data()
