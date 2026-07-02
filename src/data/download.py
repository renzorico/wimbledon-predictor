"""Download ATP match CSV files from TML-Database (GitHub)."""

import subprocess
import shutil
from pathlib import Path

import requests

from src.config import MATCH_YEARS, RAW_DIR

TML_REPO_URL = "https://github.com/Tennismylife/TML-Database.git"
TML_CLONE_DIR = Path("/tmp/TML-Database")
TML_STATS_BASE_URL = "https://stats.tennismylife.org/data"


def _download_year_from_tml_stats(year: int, dst_file: Path) -> bool:
    """Download a yearly ATP CSV from the live TennisMyLife site."""
    url = f"{TML_STATS_BASE_URL}/{year}.csv"
    try:
        response = requests.get(url, timeout=60)
    except requests.RequestException:
        return False
    if response.status_code != 200:
        return False
    dst_file.write_bytes(response.content)
    return True


def download_atp_data(
    years: range = MATCH_YEARS,
    dest: Path = RAW_DIR,
    refresh_existing: bool = True,
) -> None:
    """Fetch ATP match CSVs from Tennismylife/TML-Database.

    TML-Database uses Sackmann-compatible schema (same columns) and
    is live-updated through the current season including 2026.
    """
    dest.mkdir(parents=True, exist_ok=True)

    # First try the live TennisMyLife downloads.
    copied = 0
    downloaded_live: set[int] = set()
    for year in years:
        dst_file = dest / f"atp_matches_{year}.csv"
        if _download_year_from_tml_stats(year, dst_file):
            copied += 1
            downloaded_live.add(year)

    if copied == len(list(years)):
        print(f"done: {copied} refreshed from TennisMyLife ({len(list(years))} total)")
        return

    # Fallback to the historical GitHub mirror for anything missing.
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
    for year in years:
        if year in downloaded_live:
            continue
        src_file = TML_CLONE_DIR / f"{year}.csv"
        dst_file = dest / f"atp_matches_{year}.csv"
        if not src_file.exists():
            continue
        if refresh_existing or not dst_file.exists():
            shutil.copy2(src_file, dst_file)
            copied += 1

    total = len(list(years))
    print(f"done: {copied} refreshed from TennisMyLife/GitHub fallback ({total} total)")


if __name__ == "__main__":
    print("downloading ATP data from TML-Database...")
    download_atp_data()
