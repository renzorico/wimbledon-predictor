"""Clean and preprocess raw match data."""

import re

import pandas as pd

from src.config import PROCESSED_DIR


# Columns that should be numeric but arrive as object due to NaNs
STAT_COLS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]


def _is_completed_match(score: str) -> bool:
    """Return True if the match was played to completion (no W/O, RET, DEF)."""
    if not isinstance(score, str):
        return False
    flags = ["W/O", "RET", "DEF", "ABN", "UNP", "Walkover"]
    return not any(f.lower() in score.lower() for f in flags)


def _count_sets(score: str) -> int | None:
    """Count completed sets from a score string like '6-4 6-7(5) 7-5'."""
    if not isinstance(score, str):
        return None
    sets = re.findall(r"\d+-\d+", score)
    return len(sets) if sets else None


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Clean raw match DataFrame. Returns copy."""
    df = matches.copy()

    # Drop walkovers and retirements
    has_score = df["score"].notna()
    completed = df["score"].apply(_is_completed_match)
    df = df[has_score & completed].reset_index(drop=True)

    # Deduplicate
    df = df.drop_duplicates(
        subset=["tourney_id", "match_num"], keep="first"
    ).reset_index(drop=True)

    # Cast stat columns to numeric
    for col in STAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived serve/return percentages (winner side)
    df["w_serve_pct"] = (df["w_1stWon"] + df["w_2ndWon"]) / df["w_svpt"]
    df["w_1st_in_pct"] = df["w_1stIn"] / df["w_svpt"]
    df["w_1st_won_pct"] = df["w_1stWon"] / df["w_1stIn"]
    df["w_2nd_won_pct"] = df["w_2ndWon"] / (df["w_svpt"] - df["w_1stIn"])
    df["w_bp_saved_pct"] = df["w_bpSaved"] / df["w_bpFaced"]
    df["w_ace_rate"] = df["w_ace"] / df["w_SvGms"]
    df["w_df_rate"] = df["w_df"] / df["w_SvGms"]

    # Derived serve/return percentages (loser side)
    df["l_serve_pct"] = (df["l_1stWon"] + df["l_2ndWon"]) / df["l_svpt"]
    df["l_1st_in_pct"] = df["l_1stIn"] / df["l_svpt"]
    df["l_1st_won_pct"] = df["l_1stWon"] / df["l_1stIn"]
    df["l_2nd_won_pct"] = df["l_2ndWon"] / (df["l_svpt"] - df["l_1stIn"])
    df["l_bp_saved_pct"] = df["l_bpSaved"] / df["l_bpFaced"]
    df["l_ace_rate"] = df["l_ace"] / df["l_SvGms"]
    df["l_df_rate"] = df["l_df"] / df["l_SvGms"]

    # Return points won = 1 - opponent serve pct
    df["w_return_pct"] = 1 - df["l_serve_pct"]
    df["l_return_pct"] = 1 - df["w_serve_pct"]

    # Number of sets
    df["n_sets"] = df["score"].apply(_count_sets)

    return df


def save_clean_matches(df: pd.DataFrame) -> None:
    """Save cleaned DataFrame to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "matches_clean.parquet"
    df.to_parquet(path, index=False)
    print(f"saved {len(df):,} matches to {path}")


if __name__ == "__main__":
    from src.data.loader import load_matches

    print("loading raw matches...")
    raw = load_matches()
    print(f"  raw: {len(raw):,} rows")

    print("cleaning...")
    clean = clean_matches(raw)
    print(f"  clean: {len(clean):,} rows")

    save_clean_matches(clean)
