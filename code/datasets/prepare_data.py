"""
Stage 1: Data Engineering
Loading -> cleaning -> target generation -> splitting into train/test

"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/atp_tennis.csv")
PROCESSED_DIR = Path("data/processed")

# In order for pipeline to be able to fully execute in 5 minutes
# I will limit the range of year
MIN_YEAR = 2015

# Columns, where spaces are set as -1, not as NaN
MISSING_AS_MINUS_ONE = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]

# Quantile for filtering outliers by rating
RANK_OUTLIER_QUANTILE = 0.99

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] Loaded lines: {len(df)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning: omissions, data leaks, emissions."""

    # Turning every space which were set as -1 to NaN,
    # otherwise model will be treating it as a valid result.
    df[MISSING_AS_MINUS_ONE] = df[MISSING_AS_MINUS_ONE].replace(-1, np.nan)
    print(f"[clean] Gaps after replacing -1 with NaN:\n{df[MISSING_AS_MINUS_ONE].isna().sum()}")

    # Dropping columns with data leakage
    df = df.drop(columns=["Odd_1", "Odd_2", "Score"])

    # Working only with the matches after 2015
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.year >= MIN_YEAR]
    print(f"[clean] Filter by the year (>= {MIN_YEAR}): {len(df)} lines")

    # Removing rows with gaps in key numerical features
    df = df.dropna(subset=["Rank_1", "Rank_2", "Pts_1", "Pts_2"])
    print(f"[clean] After deleting the rows with NaN: {len(df)} lines")

    # Outliers by rating: The data includes ranks in the thousands, which probably
    # are qualifier players and data errors, so we cut off the tail by quantile.
    rank_threshold = df[["Rank_1", "Rank_2"]].stack().quantile(RANK_OUTLIER_QUANTILE)
    df = df[(df["Rank_1"] <= rank_threshold) & (df["Rank_2"] <= rank_threshold)]
    print(f"[clean] Rating threshold for an outlier: {rank_threshold:.0f}, left {len(df)} lines")

    # Winner must match one of the players in the list
    valid = (df["Winner"] == df["Player_1"]) | (df["Winner"] == df["Player_2"])
    df = df[valid]

    return df.reset_index(drop=True)


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a binary target: target = 1 if Player_1 won.

    Problem: in the raw data Player_1 is not a random player, but the one
    written first in the row. Taking target = (Winner == Player_1) directly
    may give imbalanced classes, and the model would learn the artifact
    of the recording order instead of real patterns.

    Solution: randomly "flip" half of the rows - swap the players together
    with their ranks and points, and invert the target. After that the class
    distribution is close to 50/50.
    """
    df = df.copy()
    df["target"] = (df["Winner"] == df["Player_1"]).astype(int)

    rng = np.random.default_rng(RANDOM_STATE)
    swap_mask = rng.random(len(df)) < 0.5

    # Swapping paired columns only in the selected rows
    for col_1, col_2 in [("Player_1", "Player_2"), ("Rank_1", "Rank_2"), ("Pts_1", "Pts_2")]:
        tmp = df.loc[swap_mask, col_1].copy()
        df.loc[swap_mask, col_1] = df.loc[swap_mask, col_2]
        df.loc[swap_mask, col_2] = tmp

    # Inverting the target where the players were swapped
    df.loc[swap_mask, "target"] = 1 - df.loc[swap_mask, "target"]

    # Winner is no longer needed - the target is already derived from it,
    # and keeping it in the features would be a direct leak of the answer.
    df = df.drop(columns=["Winner"])

    print(f"[target] Class balance:\n{df['target'].value_counts(normalize=True)}")
    return df


def split_and_save(df: pd.DataFrame) -> None:
    """Splits into train/test and saves to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # stratify keeps the class proportion in both subsets
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"[split] train: {len(train_df)} lines, test: {len(test_df)} lines")


def main() -> None:
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = make_target(df)
    split_and_save(df)
    print("[done] Stage 1 finished")


if __name__ == "__main__":
    main()