import pandas as pd
import os
import re
import numpy as np
from pathlib import Path


def convert_height_to_inches(height_str):
    """Convert height from feet-inches format to total inches."""
    if pd.isna(height_str):
        return None
    feet, inches = map(int, height_str.split("-"))
    return feet * 12 + inches


def extract_draft_info(draft_str):
    """Extract round number and pick number from draft string."""
    if pd.isna(draft_str):
        return pd.Series({"Round": None, "Pick": None})

    # Extract round and pick numbers using regex
    round_match = re.search(r"(\d+)(?:st|nd|rd|th)", draft_str)
    pick_match = re.search(r"(\d+)(?:st|nd|rd|th) pick", draft_str)

    round_num = int(round_match.group(1)) if round_match else None
    pick_num = int(pick_match.group(1)) if pick_match else None

    return pd.Series({"Round": round_num, "Pick": pick_num})


def main():
    # Get the path to raw data directory
    raw_data_dir = Path(__file__).parents[2] / "data" / "raw_data"

    # Get all CSV files and sort them by year
    csv_files = sorted([f for f in os.listdir(raw_data_dir) if f.endswith(".csv")])

    # Read and combine all CSV files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(raw_data_dir / file)
        dfs.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Convert height to inches
    combined_df["Ht"] = combined_df["Ht"].apply(convert_height_to_inches)

    # Extract draft round and pick information
    draft_info = combined_df["Drafted (tm/rnd/yr)"].apply(extract_draft_info)
    combined_df = pd.concat([combined_df, draft_info], axis=1)

    # convert height and weight, round, bench reps, and pick to integers if possible
    combined_df["Ht"] = (
        pd.to_numeric(combined_df["Ht"], errors="coerce").fillna(0).astype(int)
    )
    combined_df["Wt"] = (
        pd.to_numeric(combined_df["Wt"], errors="coerce").fillna(0).astype(int)
    )
    combined_df["Round"] = (
        pd.to_numeric(combined_df["Round"], errors="coerce").fillna(8).astype(int)
    )
    combined_df["Pick"] = (
        pd.to_numeric(combined_df["Pick"], errors="coerce").fillna(263).astype(int)
    )

    # if height is 0, or weight is 0, delete the row
    combined_df = combined_df[
        (combined_df["Ht"] != 0) & (combined_df["Wt"] != 0)
    ].reset_index(drop=True)

    # Remove specified columns
    columns_to_remove = [
        "Drafted (tm/rnd/yr)",
        "Player",
        "School",
        "College",
        "Player-additional",
        "zPlayer",
        "Pick",
    ]
    combined_df = combined_df.drop(columns=columns_to_remove)

    # Save the processed data
    output_path = (
        Path(__file__).parents[2] / "data" / "processed" / "merged_data_2000_2025.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    print("\nFirst few rows of the processed data:")
    print(combined_df.head())
    print("\nDataset shape:", combined_df.shape)


if __name__ == "__main__":
    main()
