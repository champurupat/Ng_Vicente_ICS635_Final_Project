import pandas as pd
import os
from pathlib import Path


def analyze_category_distribution(df, column):
    """Analyze the distribution of a categorical column."""
    # Get value counts and calculate percentages
    counts = df[column].value_counts().sort_index()
    total = len(df)
    percentages = (counts / total * 100).round(2)

    # Create a DataFrame with both counts and percentages
    distribution = pd.DataFrame({"Count": counts, "Percentage (%)": percentages})

    # Sort by percentage in ascending order
    distribution = distribution.sort_values("Percentage (%)")

    # Add total row
    distribution.loc["Total"] = [total, 100.0]

    return distribution


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    """Analyze the distribution of positions and rounds in the filtered data."""
    # Read the filtered data
    data_path = "data/processed/filtered_data.csv"
    df = pd.read_csv(data_path)

    # Create output directory
    output_dir = "results/data_analysis"
    ensure_dir(output_dir)

    # Analyze Position distribution
    pos_dist = analyze_category_distribution(df, "Pos")
    pos_output = os.path.join(output_dir, "position_distribution.csv")
    pos_dist.to_csv(pos_output)

    # Analyze Round distribution
    round_dist = analyze_category_distribution(df, "Round")
    round_output = os.path.join(output_dir, "round_distribution.csv")
    round_dist.to_csv(round_output)

    # Print results with nice formatting
    print("\nPosition Distribution (sorted by percentage):")
    print("=" * 50)
    print(pos_dist.to_string(float_format=lambda x: "{:.2f}".format(x)))
    print(f"\nSaved to: {pos_output}")

    print("\nRound Distribution (sorted by percentage):")
    print("=" * 50)
    print(round_dist.to_string(float_format=lambda x: "{:.2f}".format(x)))
    print(f"\nSaved to: {round_output}")


if __name__ == "__main__":
    main()
