import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set style for better-looking plots
plt.style.use("default")
sns.set_theme(style="whitegrid")

# Plot settings
dpi_setting = 250

# Model name mapping for better display
MODEL_DISPLAY_NAMES = {
    "knn_weight": "K-Nearest Neighbors",
    "rf_weight": "Random Forest",
    "nn_weight": "MLP Neural Network",
    "tabnet_weight": "TabNet",
}

# Column name mapping
COL_NAMES = {"position": "Pos", "round": "Round"}


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_ensemble_weights(model_dir, output_dir, model_type):
    """Plot ensemble model weights visualization."""
    weights_path = os.path.join(
        model_dir, model_type, "plotting_data/ensemble_weights.csv"
    )
    if not os.path.exists(weights_path):
        print(f"No ensemble weights found in {model_dir}/{model_type}")
        return

    # Read weights data
    weights_df = pd.read_csv(weights_path)

    # Sort by importance
    weights_df = weights_df.sort_values("Importance", ascending=True)

    # Replace technical names with display names
    weights_df["Parameter"] = weights_df["Parameter"].map(MODEL_DISPLAY_NAMES)

    # Create horizontal bar plot
    plt.figure(figsize=(12, 6))

    # Create bars with gradient colors
    bars = plt.barh(weights_df["Parameter"], weights_df["Importance"])

    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    col_name = COL_NAMES[model_type]
    plt.xlabel(f"Weight ({col_name} Prediction)")

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, model_type, "ensemble_weights.png")
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=dpi_setting, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    # Set up directories
    model_dir = "results/4_ensemble"
    figures_dir = "results/figures/4_ensemble"
    ensure_dir(figures_dir)

    print("\nGenerating ensemble model visualizations...")

    # Process both position and round predictions
    model_types = ["position", "round"]

    # Generate plots
    plots = []

    for model_type in model_types:
        if plot := plot_ensemble_weights(model_dir, figures_dir, model_type):
            plots.append(plot)
            print(f"✓ Ensemble weights plot saved to {plot} ({model_type})")

    if plots:
        print(f"\nGenerated {len(plots)} plots in {figures_dir}")
        print("Directory structure:")
        for plot in plots:
            print(
                f"  └─ {os.path.basename(os.path.dirname(plot))}/{os.path.basename(plot)}"
            )
    else:
        print("No plots were generated. Please check if the required data files exist.")


if __name__ == "__main__":
    main()
