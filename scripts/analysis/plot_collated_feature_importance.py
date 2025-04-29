import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path

# Model name mapping for better display
MODEL_NAMES = {"1_random_forest": "Random Forest", "3_tabnet": "TabNet"}


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_single_image(ax, img_path, ylabel=None):
    """Helper function to plot a single image onto an axis with optional ylabel."""
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", rotation=90, labelpad=10)
    ax.set_xticks([])
    ax.set_yticks([])


def collate_feature_importance_plots():
    """Create two separate 2x1 figures for feature importance (RF above TabNet)."""
    figures_dir = "results/figures"
    output_dir = os.path.join(figures_dir, "collated")
    ensure_dir(output_dir)

    # Models to compare
    model_dirs = ["1_random_forest", "3_tabnet"]

    # --- Figure 1: Round Prediction Feature Importance (2x1) ---
    fig1, axes1 = plt.subplots(2, 1, figsize=(8, 10))  # Adjusted for 2x1 layout
    plt.subplots_adjust(hspace=0.1)

    for i, model_dir in enumerate(model_dirs):
        model_name = MODEL_NAMES.get(model_dir, model_dir)
        round_path = os.path.join(
            figures_dir, model_dir, "round", "feature_importance.png"
        )
        plot_single_image(axes1[i], round_path, ylabel=model_name)

    output_path1 = os.path.join(
        output_dir, "feature_importances_round_stacked.png"
    )  # New name
    plt.savefig(output_path1, dpi=250, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved Round Feature Importance (Stacked) to {output_path1}")

    # --- Figure 2: Position Prediction Feature Importance (2x1) ---
    fig2, axes2 = plt.subplots(2, 1, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.1)

    for i, model_dir in enumerate(model_dirs):
        model_name = MODEL_NAMES.get(model_dir, model_dir)
        pos_path = os.path.join(
            figures_dir, model_dir, "position", "feature_importance.png"
        )
        plot_single_image(axes2[i], pos_path, ylabel=model_name)

    output_path2 = os.path.join(
        output_dir, "feature_importances_position_stacked.png"
    )  # New name
    plt.savefig(output_path2, dpi=250, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved Position Feature Importance (Stacked) to {output_path2}")


def main():
    """Main function to run the collation script."""
    print("\nCollating feature importance plots into stacked figures...")
    collate_feature_importance_plots()


if __name__ == "__main__":
    main()
