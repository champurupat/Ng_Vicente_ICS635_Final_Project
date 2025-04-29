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


def collate_feature_importance_plots():
    """Create a 2x2 grid of feature importance plots."""
    # Set up the figure
    fig, axes = plt.subplots(
        2, 2, figsize=(20, 12)
    )  # Similar width to optimization plots

    axes[0, 0].set_title("Round Prediction", pad=15, fontsize=22, fontweight="bold")
    axes[0, 1].set_title("Position Prediction", pad=15, fontsize=22, fontweight="bold")

    # Add space between columns and adjust margins
    plt.subplots_adjust(wspace=0.05, hspace=0.03)

    # Get relevant model directories
    figures_dir = "results/figures"
    model_dirs = ["1_random_forest", "3_tabnet"]  # Only RF and TabNet

    for i, model_dir in enumerate(model_dirs):
        model_name = MODEL_NAMES.get(model_dir, model_dir)

        # Load and plot round prediction feature importance
        round_path = os.path.join(
            figures_dir, model_dir, "round", "feature_importance.png"
        )
        if os.path.exists(round_path):
            img = mpimg.imread(round_path)
            axes[i, 0].imshow(img)
            # Larger row labels with background for better visibility
            axes[i, 0].set_ylabel(
                model_name,
                fontsize=18,  # Increased font size
                fontweight="bold",
                rotation=90,
                labelpad=15,  # Increased padding
            )

        # Load and plot position prediction feature importance
        pos_path = os.path.join(
            figures_dir, model_dir, "position", "feature_importance.png"
        )
        if os.path.exists(pos_path):
            img = mpimg.imread(pos_path)
            axes[i, 1].imshow(img)

        # Remove ticks for both plots
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    # Save the collated figure
    output_dir = os.path.join(figures_dir, "collated")
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "feature_importances.png")
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"Collated feature importance plots saved to {output_path}")
    return output_path


def main():
    """Main function to run the collation script."""
    print("\nCollating feature importance plots...")
    collate_feature_importance_plots()


if __name__ == "__main__":
    main()
