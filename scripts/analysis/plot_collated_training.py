import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path

# Model name mapping for better display
MODEL_NAMES = {"2_mlp_nn": "Multi-Layer Perceptron", "3_tabnet": "TabNet"}


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def collate_training_plots(model_dir, model_name):
    """Create a 2x1 grid of training history plots for a model."""
    # Set up the figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    # fig.suptitle(
    #     f"{model_name} Training History", fontsize=22, fontweight="bold", y=0.95
    # )

    # Add space between rows
    plt.subplots_adjust(
        left=0.1,  # Left margin
        right=0.9,  # Right margin
        hspace=0.1,  # Height space between subplots
        top=0.93,  # Top margin
        bottom=0.05,  # Bottom margin
    )

    # Load and plot round prediction training history
    round_path = os.path.join(
        "results/figures", model_dir, "round", "training_history.png"
    )
    if os.path.exists(round_path):
        img = mpimg.imread(round_path)
        axes[0].imshow(img)
        axes[0].set_title(
            f"{model_name} Round Prediction", pad=5, fontsize=16, fontweight="bold"
        )

    # Load and plot position prediction training history
    pos_path = os.path.join(
        "results/figures", model_dir, "position", "training_history.png"
    )
    if os.path.exists(pos_path):
        img = mpimg.imread(pos_path)
        axes[1].imshow(img)
        axes[1].set_title(
            f"{model_name} Position Prediction", pad=5, fontsize=16, fontweight="bold"
        )

    # Remove ticks for both plots
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Save the collated figure
    output_dir = os.path.join("results/figures/collated")
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, f"{model_dir}_training_histories.png")
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"Collated training history plots for {model_name} saved to {output_path}")
    return output_path


def main():
    """Main function to run the collation script."""
    print("\nCollating training history plots...")

    # Process both MLP and TabNet models
    for model_dir, model_name in MODEL_NAMES.items():
        collate_training_plots(model_dir, model_name)


if __name__ == "__main__":
    main()
