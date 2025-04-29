import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path

# Model name mapping for better display
MODEL_NAMES = {
    "0_knn_classifier": "K-Nearest Neighbors",
    "1_random_forest": "Random Forest",
    "2_mlp_nn": "Multi-Layer Perceptron",
    "3_tabnet": "TabNet",
    "4_ensemble": "Ensemble Model",
}


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def collate_optimization_plots():
    """Create a 5x2 grid of optimization history plots."""
    # Set up the figure
    fig, axes = plt.subplots(5, 2, figsize=(26, 30))  # Increased width for more spacing

    axes[0, 0].set_title("Round Prediction", pad=15, fontsize=22, fontweight="bold")
    axes[0, 1].set_title("Position Prediction", pad=15, fontsize=22, fontweight="bold")

    # Add space between columns and adjust margins
    plt.subplots_adjust(
        wspace=0.05, hspace=0.03
    )  # Increased wspace for wider column separation

    # Get all model directories in order
    figures_dir = "results/figures"
    model_dirs = sorted([d for d in os.listdir(figures_dir) if d[0].isdigit()])

    for i, model_dir in enumerate(model_dirs):
        model_name = MODEL_NAMES.get(model_dir, model_dir)

        # Load and plot round prediction optimization
        round_path = os.path.join(
            figures_dir, model_dir, "round", "optimization_history.png"
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

        # Load and plot position prediction optimization
        pos_path = os.path.join(
            figures_dir, model_dir, "position", "optimization_history.png"
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
    output_path = os.path.join(output_dir, "optimization_histories.png")
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"Collated optimization history plots saved to {output_path}")
    return output_path


def main():
    """Main function to run the collation script."""
    print("\nCollating optimization history plots...")
    collate_optimization_plots()


if __name__ == "__main__":
    main()
