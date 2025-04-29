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


def plot_single_matrix(ax, img_path, title=None, ylabel=None):
    """Helper function to plot a single matrix image onto an axis."""
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
    if title:
        ax.set_title(title, pad=15, fontsize=22, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=18, fontweight="bold", rotation=90, labelpad=15)
    ax.set_xticks([])
    ax.set_yticks([])


def collate_confusion_matrices():
    """Create three separate figures (2x2, 2x2, 1x2) for confusion matrices."""
    figures_dir = "results/figures"
    output_dir = os.path.join(figures_dir, "collated")
    ensure_dir(output_dir)

    # Get all model directories in order
    model_dirs = sorted([d for d in os.listdir(figures_dir) if d[0].isdigit()])

    # --- Figure 1: KNN & Random Forest (2x2) ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 13))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes1[0, 0].set_title("Round Prediction", pad=15, fontsize=22, fontweight="bold")
    axes1[0, 1].set_title("Position Prediction", pad=15, fontsize=22, fontweight="bold")

    for i, model_dir in enumerate(model_dirs[0:2]):  # First two models
        model_name = MODEL_NAMES.get(model_dir, model_dir)
        round_path = os.path.join(
            figures_dir, model_dir, "round", "confusion_matrix.png"
        )
        pos_path = os.path.join(
            figures_dir, model_dir, "position", "confusion_matrix.png"
        )

        plot_single_matrix(axes1[i, 0], round_path, ylabel=model_name)
        plot_single_matrix(axes1[i, 1], pos_path)

    output_path1 = os.path.join(output_dir, "confusion_matrices_part1.png")
    plt.savefig(output_path1, dpi=250, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved Part 1 to {output_path1}")

    # --- Figure 2: MLP & TabNet (2x2) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 13))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes2[0, 0].set_title("Round Prediction", pad=15, fontsize=22, fontweight="bold")
    axes2[0, 1].set_title("Position Prediction", pad=15, fontsize=22, fontweight="bold")

    for i, model_dir in enumerate(model_dirs[2:4]):  # Models 3 and 4
        model_name = MODEL_NAMES.get(model_dir, model_dir)
        round_path = os.path.join(
            figures_dir, model_dir, "round", "confusion_matrix.png"
        )
        pos_path = os.path.join(
            figures_dir, model_dir, "position", "confusion_matrix.png"
        )

        plot_single_matrix(axes2[i, 0], round_path, ylabel=model_name)
        plot_single_matrix(axes2[i, 1], pos_path)

    output_path2 = os.path.join(output_dir, "confusion_matrices_part2.png")
    plt.savefig(output_path2, dpi=250, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved Part 2 to {output_path2}")

    # --- Figure 3: Ensemble (1x2) ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 7.5))  # Adjusted size for 1x2
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    axes3[0].set_title("Round Prediction", pad=15, fontsize=22, fontweight="bold")
    axes3[1].set_title("Position Prediction", pad=15, fontsize=22, fontweight="bold")

    model_dir = model_dirs[4]  # Last model
    model_name = MODEL_NAMES.get(model_dir, model_dir)
    round_path = os.path.join(figures_dir, model_dir, "round", "confusion_matrix.png")
    pos_path = os.path.join(figures_dir, model_dir, "position", "confusion_matrix.png")

    plot_single_matrix(axes3[0], round_path, ylabel=model_name)
    plot_single_matrix(axes3[1], pos_path)

    output_path3 = os.path.join(output_dir, "confusion_matrices_part3.png")
    plt.savefig(output_path3, dpi=250, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved Part 3 to {output_path3}")


def main():
    """Main function to run the collation script."""
    print("\nCollating confusion matrices into parts...")
    collate_confusion_matrices()


if __name__ == "__main__":
    main()
