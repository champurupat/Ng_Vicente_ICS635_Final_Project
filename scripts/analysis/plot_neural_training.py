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
MODEL_NAMES = {"2_mlp_nn": "Multi-Layer Perceptron", "3_tabnet": "TabNet"}

# Metric mapping for different models
METRIC_MAPPING = {
    "2_mlp_nn": {
        "train_acc": "accuracy",
        "val_acc": "val_accuracy",
        "train_loss": "loss",
        "val_loss": "val_loss",
    },
    "3_tabnet": {
        "train_acc": "train_accuracy",
        "val_acc": "val_accuracy",
        "train_loss": "train_logloss",
        "val_loss": "val_logloss",
    },
}

# Column name mapping
COL_NAMES = {"position": "Pos", "round": "Round"}


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_training_history(model_dir, output_dir, model_type):
    """Plot training history metrics for a neural network model."""
    history_path = os.path.join(
        model_dir, model_type, "plotting_data/training_history.csv"
    )
    if not os.path.exists(history_path):
        print(f"No training history found for {model_dir}/{model_type}")
        return

    # Read training history
    history_df = pd.read_csv(history_path)
    epochs = range(1, len(history_df) + 1)

    # Get correct metric names for this model
    model_name = Path(model_dir).name
    metrics = METRIC_MAPPING[model_name]

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot accuracy
    ax1.plot(
        epochs, history_df[metrics["train_acc"]], "b-", label="Training", linewidth=2
    )
    ax1.plot(
        epochs, history_df[metrics["val_acc"]], "r-", label="Validation", linewidth=2
    )

    col_name = COL_NAMES[model_type]
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(f"{col_name} Prediction Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True)

    # Plot loss
    ax2.plot(
        epochs, history_df[metrics["train_loss"]], "b-", label="Training", linewidth=2
    )
    ax2.plot(
        epochs, history_df[metrics["val_loss"]], "r-", label="Validation", linewidth=2
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, model_type, "training_history.png")
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=dpi_setting, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    # Create figures directory if it doesn't exist
    figures_dir = "results/figures"
    ensure_dir(figures_dir)

    # Process neural network models
    neural_models = ["2_mlp_nn", "3_tabnet"]
    model_types = ["position", "round"]

    print("Starting neural network training visualization generation...")

    for model_name in neural_models:
        model_dir = f"results/{model_name}"
        if not os.path.exists(model_dir):
            print(f"Model directory {model_dir} not found!")
            continue

        # Create model-specific output directory
        output_dir = os.path.join(figures_dir, model_name)
        ensure_dir(output_dir)

        # Generate and save plots for both position and round predictions
        for model_type in model_types:
            if plot_path := plot_training_history(model_dir, output_dir, model_type):
                print(
                    f"Training history plot saved for {MODEL_NAMES[model_name]} ({model_type})"
                )
                print(f"  └─ {plot_path}")


if __name__ == "__main__":
    main()
