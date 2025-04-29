import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
from pathlib import Path
from constants import labels_pos_, labels_round_

# Set style for better-looking plots
plt.style.use("default")

# Plot settings
dpi_setting = 250


# Model name mapping for better display
MODEL_NAMES = {
    "0_knn_classifier": "K-Nearest Neighbors",
    "1_random_forest": "Random Forest",
    "2_mlp_nn": "Multi-Layer Perceptron",
    "3_tabnet": "TabNet",
    "4_ensemble": "Ensemble Model",
}

# Column name mapping
COL_NAMES = {"position": "Pos", "round": "Round"}


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_feature_importances(model_dir, output_dir):
    """Plot feature importance analysis for a model."""
    feature_importance_path = os.path.join(
        model_dir, "plotting_data/feature_importances.csv"
    )
    if not os.path.exists(feature_importance_path):
        print(f"No feature importance data found for {model_dir}")
        return

    # Read feature importances
    fi_df = pd.read_csv(feature_importance_path)

    # Sort by importance
    fi_df = fi_df.sort_values("importance", ascending=True)

    # Create horizontal bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(fi_df["feature"], fi_df["importance"])

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

    model_name = MODEL_NAMES.get(Path(model_dir).name, Path(model_dir).name)
    plt.xlabel("Importance Score")
    plt.tight_layout()

    # Save to model-specific subdirectory in figures
    output_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(output_path, dpi=dpi_setting, bbox_inches="tight")
    plt.close()
    return output_path


def plot_optimization_history(model_dir, output_dir):
    """Plot optimization history for a model."""
    optimization_path = os.path.join(
        model_dir, "plotting_data/optimization_history.csv"
    )
    if not os.path.exists(optimization_path):
        print(f"No optimization history found for {model_dir}")
        return

    # Read optimization history
    opt_df = pd.read_csv(optimization_path)

    # Create figure with two subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot optimization progress (individual points)
    ax1.plot(
        opt_df["number"], opt_df["value"], "o", markersize=8, label="Objective Value"
    )

    # Calculate and plot the best value line
    best_values = []
    current_best = float("-inf")
    for value in opt_df["value"]:
        current_best = max(current_best, value)
        best_values.append(current_best)

    ax1.plot(opt_df["number"], best_values, "r-", label="Best Value", linewidth=2)

    # Find first occurrence of maximum value
    max_value = max(best_values)
    max_trial = opt_df["number"][opt_df["value"] == max_value].iloc[0]

    # Determine if max occurs in left or right half of plot
    x_range = opt_df["number"].max() - opt_df["number"].min()
    x_mid = opt_df["number"].min() + x_range / 2

    # Position annotation below the maximum point
    if max_trial < x_mid:  # Max is in left half
        xytext = (30, -40)  # Position slightly right and below
        ha = "left"
    else:  # Max is in right half
        xytext = (-30, -40)  # Position slightly left and below
        ha = "right"

    ax1.annotate(
        f"Best: {max_value:.3f}",
        xy=(max_trial, max_value),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=1.0",
            fc="yellow",
            alpha=0.5,
            edgecolor="black",
            linewidth=2,
        ),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", linewidth=2),
    )

    ax1.legend(loc="lower right")
    # ax1.set_title("Optimization History Plot")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Objective Value")
    ax1.grid(True)

    plt.tight_layout()

    # Save to model-specific subdirectory in figures
    output_path = os.path.join(output_dir, "optimization_history.png")
    plt.savefig(output_path, dpi=dpi_setting, bbox_inches="tight")
    plt.close()
    return output_path


def plot_confusion_matrix(model_dir, output_dir, labels_, model_type):
    """Plot confusion matrix for a model."""
    confusion_path = os.path.join(model_dir, "plotting_data/predictions.csv")

    # Get the appropriate column names based on model type
    col_name = COL_NAMES[model_type]
    actual_col = f"Actual_{col_name}"
    pred_col = f"Predicted_{col_name}"

    # Read predictions
    predictions = pd.read_csv(confusion_path)

    # Create confusion matrix, explicitly using labels_ for order
    cm = confusion_matrix(
        predictions[actual_col],
        predictions[pred_col],
        labels=labels_,  # Ensure calculation uses the provided label order
    )

    # Create heatmap with improved visualization, using the same labels_
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_)
    cm_display.plot()

    model_name = MODEL_NAMES.get(Path(model_dir).name, Path(model_dir).name)
    plt.ylabel(f"True {col_name}")
    plt.xlabel(f"Predicted {col_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save to model-specific subdirectory in figures
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path, dpi=dpi_setting, bbox_inches="tight")
    plt.close()
    return output_path


def process_model(model_dir, model_type, labels_):
    """Process all plots for a single model."""
    # Extract the model name from the path
    model_name = Path(model_dir).parts[-2]
    print(f"\nProcessing {MODEL_NAMES.get(model_name, model_name)}...")

    # Ensure the model directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found!")
        return

    # Create model-specific output directory in figures
    output_dir = os.path.join("results/figures", model_name, model_type)
    ensure_dir(output_dir)

    # Create plots and collect paths
    plots = []
    if plot := plot_feature_importances(model_dir, output_dir):
        plots.append(plot)
        print(f"Feature importance plot saved to {plot}")

    if plot := plot_optimization_history(model_dir, output_dir):
        plots.append(plot)
        print(f"Optimization history plot saved to {plot}")

    if plot := plot_confusion_matrix(model_dir, output_dir, labels_, model_type):
        plots.append(plot)
        print(f"Confusion matrix plot saved to {plot}")

    return plots


def main():
    model_types = ["position", "round"]
    labels = [labels_pos_, labels_round_]
    for model_type, label in zip(model_types, labels):
        # Get all model directories
        results_dir = "results"
        model_dirs = sorted(
            [
                d
                for d in os.listdir(results_dir)
                if os.path.isdir(os.path.join(results_dir, d)) and d[0].isdigit()
            ]
        )  # Only process numbered model directories

        # Process each model
        all_plots = []
        for model_dir in model_dirs:
            full_path = os.path.join(results_dir, model_dir, model_type)
            plots = process_model(full_path, model_type, label)
            if plots:
                all_plots.extend(plots)


if __name__ == "__main__":
    main()
