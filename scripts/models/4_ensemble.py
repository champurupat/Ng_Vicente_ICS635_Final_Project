import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
from io import StringIO
import os
from datetime import datetime
import optuna
from constants import features_pos_, features_round_, optuna_num_trials

feature_sets = [features_pos_, features_round_]
path_sets = ["position", "round"]
col_name = ["Pos", "Round"]

for p_set, c_name in zip(path_sets, col_name):
    results_path = f"results/4_ensemble/{p_set}"
    # Create results directories if they don't exist
    os.makedirs(f"{results_path}/logs", exist_ok=True)

    # Set up logging for hyperparameter optimization
    log_file = f'{results_path}/logs/optuna_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    def log_progress(message):
        with open(log_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")

    # Create a string buffer to capture all print outputs
    output_buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer

    # Load predictions from individual models
    knn_predictions = pd.read_csv(
        f"results/0_knn_classifier/{p_set}/plotting_data/predictions.csv"
    )
    rf_predictions = pd.read_csv(
        f"results/1_random_forest/{p_set}/plotting_data/predictions.csv"
    )
    nn_predictions = pd.read_csv(
        f"results/2_mlp_nn/{p_set}/plotting_data/predictions.csv"
    )
    tabnet_predictions = pd.read_csv(
        f"results/3_tabnet/{p_set}/plotting_data/predictions.csv"
    )

    # Define column names based on the prediction type
    actual_col = f"Actual_{c_name}"
    pred_col = f"Predicted_{c_name}"

    # Verify all predictions are for the same test samples
    models = {
        "KNN": knn_predictions,
        "Random Forest": rf_predictions,
        "Neural Network": nn_predictions,
        "TabNet": tabnet_predictions,
    }

    for model_name, model in models.items():
        try:
            assert len(knn_predictions) == len(model)
            assert all(knn_predictions[actual_col] == model[actual_col])
        except AssertionError:
            print(f"Assertion failed for model: {model_name}")
            # Print indices where the actual values do not match
            mismatched_indices = knn_predictions[actual_col] != model[actual_col]
            print(
                "Mismatched indices:",
                mismatched_indices[mismatched_indices].index.tolist(),
            )
            # Optionally, print the mismatched entries
            print("Mismatched entries in KNN:", knn_predictions[mismatched_indices])
            print("Mismatched entries in", model_name, ":", model[mismatched_indices])

    def weighted_vote(predictions_list, weights):
        """
        Combine predictions from multiple models using weighted voting

        Args:
            predictions_list: List of predictions from each model
            weights: Dictionary mapping model names to their weights

        Returns:
            List of ensemble predictions
        """
        ensemble_predictions = []

        for i in range(len(predictions_list[0])):
            # Get predictions for current sample from all models
            current_predictions = [
                model[pred_col].iloc[i] for model in predictions_list
            ]

            # Get unique predictions from all models
            unique_predictions = set(current_predictions)

            # Initialize vote dictionary
            votes = {pred: 0 for pred in unique_predictions}

            # Add weighted votes
            for pred, (model_name, weight) in zip(current_predictions, weights.items()):
                votes[pred] += weight

            # Return prediction with highest weighted votes
            ensemble_predictions.append(max(votes.items(), key=lambda x: x[1])[0])

        return ensemble_predictions

    def objective(trial):
        # Define weights for each model as hyperparameters
        weights = {
            "knn": trial.suggest_float("knn_weight", 0, 1),
            "rf": trial.suggest_float("rf_weight", 0, 1),
            "nn": trial.suggest_float("nn_weight", 0, 1),
            "tabnet": trial.suggest_float("tabnet_weight", 0, 1),
        }

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Generate ensemble predictions
        predictions_list = [
            knn_predictions,
            rf_predictions,
            nn_predictions,
            tabnet_predictions,
        ]
        ensemble_preds = weighted_vote(predictions_list, weights)

        # Calculate accuracy
        accuracy = accuracy_score(knn_predictions[actual_col], ensemble_preds)

        # Log progress
        log_progress(f"Trial {trial.number}:")
        log_progress(f"Weights: {weights}")
        log_progress(f"Accuracy: {accuracy:.4f}\n")

        return accuracy

    # Run hyperparameter optimization
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_num_trials)

    # Log best results
    # normalize weights for logging
    total_weight = sum(study.best_trial.params.values())
    best_weights = {k: v / total_weight for k, v in study.best_trial.params.items()}
    log_progress("\nBest trial:")
    log_progress(f"Value: {study.best_trial.value:.4f}")
    log_progress(f"Params: {best_weights}")

    # Get best weights and normalize them
    best_weights = {
        "knn": study.best_trial.params["knn_weight"],
        "rf": study.best_trial.params["rf_weight"],
        "nn": study.best_trial.params["nn_weight"],
        "tabnet": study.best_trial.params["tabnet_weight"],
    }
    total_weight = sum(best_weights.values())
    best_weights = {k: v / total_weight for k, v in best_weights.items()}

    print("\nBest Model Weights:")
    for model, weight in best_weights.items():
        print(f"{model.upper():6}: {weight:.3f}")

    # Generate final ensemble predictions with best weights
    predictions_list = [
        knn_predictions,
        rf_predictions,
        nn_predictions,
        tabnet_predictions,
    ]
    final_predictions = weighted_vote(predictions_list, best_weights)

    # Calculate and print metrics
    true_labels = knn_predictions[actual_col]
    print("\nEnsemble Model Performance:")
    print(f"Overall Accuracy: {accuracy_score(true_labels, final_predictions):.2%}")

    print(f"\nPer-{c_name} Accuracy:")
    for label in sorted(true_labels.unique()):
        mask = true_labels == label
        if sum(mask) > 0:
            label_accuracy = accuracy_score(
                true_labels[mask], [p for m, p in zip(mask, final_predictions) if m]
            )
            print(
                f"{label:4} - Accuracy: {label_accuracy:.2%} (Test samples: {sum(mask)})"
            )

    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, final_predictions))

    # Print comparison of model accuracies
    print("\nModel Accuracy Comparison:")
    models = {
        "KNN": knn_predictions,
        "Random Forest": rf_predictions,
        "Neural Network": nn_predictions,
        "TabNet": tabnet_predictions,
        "Ensemble": pd.DataFrame(
            {actual_col: true_labels, pred_col: final_predictions}
        ),
    }

    for model_name, predictions in models.items():
        acc = accuracy_score(predictions[actual_col], predictions[pred_col])
        print(f"{model_name:15}: {acc:.2%}")

    # Restore original stdout and save the output to a file
    sys.stdout = original_stdout
    with open(f"{results_path}/logs/results.txt", "w") as f:
        f.write(output_buffer.getvalue())

    # Log best parameters to a separate file
    best_params_log_file = f"{results_path}/logs/best_params.log"
    with open(best_params_log_file, "w") as f:
        f.write("Best Parameters:\n")
        for param, value in study.best_trial.params.items():
            f.write(f"{param}: {value}\n")

    # Create plotting_data directory
    plotting_data_path = f"{results_path}/plotting_data"
    os.makedirs(plotting_data_path, exist_ok=True)

    # Save ensemble predictions to CSV
    pd.DataFrame({actual_col: true_labels, pred_col: final_predictions}).to_csv(
        f"{plotting_data_path}/predictions.csv", index=False
    )

    # Save optimization history data
    optimization_history = study.trials_dataframe()
    optimization_history.to_csv(
        f"{plotting_data_path}/optimization_history.csv", index=False
    )

    # Save raw importances data
    param_importances = optuna.importance.get_param_importances(study)
    pd.DataFrame(param_importances.items(), columns=["Parameter", "Importance"]).to_csv(
        f"{plotting_data_path}/ensemble_weights.csv", index=False
    )
