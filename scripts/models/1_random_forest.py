import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import sys
import optuna
from datetime import datetime
from io import StringIO
import os
from constants import features_pos_, features_round_, optuna_num_trials

feature_sets = [features_pos_, features_round_]
path_sets = ["position", "round"]
col_name = ["Pos", "Round"]

for f_set, p_set, c_name in zip(feature_sets, path_sets, col_name):
    results_path = f"results/1_random_forest/{p_set}"
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

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load and preprocess data
    df = pd.read_csv("data/processed/filtered_data.csv")
    features = f_set
    X = df[features]
    y = df[c_name]

    # Split the data using the same seed as other models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Set up preprocessing based on feature set
    if "Pos" in features:
        # For round prediction (includes 'Pos' categorical feature)
        categorical_features = ["Pos"]
        numerical_features = [col for col in features if col != "Pos"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                (
                    "cat",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )

        # Process training and test data
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)

        # Get feature names after transformation
        encoded_features = numerical_features + [
            f"Pos_{cat}"
            for cat in preprocessor.named_transformers_["cat"].categories_[0]
        ]
    else:
        # For position prediction (all numerical features)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        encoded_features = features

    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

        # Create and evaluate model using cross-validation
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="accuracy"
        )
        accuracy = scores.mean()

        # Log progress
        log_progress(f"Trial {trial.number}:")
        log_progress(f"Params: {params}")
        log_progress(f"Accuracy: {accuracy:.4f}\n")

        return accuracy

    def create_random_forest(best_params):
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
        return model

    # Run hyperparameter optimization
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_num_trials)

    # Log best results
    log_progress("\nBest trial:")
    log_progress(f"Value: {study.best_trial.value:.4f}")
    log_progress(f"Params: {study.best_trial.params}")

    # Log best parameters to a separate file
    best_params_log_file = f"{results_path}/logs/best_params.log"
    with open(best_params_log_file, "w") as f:
        f.write("Best Parameters:\n")
        for param, value in study.best_trial.params.items():
            f.write(f"{param}: {value}\n")

    def train_and_evaluate(model, X_train, y_train, X_test, y_test):
        # Train model
        print("\nTraining Random Forest with optimized hyperparameters...")
        model.fit(X_train, y_train)

        # Get predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nRandom Forest Test Accuracy: {accuracy:.2%}")

        # Print per-position/round accuracy
        print(f"\nPer-{c_name} Accuracy:")
        for label in sorted(y.unique()):
            mask = y_test == label
            if sum(mask) > 0:
                label_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                print(
                    f"{label:4} - Accuracy: {label_accuracy:.2%} (Test samples: {sum(mask)})"
                )

        # Print classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))

        # Get feature importances using encoded feature names
        importances = pd.DataFrame(
            {"feature": encoded_features, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=True)

        # Print feature importances
        print("\nFeature Importances:")
        for idx, row in importances.iterrows():
            print(f"{row['feature']:20}: {row['importance']:.4f}")

        return y_test, y_pred, importances

    # Create and train the model with best parameters
    print("\nBest hyperparameters:", study.best_trial.params)
    rf = create_random_forest(study.best_trial.params)
    y_test, y_pred, importances = train_and_evaluate(
        rf, X_train_scaled, y_train, X_test_scaled, y_test
    )

    # Restore original stdout and save the output to a file
    sys.stdout = original_stdout
    with open(f"{results_path}/logs/training_log.txt", "w") as f:
        f.write(output_buffer.getvalue())

    # Create plotting_data directory
    plotting_data_path = f"{results_path}/plotting_data"
    os.makedirs(plotting_data_path, exist_ok=True)

    # Save predictions to CSV
    pd.DataFrame({f"Actual_{c_name}": y_test, f"Predicted_{c_name}": y_pred}).to_csv(
        f"{plotting_data_path}/predictions.csv", index=False
    )

    # Save optimization history data
    optimization_history = study.trials_dataframe()
    optimization_history.to_csv(
        f"{plotting_data_path}/optimization_history.csv", index=False
    )

    # Save parameter importances data
    param_importances = optuna.importance.get_param_importances(study)
    pd.DataFrame(param_importances.items(), columns=["Parameter", "Importance"]).to_csv(
        f"{plotting_data_path}/param_importances.csv", index=False
    )

    # Save feature importances data
    importances.to_csv(f"{plotting_data_path}/feature_importances.csv", index=False)
