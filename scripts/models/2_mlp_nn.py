import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
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
    results_path = f"results/2_mlp_nn/{p_set}"
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

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and preprocess data
    df = pd.read_csv("data/processed/filtered_data.csv")
    features = f_set
    X = df[features]
    y = df[c_name]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    # Split the data using the same seed as previous models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
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

        # Get feature names after transformation and update input shape
        encoded_features = numerical_features + [
            f"Pos_{cat}"
            for cat in preprocessor.named_transformers_["cat"].categories_[0]
        ]
        input_shape = len(encoded_features)
    else:
        # For position prediction (all numerical features)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        encoded_features = features
        input_shape = len(features)

    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            "units_layer1": trial.suggest_int("units_layer1", 32, 256),
            "units_layer2": trial.suggest_int("units_layer2", 16, 128),
            "dropout1": trial.suggest_float("dropout1", 0.1, 0.5),
            "dropout2": trial.suggest_float("dropout2", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_index, val_index in cv.split(X_train_scaled, y_train):
            X_train_fold = X_train_scaled[train_index]
            y_train_fold = y_train[train_index]
            X_val_fold = X_train_scaled[val_index]
            y_val_fold = y_train[val_index]

            # Create model with trial parameters
            inputs = layers.Input(shape=(input_shape,))
            x = layers.Dense(params["units_layer1"], activation="relu")(inputs)
            x = layers.Dropout(params["dropout1"])(x)
            x = layers.Dense(params["units_layer2"], activation="relu")(x)
            x = layers.Dropout(params["dropout2"])(x)
            outputs = layers.Dense(n_classes, activation="softmax")(x)
            model = models.Model(inputs=inputs, outputs=outputs)

            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )

            # Train model
            history = model.fit(
                X_train_fold,
                y_train_fold,
                epochs=50,
                batch_size=params["batch_size"],
                validation_data=(X_val_fold, y_val_fold),
                verbose=0,
                callbacks=[early_stopping],
            )

            # Evaluate model on validation set
            val_accuracy = max(history.history["val_accuracy"])
            scores.append(val_accuracy)

        accuracy = np.mean(scores)

        # Log progress
        log_progress(f"Trial {trial.number}:")
        log_progress(f"Params: {params}")
        log_progress(f"Best Validation Accuracy: {accuracy:.4f}\n")

        return accuracy

    def create_basic_nn(best_params):
        inputs = layers.Input(shape=(input_shape,))
        x = layers.Dense(best_params["units_layer1"], activation="relu")(inputs)
        x = layers.Dropout(best_params["dropout1"])(x)
        x = layers.Dense(best_params["units_layer2"], activation="relu")(x)
        x = layers.Dropout(best_params["dropout2"])(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        model = models.Model(inputs=inputs, outputs=outputs)
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

    def train_and_evaluate(
        model, model_name, X_train, y_train, X_test, y_test, best_params
    ):
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Train model
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=best_params["batch_size"],
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stopping],
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n{model_name} Test Accuracy: {test_accuracy:.2%}")

        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Convert predictions back to original labels
        y_test_original = label_encoder.inverse_transform(y_test)
        y_pred_original = label_encoder.inverse_transform(y_pred_classes)

        return y_test_original, y_pred_original, history

    # Train and evaluate Basic Neural Network with best parameters
    print("\nBest hyperparameters:", study.best_trial.params)
    basic_nn = create_basic_nn(study.best_trial.params)
    y_test_basic, y_pred_basic, history_optimal = train_and_evaluate(
        basic_nn,
        "Basic NN",
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        study.best_trial.params,
    )

    # Restore original stdout and save the output to a file
    sys.stdout = original_stdout
    with open(f"{results_path}/logs/training_log.txt", "w") as f:
        f.write(output_buffer.getvalue())

    # Create plotting_data directory
    plotting_data_path = f"{results_path}/plotting_data"
    os.makedirs(plotting_data_path, exist_ok=True)

    # Save predictions to CSV
    pd.DataFrame(
        {f"Actual_{c_name}": y_test_basic, f"Predicted_{c_name}": y_pred_basic}
    ).to_csv(f"{plotting_data_path}/predictions.csv", index=False)

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

    # Save training history data
    pd.DataFrame(history_optimal.history).to_csv(
        f"{plotting_data_path}/training_history.csv", index=False
    )
