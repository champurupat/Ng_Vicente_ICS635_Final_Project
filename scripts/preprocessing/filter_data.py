import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("data/processed/merged_data_2000_2025.csv")

# Print initial data shape
print(f"Initial data shape: {df.shape}")

# Step 1: Filter players with two or fewer missing values
filtered_df = df[df.isnull().sum(axis=1) <= 2].copy()
print(f"Shape after initial filtering: {filtered_df.shape}")

# Step 2: Predict missing Bench from Wt
bench_subset = filtered_df.dropna(subset=["Wt", "Bench"])
print(f"Number of samples for Bench prediction: {len(bench_subset)}")

if len(bench_subset) > 0:
    bench_model = LinearRegression().fit(bench_subset[["Wt"]], bench_subset["Bench"])
    bench_missing = filtered_df["Bench"].isnull()
    if bench_missing.sum() > 0:
        filtered_df.loc[bench_missing, "Bench"] = bench_model.predict(
            filtered_df.loc[bench_missing, ["Wt"]]
        ).round()
# convert bench to int
filtered_df["Bench"] = filtered_df["Bench"].astype(int)

# Step 3: Predict missing Vertical from Ht and Wt
vertical_subset = filtered_df.dropna(subset=["Ht", "Wt", "Vertical"])
print(f"Number of samples for Vertical prediction: {len(vertical_subset)}")

if len(vertical_subset) > 0:
    vertical_model = LinearRegression().fit(
        vertical_subset[["Ht", "Wt"]], vertical_subset["Vertical"]
    )
    vertical_missing = filtered_df["Vertical"].isnull()
    if vertical_missing.sum() > 0:
        filtered_df.loc[vertical_missing, "Vertical"] = vertical_model.predict(
            filtered_df.loc[vertical_missing, ["Ht", "Wt"]]
        ).round(1)

# Step 4: Fill remaining athletic features
features_to_fill = ["Broad Jump", "40yd", "Shuttle", "3Cone"]
input_features = ["Ht", "Wt"]

for feature in features_to_fill:
    subset = filtered_df.dropna(subset=input_features + [feature])
    print(f"Number of samples for {feature} prediction: {len(subset)}")

    if len(subset) == 0:
        print(f"Skipping {feature}: no training data.")
        continue

    model = LinearRegression().fit(subset[input_features], subset[feature])
    missing = filtered_df[feature].isnull()
    if missing.sum() == 0:
        print(f"No missing values to fill for {feature}.")
        continue

    filtered_df.loc[missing, feature] = model.predict(
        filtered_df.loc[missing, input_features]
    ).round(2)

# Step 5: Remove rows where Pos is LS, G, SAF, P because of low number of samples
filtered_df = filtered_df[~filtered_df["Pos"].isin(["LS", "G", "SAF", "P"])]
print(f"Final shape after position filtering: {filtered_df.shape}")

# Save the final dataset
filtered_df.to_csv("data/processed/filtered_data.csv", index=False)
print(
    "Finished saving filtered and filled dataset to 'data/processed/filtered_data.csv'"
)
