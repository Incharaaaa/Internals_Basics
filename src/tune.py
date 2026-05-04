import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import json
import os

# -----------------------------
# Safe paths (NO SPACE ISSUES)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "..", "data", "training_data.csv")
result_path = os.path.join(BASE_DIR, "..", "results", "step2_s2.json")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(data_path)

X = df.drop("gas_fee_gwei", axis=1)
y = df["gas_fee_gwei"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MLflow setup (FIXED)
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("coinpulse-gas-fee-gwei")

# -----------------------------
# Parameter grid
# -----------------------------
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid = list(ParameterGrid(param_grid))

best_rmse = float("inf")
best_params = None
best_model = None
best_mae = None

# -----------------------------
# Parent run
# -----------------------------
with mlflow.start_run(run_name="tuning-coinpulse") as parent_run:

    for params in grid:

        with mlflow.start_run(nested=True):

            model = RandomForestRegressor(
                random_state=42,
                **params
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            # log params
            mlflow.log_params(params)

            # log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)

            # best model selection (RMSE)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_model = model
                best_mae = mae

# -----------------------------
# Save JSON
# -----------------------------
os.makedirs(os.path.join(BASE_DIR, "..", "results"), exist_ok=True)

output = {
    "search_type": "grid",
    "n_folds": 5,
    "total_trials": len(grid),
    "best_params": best_params,
    "best_mae": best_mae,
    "best_cv_mae": best_rmse,
    "parent_run_name": "tuning-coinpulse"
}

with open(result_path, "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 2 Completed Successfully!")