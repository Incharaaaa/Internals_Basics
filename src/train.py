import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import json
import os
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../data/training_data.csv")

X = df.drop("gas_fee_gwei", axis=1)
y = df["gas_fee_gwei"]

# Train-test split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MLflow Setup
# -----------------------------
mlflow.set_experiment("coinpulse-gas-fee-gwei")

models = {
    "SVR": SVR(C=1.0, epsilon=0.2, kernel='rbf'),
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
}

results = []
best_rmse = float("inf")
best_model_name = None
best_model = None

# -----------------------------
# Train & Log Models
# -----------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log params (IMPORTANT FOR MARKS)
        for param, value in model.get_params().items():
            mlflow.log_param(param, value)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Tag
        mlflow.set_tag("team", "ml_engineering")

        # Save model in MLflow
        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

        # Select best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model

# -----------------------------
# Save Best Model
# -----------------------------
os.makedirs("../models", exist_ok=True)
joblib.dump(best_model, "../models/best_model.pkl")

# -----------------------------
# Save JSON Output
# -----------------------------
output = {
    "experiment_name": "coinpulse-gas-fee-gwei",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": best_rmse
}

os.makedirs("../results", exist_ok=True)

with open("../results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 1 Completed Successfully!")