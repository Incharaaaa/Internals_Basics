import argparse
import joblib
import numpy as np
import os

# Fix path issue
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument("--network_congestion", type=float, required=True)
parser.add_argument("--txn_size_bytes", type=int, required=True)
parser.add_argument("--is_priority", type=int, required=True)
parser.add_argument("--block_fullness_pct", type=float, required=True)

args = parser.parse_args()

# Load model
model = joblib.load(model_path)

# Prepare input
features = np.array([[
    args.network_congestion,
    args.txn_size_bytes,
    args.is_priority,
    args.block_fullness_pct
]])

# Predict
prediction = model.predict(features)[0]

print(prediction)