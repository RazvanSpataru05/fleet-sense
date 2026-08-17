import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from load_mat import DATASET_DIR, build_feature_row

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "scalar"

IDENTIFIER_COLS = ['bearing_code', 'condition_code', 'repetition']

def get_ordered_feature_names():

    df = pd.read_csv(ARTIFACTS_DIR / "training_features.csv", nrows=1)
    return [c for c in df.columns if c not in IDENTIFIER_COLS]

def row_to_vector(row: dict, feature_names: list) -> np.ndarray:
    return np.array([[row[name] for name in feature_names]])

def print_stats(label: str, scores: np.ndarray):
    print(f"{label} (n={len(scores)}):")
    print(f"  mean:  {scores.mean():.4f}")
    print(f"  std:   {scores.std():.4f}")
    print(f"  min:   {scores.min():.4f}")
    print(f"  max:   {scores.max():.4f}")

def main():
    scaler = joblib.load(ARTIFACTS_DIR / 'scaler.pkl')
    model = joblib.load(ARTIFACTS_DIR / 'isolation_forest.pkl')
    feature_names = get_ordered_feature_names()

    damaged_file = DATASET_DIR / 'KA04' / 'N09_M07_F10_KA04_1.mat'
    damaged_condition = 'N09_M07_F10'

    row = build_feature_row(damaged_file)
    X_damaged = row_to_vector(row, feature_names)
    X_damaged_scaled = scaler.transform(X_damaged)

    score = model.score_samples(X_damaged_scaled)[0]

    X_val = np.load(ARTIFACTS_DIR / 'X_val.npy')
    condition_val = np.load(ARTIFACTS_DIR / 'condition_val.npy', allow_pickle=True)
    val_scores = model.score_samples(X_val)

    same_condition_mask = condition_val == damaged_condition
    same_condition_scores = val_scores[same_condition_mask]

    print(f"Damaged file:             {damaged_file}")
    print(f"Damaged file score:       {score:.4f}\n")

    print_stats("Healthy val, all conditions mixed", val_scores)
    print(f"  diff from mean: {score - val_scores.mean():.4f}\n")

    print_stats(f"Healthy val, condition == {damaged_condition} only", same_condition_scores)
    print(f"  diff from mean: {score - same_condition_scores.mean():.4f}")

if __name__ == '__main__':
    main()