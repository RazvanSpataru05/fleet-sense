from pathlib import Path

import joblib
import numpy as np

from load_mat import DATASET_DIR
from envelope_dataset import windows_for_file
from windowed_dataset import list_condition_files, split_files

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "envelope"

DAMAGED_FILES = {
    "KA04 (OR fault)": DATASET_DIR / "KA04" / "N09_M07_F10_KA04_1.mat",
    "KI04 (IR+OR fault)": DATASET_DIR / "KI04" / "N09_M07_F10_KI04_1.mat",
}


def print_stats(label: str, scores: np.ndarray):
    print(f"{label} (n={len(scores)}):")
    print(f"  mean:  {scores.mean():.4f}")
    print(f"  std:   {scores.std():.4f}")
    print(f"  min:   {scores.min():.4f}")
    print(f"  max:   {scores.max():.4f}")


def score_file(mat_path: Path, scaler, model) -> np.ndarray:
    rows = np.array(windows_for_file(mat_path))
    rows_scaled = scaler.transform(rows)
    return model.score_samples(rows_scaled)


def per_file_worst_scores(files, scaler, model) -> np.ndarray:
    """Worst (min) window score per file -- the correct baseline for comparing against
    another file's worst-window score (see evaluate_windowed.py for why pooling
    individual windows instead is biased)."""
    return np.array([score_file(f, scaler, model).min() for f in files])


def main():
    scaler = joblib.load(ARTIFACTS_DIR / "scaler_envelope.pkl")
    model = joblib.load(ARTIFACTS_DIR / "isolation_forest_envelope.pkl")

    X_val = np.load(ARTIFACTS_DIR / "X_val_envelope.npy")
    val_scores_pooled = model.score_samples(X_val)

    # Same file-level split used to build X_val_envelope.npy (same CONDITION, random_state).
    files = list_condition_files()
    _, val_files = split_files(files)
    healthy_worst_per_file = per_file_worst_scores(val_files, scaler, model)

    print_stats("Healthy val, pooled individual windows", val_scores_pooled)
    print()
    print_stats("Healthy val, per-file worst-window scores (correct baseline)", healthy_worst_per_file)
    print()

    for label, mat_path in DAMAGED_FILES.items():
        scores = score_file(mat_path, scaler, model)
        worst_score = scores.min()
        pct_worst_corrected = (healthy_worst_per_file < worst_score).mean() * 100

        print(f"{label}: {mat_path}")
        print_stats("  per-window scores", scores)
        print(f"  worst-window score: {worst_score:.4f}")
        print(f"  [CORRECT] worst-window percentile vs per-file worst-window healthy baseline: {pct_worst_corrected:.1f}th")
        print()


if __name__ == "__main__":
    main()
