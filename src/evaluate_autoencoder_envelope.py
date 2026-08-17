from pathlib import Path

import joblib
import numpy as np

from load_mat import DATASET_DIR
from envelope_dataset import windows_for_file
from windowed_dataset import list_condition_files, split_files
from autoencoder_envelope import reconstruction_error

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "envelope"

DAMAGED_FILES = {
    # rep 1 avoided deliberately -- see evaluate_envelope.py for why.
    "KA04 (OR fault)": DATASET_DIR / "KA04" / "N09_M07_F10_KA04_2.mat",
    "KI04 (IR+OR fault)": DATASET_DIR / "KI04" / "N09_M07_F10_KI04_2.mat",
}


def print_stats(label: str, values: np.ndarray):
    print(f"{label} (n={len(values)}):")
    print(f"  mean:  {values.mean():.5f}")
    print(f"  std:   {values.std():.5f}")
    print(f"  min:   {values.min():.5f}")
    print(f"  max:   {values.max():.5f}")


def score_file(mat_path: Path, scaler, model) -> np.ndarray:
    rows = np.array(windows_for_file(mat_path))
    rows_scaled = scaler.transform(rows)
    return reconstruction_error(model, rows_scaled)


def per_file_worst_errors(files, scaler, model) -> np.ndarray:
    """Worst (max reconstruction error) window per file -- same fair, apples-to-apples
    baseline used for the Isolation Forest worst-window comparison (see evaluate_envelope.py)."""
    return np.array([score_file(f, scaler, model).max() for f in files])


def main():
    scaler = joblib.load(ARTIFACTS_DIR / "scaler_envelope.pkl")
    model = joblib.load(ARTIFACTS_DIR / "autoencoder_envelope.pkl")

    X_val = np.load(ARTIFACTS_DIR / "X_val_envelope.npy")
    val_errors_pooled = reconstruction_error(model, X_val)

    files = list_condition_files()
    _, val_files = split_files(files)
    healthy_worst_per_file = per_file_worst_errors(val_files, scaler, model)

    print_stats("Healthy val, pooled individual windows", val_errors_pooled)
    print()
    print_stats("Healthy val, per-file worst-window error (correct baseline)", healthy_worst_per_file)
    print()

    for label, mat_path in DAMAGED_FILES.items():
        errors = score_file(mat_path, scaler, model)
        worst_error = errors.max()
        pct_worse = (healthy_worst_per_file < worst_error).mean() * 100

        print(f"{label}: {mat_path}")
        print_stats("  per-window reconstruction error", errors)
        print(f"  worst-window error: {worst_error:.5f}")
        print(f"  percentile vs per-file worst-window healthy baseline: {pct_worse:.1f}th")
        print()


if __name__ == "__main__":
    main()
