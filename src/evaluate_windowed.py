from pathlib import Path

import joblib
import numpy as np

from load_mat import DATASET_DIR
from windowed_dataset import list_condition_files, split_files, windows_for_file

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "windowed"

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
    another file's worst-window score, since max-of-N-windows is systematically more
    extreme than a single window, purely from having more draws."""
    return np.array([score_file(f, scaler, model).min() for f in files])


def main():
    scaler = joblib.load(ARTIFACTS_DIR / "scaler_windowed.pkl")
    model = joblib.load(ARTIFACTS_DIR / "isolation_forest_windowed.pkl")

    X_train = np.load(ARTIFACTS_DIR / "X_train_windowed.npy")
    X_val = np.load(ARTIFACTS_DIR / "X_val_windowed.npy")
    healthy_scores_pooled = model.score_samples(np.vstack([X_train, X_val]))
    val_scores_pooled = model.score_samples(X_val)

    # Recompute the same file-level split used to build X_val_windowed.npy (same random_state),
    # so we can group val windows back by source file for the per-file worst-window baseline.
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

        pct_ranks_pooled = (healthy_scores_pooled[None, :] < scores[:, None]).mean(axis=1) * 100
        pct_worst_biased = (healthy_scores_pooled < worst_score).mean() * 100
        pct_worst_corrected = (healthy_worst_per_file < worst_score).mean() * 100

        print(f"{label}: {mat_path}")
        print_stats("  per-window scores", scores)
        print(f"  diff from healthy-val mean: {scores.mean() - val_scores_pooled.mean():.4f}")
        print(f"  mean percentile rank vs pooled healthy windows: {pct_ranks_pooled.mean():.1f}th")
        print(f"  worst-window score: {worst_score:.4f}")
        print(f"  [BIASED] worst-window percentile vs pooled individual healthy windows: {pct_worst_biased:.1f}th")
        print(f"  [CORRECT] worst-window percentile vs per-file worst-window healthy baseline: {pct_worst_corrected:.1f}th")
        print()


if __name__ == "__main__":
    main()
