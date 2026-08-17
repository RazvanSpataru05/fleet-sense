"""
Build a windowed, wide-spectrum feature dataset from healthy Paderborn recordings,
scoped to a single operating condition (matches the KA04/KI04 test files and the
condition-scoping already validated to help Isolation Forest separation).

Train/val split happens at the FILE level, before windowing, so no window from a
given file can appear in both the train and val sets.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from load_mat import DATASET_DIR, FS_CURRENT, compute_fft, extract_channels, load_struct, parse_filename

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "windowed"

CONDITION = "N09_M07_F10"
RANDOM_STATE = 42
VAL_SIZE = 0.2

WINDOW_SEC = 0.5
OVERLAP = 0.5  # fraction of window length
WINDOW_SAMPLES = int(WINDOW_SEC * FS_CURRENT)
STRIDE_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))

FREQ_BAND = (0.0, 150.0)  # Hz, wide spectral band kept per window


def list_condition_files(condition=CONDITION):
    files = sorted(DATASET_DIR.glob("K0*/*.mat"))
    return [f for f in files if parse_filename(f)[1] == condition]


def split_files(files, val_size=VAL_SIZE, random_state=RANDOM_STATE):
    return train_test_split(files, test_size=val_size, random_state=random_state)


def window_signal(signal, window_samples=WINDOW_SAMPLES, stride_samples=STRIDE_SAMPLES):
    n = len(signal)
    starts = range(0, n - window_samples + 1, stride_samples)
    return [signal[s:s + window_samples] for s in starts]


def spectral_bins(signal, fs=FS_CURRENT, band=FREQ_BAND):
    freqs, magnitude = compute_fft(signal, fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return magnitude[mask]


def windows_for_file(mat_path: Path) -> list:
    struct = load_struct(mat_path)
    channels = extract_channels(struct)

    phase_1 = channels["phase_current_1"]
    phase_2 = channels["phase_current_2"]

    condition_dims = [
        float(np.mean(channels["speed"])),
        float(np.mean(channels["torque"])),
        float(np.mean(channels["force"])),
    ]

    windows_1 = window_signal(phase_1)
    windows_2 = window_signal(phase_2)

    rows = []
    for w1, w2 in zip(windows_1, windows_2):
        spec_1 = spectral_bins(w1)
        spec_2 = spectral_bins(w2)
        rows.append(np.concatenate([spec_1, spec_2, condition_dims]))
    return rows


def build_windowed_matrix(files) -> np.ndarray:
    rows = []
    for f in files:
        rows.extend(windows_for_file(f))
    return np.array(rows)


def main():
    files = list_condition_files()
    print(f"{len(files)} files for condition {CONDITION}")

    train_files, val_files = split_files(files)
    assert not (set(train_files) & set(val_files)), "train/val file overlap detected"
    print(f"train files: {len(train_files)}  val files: {len(val_files)}")

    X_train_raw = build_windowed_matrix(train_files)
    X_val_raw = build_windowed_matrix(val_files)
    print(f"train windows: {X_train_raw.shape}  val windows: {X_val_raw.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler_windowed.pkl")
    np.save(ARTIFACTS_DIR / "X_train_windowed.npy", X_train)
    np.save(ARTIFACTS_DIR / "X_val_windowed.npy", X_val)
    print(f"Saved scaler_windowed.pkl, X_train_windowed.npy, X_val_windowed.npy to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
