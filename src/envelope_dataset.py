"""
Envelope (Hilbert-demodulated) feature representation per window.

Classic MCSA technique: rectify/demodulate the current to get its envelope, then look
for bearing fault frequencies (BPFO/BPFI) directly in the envelope spectrum instead of
as sidebands around the electrical fundamental. Raw magnitude-spectrum sideband features
didn't cleanly separate healthy vs damaged bearings, so this tests whether concentrating
on the demodulated signal isolates the fault content better.

Reuses the same file-level train/val split and window_signal() as windowed_dataset.py.
"""
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler

from load_mat import FS_CURRENT, bearing_fault_frequencies, compute_fft, extract_channels, load_struct
from windowed_dataset import CONDITION, list_condition_files, split_files, window_signal

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "envelope"

ENVELOPE_FAULT_WINDOW_HZ = 2.0  # +/- search window around BPFO/BPFI in the envelope spectrum


def envelope_signal(signal):
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    return envelope - envelope.mean()  # drop DC before spectral analysis


def envelope_fault_magnitude(envelope, fs, fault_hz, window_hz=ENVELOPE_FAULT_WINDOW_HZ):
    freqs, magnitude = compute_fft(envelope, fs)
    mask = (freqs >= fault_hz - window_hz) & (freqs <= fault_hz + window_hz)
    return float(magnitude[mask].max()) if mask.any() else 0.0


def envelope_stats(envelope) -> dict:
    return {
        "rms": float(np.sqrt(np.mean(envelope ** 2))),
        "peak": float(np.max(np.abs(envelope))),
        "kurtosis": float(kurtosis(envelope)),
    }


def windows_for_file(mat_path: Path) -> list:
    struct = load_struct(mat_path)
    channels = extract_channels(struct)

    phase_1 = channels["phase_current_1"]
    phase_2 = channels["phase_current_2"]

    speed_rpm = float(np.mean(channels["speed"]))
    condition_dims = [
        speed_rpm,
        float(np.mean(channels["torque"])),
        float(np.mean(channels["force"])),
    ]
    fault_freqs = bearing_fault_frequencies(shaft_hz=speed_rpm / 60.0)

    windows_1 = window_signal(phase_1)
    windows_2 = window_signal(phase_2)

    rows = []
    for w1, w2 in zip(windows_1, windows_2):
        row = []
        for w in (w1, w2):
            env = envelope_signal(w)
            row.append(envelope_fault_magnitude(env, FS_CURRENT, fault_freqs["bpfo"]))
            row.append(envelope_fault_magnitude(env, FS_CURRENT, fault_freqs["bpfi"]))
            stats = envelope_stats(env)
            row.extend([stats["rms"], stats["peak"], stats["kurtosis"]])
        row.extend(condition_dims)
        rows.append(row)
    return rows


def build_matrix(files) -> np.ndarray:
    rows = []
    for f in files:
        rows.extend(windows_for_file(f))
    return np.array(rows)


def main():
    files = list_condition_files()
    print(f"{len(files)} files for condition {CONDITION}")

    train_files, val_files = split_files(files)
    print(f"train files: {len(train_files)}  val files: {len(val_files)}")

    X_train_raw = build_matrix(train_files)
    X_val_raw = build_matrix(val_files)
    print(f"train windows: {X_train_raw.shape}  val windows: {X_val_raw.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler_envelope.pkl")
    np.save(ARTIFACTS_DIR / "X_train_envelope.npy", X_train)
    np.save(ARTIFACTS_DIR / "X_val_envelope.npy", X_val)
    print(f"Saved scaler_envelope.pkl, X_train_envelope.npy, X_val_envelope.npy to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
