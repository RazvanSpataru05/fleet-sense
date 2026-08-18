"""
Envelope-based feature extraction for the MCC5-THU dataset.

The dataset's own paper (Chen, Liu, Li, Zou, He & Zhou, Data in Brief 2026) gives the
exact bearing used (SKF 6205 2Z-C3) and its fault-frequency ratios directly: BPFO =
3.585x shaft frequency, BPFI = 5.415x, BSF = 2.357x. These were cross-checked against
the standard bearing-geometry formula using the paper's own pitch/ball diameter and
ball count and matched exactly, confirming both the geometry and a 0 degree contact
angle. We add these as three precise, targeted features per phase, on top of the wide
envelope-spectrum band from the first pass -- the targeted features should carry the
real signal for bearing-related faults, while the wide band is kept for fault types
that don't have a known formula yet (rotor bar, eccentricity, winding).

Scoped to a single operating condition (20Nm, 1000rpm) for the same reason we scoped
Paderborn to one condition -- mixing conditions dilutes the anomaly signal.
"""
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from load_mcc5 import FS, SPLIT_DIRS, list_files, list_fault_types, load_recording, parse_filename

BASE_DIR = Path(__file__).resolve().parent


def artifacts_dir(split: str) -> Path:
    """Each split gets its own model/scaler/artifacts -- comparing a speed_circulation
    fault file against a torque_circulation healthy baseline (or vice versa) mixes in
    a systematic split-level difference unrelated to fault status (confirmed: healthy
    files score very differently by split alone). Keeping everything split-scoped
    avoids that confound entirely instead of trying to correct for it after the fact."""
    return BASE_DIR / "artifacts" / "envelope" / split


CONDITION = {"torque_nm": 20, "rpm": 1000}
RANDOM_STATE = 42
VAL_SIZE = 0.2

WINDOW_SEC = 0.5
OVERLAP = 0.5
WINDOW_SAMPLES = int(WINDOW_SEC * FS)
STRIDE_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))

FUNDAMENTAL_SEARCH_BAND = (10.0, 30.0)  # Hz, tight around the known ~16.67Hz at 1000rpm
ENVELOPE_BAND = (0.0, 200.0)  # Hz, ~12x shaft frequency -- wide enough for any common bearing's BPFO/BPFI
BIN_GROUP = 4  # average-pool this many native (2Hz) bins together -> ~8Hz effective resolution.
# Only 359 windows exist at this single condition (no repeated files like Paderborn);
# native resolution would give 101 bins/phase x 3 phases = 303 dims against ~287 training
# rows, close to 1:1 and a serious overfitting risk. Pooling trades resolution for a
# dimensionality the sample count can actually support.

# SKF 6205 2Z-C3, from the dataset's own paper (Table 7) -- fault frequency as a
# multiple of shaft rotational frequency. Cross-checked against the standard geometry
# formula using the paper's own pitch diameter (39.04mm), ball diameter (7.94mm), ball
# count (9), assuming 0 degree contact angle (standard for a deep-groove bearing) --
# matched all three ratios exactly.
BEARING_FAULT_RATIOS = {"bpfo": 3.585, "bpfi": 5.415, "bsf": 2.357}
TARGETED_FAULT_WINDOW_HZ = 2.0  # +/- search window around each target frequency

CURRENT_CHANNELS = ["current_a", "current_b", "current_c"]


def healthy_file(split: str) -> Path:
    files = list_files(fault="health", split=split, **CONDITION)
    assert len(files) == 1, f"expected exactly one healthy file for split={split}, got {len(files)}"
    return files[0]


def condition_files(split: str) -> list:
    return list_files(split=split, **CONDITION)


def split_rows(rows, val_size=VAL_SIZE):
    """Window-level, chronological (not shuffled) split within the single healthy
    recording available for this split+condition. Chronological rather than random:
    with 50% window overlap, a random shuffle would scatter near-duplicate neighboring
    windows across both train and val, leaking information between them. A time-ordered
    split confines the overlap to the single boundary between the two halves."""
    return train_test_split(rows, test_size=val_size, shuffle=False)


def window_signal(signal, window_samples=WINDOW_SAMPLES, stride_samples=STRIDE_SAMPLES):
    n = len(signal)
    starts = range(0, n - window_samples + 1, stride_samples)
    return [signal[s:s + window_samples] for s in starts]


def compute_fft(signal, fs=FS):
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = np.abs(fft_vals) / n
    return freqs, magnitude


def detect_fundamental_hz(freqs, magnitude, band=FUNDAMENTAL_SEARCH_BAND):
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    band_freqs, band_mag = freqs[mask], magnitude[mask]
    return float(band_freqs[np.argmax(band_mag)])


def shaft_hz_for_file(df) -> float:
    """Detected once per file from the full ~90s recording, not per 0.5s window --
    a short window's FFT frequency resolution is coarse enough that fundamental
    detection got noticeably unstable in spot checks (dropped to 10-15Hz instead of
    ~16.7Hz in some 10s slices). The full recording gives ~0.011Hz resolution and a
    stable estimate that's reused for every window in that file."""
    freqs, magnitude = compute_fft(df["current_a"].values)
    return detect_fundamental_hz(freqs, magnitude)


def bearing_fault_frequencies(shaft_hz: float, ratios=BEARING_FAULT_RATIOS) -> dict:
    return {name: ratio * shaft_hz for name, ratio in ratios.items()}


def targeted_fault_magnitude(freqs, magnitude, fault_hz, window_hz=TARGETED_FAULT_WINDOW_HZ):
    mask = (freqs >= fault_hz - window_hz) & (freqs <= fault_hz + window_hz)
    return float(magnitude[mask].max()) if mask.any() else 0.0


def envelope_signal(signal):
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    return envelope - envelope.mean()


def envelope_spectrum_bins(freqs, magnitude, band=ENVELOPE_BAND, bin_group=BIN_GROUP):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    values = magnitude[mask]
    n_groups = len(values) // bin_group
    trimmed = values[: n_groups * bin_group]
    # max, not mean: a real fault signature is a narrow spike, and averaging it with
    # neighboring quiet bins divides its visible height by ~bin_group, diluting exactly
    # the signal we're trying to detect. Max preserves peak height under the same
    # dimensionality reduction.
    return trimmed.reshape(n_groups, bin_group).max(axis=1)


def envelope_stats(envelope) -> dict:
    return {
        "rms": float(np.sqrt(np.mean(envelope ** 2))),
        "peak": float(np.max(np.abs(envelope))),
        "kurtosis": float(kurtosis(envelope)),
    }


def windows_for_file(csv_path: Path) -> list:
    df = load_recording(csv_path)
    meta = parse_filename(csv_path)

    shaft_hz = shaft_hz_for_file(df)
    fault_freqs = bearing_fault_frequencies(shaft_hz)

    windows_per_channel = {ch: window_signal(df[ch].values) for ch in CURRENT_CHANNELS}
    torque_measured = float(df["torque"].mean())

    n_windows = len(windows_per_channel[CURRENT_CHANNELS[0]])
    rows = []
    for i in range(n_windows):
        # Skip degenerate windows (raw signal is exactly flat -- a trailing
        # data-acquisition artifact seen at the very end of one recording).
        # Kurtosis of a zero-variance signal is mathematically undefined (NaN).
        if any(windows_per_channel[ch][i].std() < 1e-9 for ch in CURRENT_CHANNELS):
            continue

        row = []
        for ch in CURRENT_CHANNELS:
            w = windows_per_channel[ch][i]
            env = envelope_signal(w)
            freqs, magnitude = compute_fft(env)

            row.extend(envelope_spectrum_bins(freqs, magnitude))

            stats = envelope_stats(env)
            row.extend([stats["rms"], stats["peak"], stats["kurtosis"]])

            for fault_hz in fault_freqs.values():
                row.append(targeted_fault_magnitude(freqs, magnitude, fault_hz))
        row.extend([meta["torque_nm"], meta["rpm"], torque_measured, shaft_hz])
        rows.append(row)
    return rows


def build_matrix(files) -> np.ndarray:
    rows = []
    for f in files:
        rows.extend(windows_for_file(f))
    return np.array(rows)


def build_for_split(split: str):
    print(f"=== {split} ===")
    f = healthy_file(split)
    print(f"healthy file: {f.name}")

    rows = windows_for_file(f)
    print(f"total windows: {len(rows)}")

    train_rows, val_rows = split_rows(rows)
    X_train_raw = np.array(train_rows)
    X_val_raw = np.array(val_rows)
    print(f"train windows: {X_train_raw.shape}  val windows: {X_val_raw.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    out_dir = artifacts_dir(split)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_dir / "scaler_envelope.pkl")
    np.save(out_dir / "X_train_envelope.npy", X_train)
    np.save(out_dir / "X_val_envelope.npy", X_val)
    print(f"Saved to {out_dir}\n")


def main():
    for split in SPLIT_DIRS:
        build_for_split(split)


if __name__ == "__main__":
    main()
