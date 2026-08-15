"""
Build a feature-vector CSV from the Paderborn bearing dataset (K001-K006).

Each .mat file holds a ~4s recording of two stator phase currents (plus
speed/torque/force/vibration/temperature channels) sampled while the motor
runs under one of four fixed operating conditions. This script turns every
.mat file into a single feature row and writes them all to a CSV.
"""
import csv
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis

DATASET_DIR = Path("dataset")
OUTPUT_CSV = Path("features.csv")

FS_CURRENT = 64000  # Hz, sample rate of phase_current_1/2
FUNDAMENTAL_SEARCH_BAND = (5.0, 200.0)  # Hz, where to look for the electrical fundamental
HARMONIC_WINDOW_HZ = 2.0  # +/- window around k*fundamental when reading off harmonic magnitude
N_HARMONICS = 5

# e.g. "N09_M07_F10_K001_1" -> speed code, torque code, force code, bearing code, repetition
FILENAME_RE = re.compile(r"^(N\d{2})_(M\d{2})_(F\d{2})_(K\d{3})_(\d+)$")


def load_struct(mat_path: Path):
    var_name = mat_path.stem
    try:
        raw = loadmat(mat_path)
    except NotImplementedError:
        import mat73
        raw = mat73.loadmat(mat_path)
    return raw[var_name]


def extract_channels(struct) -> dict:
    channels = struct["Y"][0, 0][0]
    data = {}
    for ch in channels:
        name = ch["Name"][0]
        data[name] = ch["Data"][0].astype(float)
    return data


def compute_fft(signal, fs):
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = np.abs(fft_vals) / n
    return freqs, magnitude


def detect_fundamental_hz(signal, fs):
    freqs, magnitude = compute_fft(signal, fs)
    lo, hi = FUNDAMENTAL_SEARCH_BAND
    mask = (freqs >= lo) & (freqs <= hi)
    band_freqs = freqs[mask]
    band_mag = magnitude[mask]
    return float(band_freqs[np.argmax(band_mag)])


def harmonic_magnitudes(signal, fs, fundamental_hz, n_harmonics=N_HARMONICS):
    freqs, magnitude = compute_fft(signal, fs)
    mags = []
    for k in range(1, n_harmonics + 1):
        target = k * fundamental_hz
        mask = (freqs >= target - HARMONIC_WINDOW_HZ) & (freqs <= target + HARMONIC_WINDOW_HZ)
        mags.append(float(magnitude[mask].max()) if mask.any() else 0.0)
    return mags


def time_domain_features(signal) -> dict:
    mean_square = float(np.mean(signal ** 2))
    rms = float(np.sqrt(mean_square))
    peak_abs = float(np.max(np.abs(signal)))
    crest_factor = peak_abs / rms if rms > 0 else 0.0
    return {
        "mean_square": mean_square,
        "rms": rms,
        "peak_abs": peak_abs,
        "crest_factor": crest_factor,
        "kurtosis": float(kurtosis(signal)),
    }


def parse_filename(mat_path: Path):
    match = FILENAME_RE.match(mat_path.stem)
    if not match:
        raise ValueError(f"Filename doesn't match expected pattern: {mat_path.name}")
    speed_code, torque_code, force_code, bearing_code, repetition = match.groups()
    condition_code = f"{speed_code}_{torque_code}_{force_code}"
    return bearing_code, condition_code, int(repetition)


def build_feature_row(mat_path: Path) -> dict:
    bearing_code, condition_code, repetition = parse_filename(mat_path)

    struct = load_struct(mat_path)
    channels = extract_channels(struct)

    phase_1 = channels["phase_current_1"]
    phase_2 = channels["phase_current_2"]

    fundamental_hz = detect_fundamental_hz(phase_1, FS_CURRENT)

    row = {
        "bearing_code": bearing_code,
        "condition_code": condition_code,
        "repetition": repetition,
        "speed_rpm": float(np.mean(channels["speed"])),
        "torque_nm": float(np.mean(channels["torque"])),
        "radial_force_n": float(np.mean(channels["force"])),
        "detected_fundamental_hz": fundamental_hz,
    }

    for phase_num, signal in ((1, phase_1), (2, phase_2)):
        prefix = f"phase{phase_num}"

        for feat_name, value in time_domain_features(signal).items():
            row[f"{prefix}_{feat_name}"] = value

        for k, mag in enumerate(harmonic_magnitudes(signal, FS_CURRENT, fundamental_hz), start=1):
            row[f"{prefix}_mag_h{k}"] = mag

    return row


def main():
    mat_files = sorted(DATASET_DIR.glob("K0*/*.mat"))
    print(f"Found {len(mat_files)} .mat files under {DATASET_DIR}/")

    rows = []
    for i, mat_path in enumerate(mat_files, start=1):
        print(f"[{i}/{len(mat_files)}] {mat_path.relative_to(DATASET_DIR)}")
        try:
            rows.append(build_feature_row(mat_path))
        except Exception as exc:
            print(f"  FAILED: {exc}")

    if not rows:
        print("No rows extracted, nothing written.")
        return

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
