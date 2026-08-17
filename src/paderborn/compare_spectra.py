"""
Quick visual check: does phase_current_1's FFT actually look different between
a healthy bearing and a damaged one, under the same operating condition?
"""
from pathlib import Path

import matplotlib.pyplot as plt

from load_mat import DATASET_DIR, compute_fft, extract_channels, load_struct

FS_CURRENT = 64000  # Hz

HEALTHY_FILE = DATASET_DIR / "K001" / "N09_M07_F10_K001_1.mat"
DAMAGED_FILE = DATASET_DIR / "KA04" / "N09_M07_F10_KA04_1.mat"


def load_phase_current_1(mat_path: Path):
    struct = load_struct(mat_path)
    channels = extract_channels(struct)
    return channels["phase_current_1"]


def main():
    healthy_signal = load_phase_current_1(HEALTHY_FILE)
    damaged_signal = load_phase_current_1(DAMAGED_FILE)

    freqs_healthy, mag_healthy = compute_fft(healthy_signal, FS_CURRENT)
    freqs_damaged, mag_damaged = compute_fft(damaged_signal, FS_CURRENT)

    fig, (ax_wide, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8))

    for ax, xlim in ((ax_wide, (0, 500)), (ax_zoom, (0, 150))):
        ax.plot(freqs_healthy, mag_healthy, label=f"Healthy ({HEALTHY_FILE.parent.name})", alpha=0.8)
        ax.plot(freqs_damaged, mag_damaged, label=f"Damaged ({DAMAGED_FILE.parent.name})", alpha=0.8)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax_wide.set_title("phase_current_1 FFT — 0-500 Hz")
    ax_zoom.set_title("phase_current_1 FFT — 0-150 Hz (fundamental region)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
