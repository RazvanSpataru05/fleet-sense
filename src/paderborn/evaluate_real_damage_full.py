"""
Evaluate the Autoencoder against the full pool of REAL bearing-damage recordings
(KI + KB codes -- both are lifetime-test-induced natural damage, confirmed via their
damage-profile PDFs, unlike the KA codes which are artificially induced). Same
worst-window-per-file methodology validated on KA04/KI04 rep 2, scaled up from 2
spot-check files to every real-damage file available at the trained condition.
"""
import numpy as np
import joblib

from load_mat import DATASET_DIR, parse_filename
from windowed_dataset import CONDITION, EXCLUDE_REPETITIONS, list_condition_files, split_files
from evaluate_autoencoder_envelope import ARTIFACTS_DIR, per_file_worst_errors, print_stats, score_file

REAL_DAMAGE_CODES = [
    # Verified via each bearing's own damage-profile PDF -- NOT by prefix, since neither
    # "KI" nor "KA" reliably indicates real vs artificial damage (checked case by case).
    # Real (lifetime test, actual operating data): these 10.
    # Confirmed artificial despite prefix, excluded: KI01, KI03, KI05, KI07, KI08
    # ("artificial damage, bearing was not operated" -- electric engraver, never run).
    "KA04", "KA15", "KA16", "KA22", "KA30",
    "KI04", "KI14", "KI16", "KI17", "KI18", "KI21",
    "KB23", "KB24", "KB27",
]


def list_damage_files(code: str, condition=CONDITION, exclude_reps=EXCLUDE_REPETITIONS):
    files = sorted((DATASET_DIR / code).glob("*.mat"))
    return [
        f for f in files
        if parse_filename(f)[1] == condition and parse_filename(f)[2] not in exclude_reps
    ]


def main():
    scaler = joblib.load(ARTIFACTS_DIR / "scaler_envelope.pkl")
    model = joblib.load(ARTIFACTS_DIR / "autoencoder_envelope.pkl")

    healthy_files = list_condition_files()
    _, healthy_val_files = split_files(healthy_files)
    healthy_worst_per_file = per_file_worst_errors(healthy_val_files, scaler, model)

    print_stats("Healthy val, per-file worst-window error (baseline)", healthy_worst_per_file)
    print()

    all_pct = []
    print(f"{'code':6s} {'n':>3s} {'mean%ile':>9s} {'min%ile':>8s} {'max%ile':>8s} {'>=90th':>7s}")
    for code in REAL_DAMAGE_CODES:
        damage_files = list_damage_files(code)
        pct_list = []
        for f in damage_files:
            errors = score_file(f, scaler, model)
            worst_error = errors.max()
            pct = (healthy_worst_per_file < worst_error).mean() * 100
            pct_list.append(pct)
        pct_arr = np.array(pct_list)
        all_pct.extend(pct_list)
        flagged_90 = (pct_arr >= 90).mean() * 100
        print(f"{code:6s} {len(pct_arr):3d} {pct_arr.mean():9.1f} {pct_arr.min():8.1f} {pct_arr.max():8.1f} {flagged_90:6.0f}%")

    all_pct = np.array(all_pct)
    print()
    print(f"OVERALL across {len(all_pct)} real-damage files:")
    print(f"  mean percentile:                    {all_pct.mean():.1f}")
    print(f"  files >= 90th percentile:            {(all_pct >= 90).mean()*100:.1f}%")
    print(f"  files >= 95th percentile:            {(all_pct >= 95).mean()*100:.1f}%")
    print(f"  files at 100th percentile:           {(all_pct >= 100).mean()*100:.1f}%")
    print(f"  files below 50th percentile:         {(all_pct < 50).mean()*100:.1f}%  (would look healthy)")


if __name__ == "__main__":
    main()
