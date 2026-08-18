"""
Evaluate the MCC5-THU Autoencoder against every fault type at the scoped condition
(20Nm, 1000rpm) -- separately per split. A speed_circulation fault file is only ever
compared against the speed_circulation model and its own held-out healthy baseline,
and likewise for torque_circulation. Confirmed empirically that healthy files score
systematically differently by split alone, so mixing splits in one comparison would
measure "which archive this came from" as much as "healthy vs faulty".

Uses MEAN reconstruction error per file, not worst-of-N: comparing a worst-of-many
against individual draws from a smaller healthy pool is the sample-size bias found and
fixed for Paderborn -- taking the max of more draws is inherently more extreme,
independent of whether anything is actually anomalous. Means aren't subject to that.
"""
import numpy as np
import joblib

from load_mcc5 import SPLIT_DIRS, list_fault_types, list_files
from envelope_dataset_mcc5 import CONDITION, artifacts_dir, build_matrix
from autoencoder_mcc5 import reconstruction_error


def print_stats(label: str, values: np.ndarray):
    print(f"{label} (n={len(values)}):")
    print(f"  mean:  {values.mean():.5f}")
    print(f"  std:   {values.std():.5f}")
    print(f"  min:   {values.min():.5f}")
    print(f"  max:   {values.max():.5f}")


def evaluate_split(split: str):
    print(f"=== {split} ===")
    out_dir = artifacts_dir(split)
    scaler = joblib.load(out_dir / "scaler_envelope.pkl")
    model = joblib.load(out_dir / "autoencoder_mcc5.pkl")

    X_val = np.load(out_dir / "X_val_envelope.npy")
    val_errors = reconstruction_error(model, X_val)
    print_stats("Healthy val windows (held-out half of this split's own recording)", val_errors)
    print()

    fault_types = [f for f in list_fault_types() if f != "health"]
    print(f"{'fault':45s} {'n_win':>6s} {'mean_err':>9s} {'ratio_vs_healthy':>17s}")
    for fault in fault_types:
        files = list_files(fault=fault, split=split, **CONDITION)
        if not files:
            print(f"{fault:45s}  (no file at this condition/split)")
            continue
        rows = np.array(build_matrix(files))
        rows_scaled = scaler.transform(rows)
        errors = reconstruction_error(model, rows_scaled)
        fault_mean = errors.mean()
        ratio = fault_mean / val_errors.mean()
        print(f"{fault:45s} {len(errors):6d} {fault_mean:9.5f} {ratio:16.2f}x")
    print()


def main():
    for split in SPLIT_DIRS:
        evaluate_split(split)


if __name__ == "__main__":
    main()
