"""
Layer 1, extended across all 6 real operating conditions per split.
Without this, a perfectly healthy motor running at any
other speed/load would likely get falsely flagged as anomalous by the old model, simply
because it never saw a healthy baseline for that condition.

Reuses classifier_mcc5.py's cached multi-condition feature dataset. Layer 1 only needs the "health" rows out of it,
but sharing the extraction pass means Layer 1 and Layer 2 never diverge on what a feature
vector even is.

Validated leave-one-condition-out, but structured to avoid a circular false-positive check:
for each of the 6 conditions, the autoencoder is trained ONLY on healthy windows from the
other 5. The held-out condition's own healthy file is chronologically split into a "baseline" portion 
and a "val" portion. Fault files at the held-out
condition are then scored against that same baseline to check they're actually caught.
"""
import numpy as np
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from load_mcc5 import list_fault_types
from envelope_dataset_mcc5 import artifacts_dir
from classifier_mcc5 import build_dataset, CONDITIONS

SPLITS = ("torque_circulation", "speed_circulation")

HIDDEN_LAYERS = (32, 8, 32)
ALPHA = 0.01 # regularization parameter

MIN_SIGNAL_RATIO = 1.2 
VAL_SIZE = 0.2 


def reconstruction_error(model, X) -> np.ndarray:
    X_hat = model.predict(X)
    return np.mean((X - X_hat) ** 2, axis=1)


def _new_autoencoder():
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=ALPHA,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )


def validate_leave_one_condition_out(split: str):
    X, y, torque_nm, rpm = build_dataset(split=split)
    torque_nm, rpm = torque_nm.astype(int), rpm.astype(int)
    healthy_mask = y == "health"
    fault_types = [f for f in list_fault_types() if f != "health"]

    print(f"=== {split}: leave-one-condition-out anomaly detection ===")
    n_healthy_fp, n_healthy_total = 0, 0
    n_fault_hits, n_fault_total = 0, 0

    for test_t, test_r in CONDITIONS:
        cond_mask = (torque_nm == test_t) & (rpm == test_r)
        train_mask = healthy_mask & ~cond_mask 

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        model = _new_autoencoder()
        model.fit(X_train, X_train)

        held_out_healthy_idx = np.where(cond_mask & healthy_mask)[0]
        baseline_idx, val_idx = train_test_split(held_out_healthy_idx, test_size=VAL_SIZE, shuffle=False)

        healthy_baseline = reconstruction_error(model, scaler.transform(X[baseline_idx])).mean()
        val_ratio = reconstruction_error(model, scaler.transform(X[val_idx])).mean() / healthy_baseline
        false_positive = val_ratio >= MIN_SIGNAL_RATIO
        n_healthy_fp += int(false_positive)
        n_healthy_total += 1

        hits, n_present = 0, 0
        for fault in fault_types:
            fault_mask = cond_mask & (y == fault)
            if fault_mask.sum() == 0:
                continue
            fault_ratio = reconstruction_error(model, scaler.transform(X[fault_mask])).mean() / healthy_baseline
            n_present += 1
            hits += int(fault_ratio >= MIN_SIGNAL_RATIO)
        n_fault_hits += hits
        n_fault_total += n_present

        print(f"  held out {test_t}Nm/{test_r}rpm: healthy_val_ratio={val_ratio:.2f} "
              f"{'[FALSE POSITIVE]' if false_positive else '[ok]'}  "
              f"fault_detection={hits}/{n_present}")

    print(f"\n  healthy files falsely flagged anomalous: {n_healthy_fp}/{n_healthy_total}")
    print(f"  fault files correctly flagged anomalous:  {n_fault_hits}/{n_fault_total} "
          f"({100*n_fault_hits/n_fault_total:.1f}%)\n")


def train_final_model(split: str):
    """Trained on ALL 6 conditions' healthy windows.
    Also saves a per-condition healthy baseline error, needed for the ratio-based scoring
    a real upload would use once its condition is known."""
    X, y, torque_nm, rpm = build_dataset(split=split)
    torque_nm, rpm = torque_nm.astype(int), rpm.astype(int)
    healthy_mask = y == "health"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[healthy_mask])
    model = _new_autoencoder()
    model.fit(X_train, X_train)

    baselines = {}
    for t, r in CONDITIONS:
        m = healthy_mask & (torque_nm == t) & (rpm == r)
        if m.sum() == 0:
            continue
        baselines[(t, r)] = float(reconstruction_error(model, scaler.transform(X[m])).mean())

    out_dir = artifacts_dir(split)
    joblib.dump(model, out_dir / "anomaly_model_multi_condition.pkl", compress=3)  # lossless -- see presence_mcc5
    joblib.dump(scaler, out_dir / "anomaly_scaler_multi_condition.pkl", compress=3)
    joblib.dump(baselines, out_dir / "anomaly_healthy_baselines.pkl", compress=3)
    print(f"{split}: final model trained on {healthy_mask.sum()} healthy windows across "
          f"{len(baselines)} conditions -- saved to {out_dir}")
    return model, baselines


def main():
    for split in SPLITS:
        validate_leave_one_condition_out(split)

    print("training final deployable models on all available data...")
    for split in SPLITS:
        train_final_model(split)


if __name__ == "__main__":
    main()
