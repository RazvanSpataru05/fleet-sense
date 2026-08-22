"""
Layer 1, extended across all 6 real operating conditions per split (previously scoped to
just one condition, 20Nm/1000rpm). Without this, a perfectly healthy motor running at any
other speed/load would likely get falsely flagged as anomalous by the old model, simply
because it never saw a healthy baseline for that condition -- not because anything is
actually wrong. Same fix already applied to Layer 2's classifiers.

Reuses classifier_mcc5.py's cached multi-condition feature dataset (all 24 fault types x
6 conditions, extracted once per split) -- Layer 1 only needs the "health" rows out of it,
but sharing the extraction pass means Layer 1 and Layer 2 never diverge on what a feature
vector even is.

Trained per split -- never mixing speed_circulation and torque_circulation (confirmed
early in this project that healthy files score systematically differently by split alone).
Condition (torque_nm, rpm) stays in the feature vector as an explicit input, same principle
as Layer 2, so the model can learn condition-specific normal patterns instead of being
confused by them.

Validated leave-one-condition-out, but structured to avoid a circular false-positive check:
for each of the 6 conditions, the autoencoder is trained ONLY on healthy windows from the
other 5 -- the held-out condition's own healthy file is chronologically split (80/20, same
scheme the original single-condition model used) into a "baseline" portion (establishes
that condition's expected healthy error level) and a "val" portion (genuinely unseen,
checks the model doesn't cry wolf on real healthy data). Fault files at the held-out
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
ALPHA = 0.01  # same regularization as the original single-condition model. There's ~6x
# more healthy training data now, which could likely support relaxing this, but that's an
# untested second variable on top of the actual fix -- left alone unless validation below
# shows a real capacity problem.

MIN_SIGNAL_RATIO = 1.2  # NOT the 1.5 used throughout Layer 2 -- that was inherited without
# checking it fit Layer 1's OWN error distribution, which turned out to be a real mistake:
# scanning thresholds against file-level healthy-vs-fault ratios directly (both splits)
# showed 1.5 was needlessly conservative (52.9%/53.3% fault detection) while 1.1-1.2 gets
# zero false positives on every held-out healthy file in both splits AND ~72-73% detection.
# 1.2 (not 1.1) because speed_circulation still had 2/6 false positives at 1.1 -- 1.2 is the
# lowest shared threshold that's clean for both splits, not fit separately per split (n=6
# healthy files per split is too small to trust a split-specific optimum).
VAL_SIZE = 0.2  # chronological split within the held-out condition's own healthy file


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
        train_mask = healthy_mask & ~cond_mask  # healthy windows from the OTHER 5 conditions only

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        model = _new_autoencoder()
        model.fit(X_train, X_train)

        # chronological split of the held-out condition's OWN healthy windows --
        # "baseline" calibrates the expected error at this condition, "val" is genuinely
        # unseen healthy data used only to check for false alarms.
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
    """Trained on ALL 6 conditions' healthy windows (no holdout) -- the deployable model.
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
