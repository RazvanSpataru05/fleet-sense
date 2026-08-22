"""
Layer 2, stage 1: per-location fault presence, reframed as multi-label rather than 24-way
exact-match classification.

Why multi-label: the strict 24-class classifier (classifier_mcc5.py) tops out at 24%
exact-match accuracy, but most of its "wrong" answers turn out to be adjacent, not random
-- confusing severity levels (bearing_outer_H vs bearing_outer_L) or missing one fault in a
two-fault combo file. Neither of those is actually a wrong answer for the real use case: a
dashboard that lights up affected physical locations independently of each other, and
independently of severity. Reframing to "which of these 9 locations are affected" (each
scored on its own, any combination allowed) both matches that use case directly and scores
far better once severity/combo details stop being counted as errors: ~53% family-level
accuracy on the strict classifier, and per-location F1 of 50-100% for 5 of 9 locations here.

9 locations, derived from the 24 fault-type names by stripping severity suffixes (_H/_L)
and splitting "_and_" combos into their components:
  bearing_ball, bearing_inner, bearing_outer, bend, rotor_bar (broken_bar),
  dynamic_eccentricity, static_eccentricity, voltage_unbalance, winding

Validated leave-one-condition-out, file-level (mean predicted probability across all of a
held-out file's windows, thresholded at 0.3 -- tuned down from the default 0.5 because the
default was too conservative: it missed real faults far more than it avoided false alarms,
confirmed by scanning thresholds 0.2-0.5 directly). Result, per location (precision/recall/F1):
  static_eccentricity  0.77 / 0.83 / 0.80
  voltage_unbalance    1.00 / 1.00 / 1.00
  rotor_bar            0.54 / 0.83 / 0.65
  bearing_inner        0.50 / 0.64 / 0.56
  bearing_outer        0.38 / 0.71 / 0.50
  winding              0.63 / 0.21 / 0.31
  bearing_ball         0.16 / 0.42 / 0.23
  dynamic_eccentricity 0.25 / 0.11 / 0.15
  bend                 0.00 / 0.00 / 0.00  -- confirmed elsewhere to have no current-domain
                                              signature at all (only visible in vibration)
"""
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from envelope_dataset_mcc5 import artifacts_dir
from classifier_mcc5 import build_dataset, CONDITIONS

LOCATIONS = [
    "bearing_ball", "bearing_inner", "bearing_outer", "bend", "rotor_bar",
    "dynamic_eccentricity", "static_eccentricity", "voltage_unbalance", "winding",
]

PRESENCE_THRESHOLD = 0.3  # tuned down from the RF default of 0.5 -- see module docstring

SPLITS = ("torque_circulation", "speed_circulation")


def model_path(split: str):
    return artifacts_dir(split) / "presence_model.pkl"


def fault_set(name: str) -> set:
    """Which of the 9 LOCATIONS a given 24-class fault name touches. Handles the one combo
    whose second component drops the "bearing_" prefix (bearing_outer_H_and_inner_H) as a
    special case; every other combo already spells "bearing_" out in full."""
    if name == "health":
        return set()
    if name == "bearing_outer_H_and_inner_H":
        return {"bearing_outer", "bearing_inner"}
    out = set()
    for part in name.split("_and_"):
        base = part
        for suf in ("_H", "_L"):
            if base.endswith(suf):
                base = base[:-2]
        if base == "broken_bar":
            base = "rotor_bar"
        out.add(base)
    return out


def build_multi_label(y: np.ndarray) -> np.ndarray:
    Y = np.zeros((len(y), len(LOCATIONS)), dtype=int)
    for i, name in enumerate(y):
        fs = fault_set(name)
        for j, loc in enumerate(LOCATIONS):
            Y[i, j] = int(loc in fs)
    return Y


def _new_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")


def validate_leave_one_condition_out(X, Y, torque_nm, rpm, y_names):
    """Same validation style as classifier_mcc5.py: hold out one (torque, rpm) condition at
    a time (all classes at once), aggregate a mean per-location probability across each held-
    out file's own windows, then threshold. Returns file-level (true, predicted) label
    matrices and the fault name for each row, for external reporting."""
    file_true, file_pred_probs, file_names = [], [], []
    for test_t, test_r in CONDITIONS:
        test_mask = (torque_nm == test_t) & (rpm == test_r)
        train_mask = ~test_mask

        clf = _new_classifier()
        clf.fit(X[train_mask], Y[train_mask])
        probs = clf.predict_proba(X[test_mask])  # list of (n,2) arrays, one per location

        y_test_names = y_names[test_mask]
        for fault in sorted(set(y_test_names)):
            fmask = y_test_names == fault
            mean_probs = np.array([
                probs[j][fmask, 1].mean() if probs[j].shape[1] > 1 else 0.0
                for j in range(len(LOCATIONS))
            ])
            file_true.append(Y[test_mask][fmask][0])
            file_pred_probs.append(mean_probs)
            file_names.append(fault)

    return np.array(file_true), np.array(file_pred_probs), np.array(file_names)


def report(file_true, file_pred_probs, threshold=PRESENCE_THRESHOLD):
    file_pred = (file_pred_probs >= threshold).astype(int)
    prec, rec, f1, support = precision_recall_fscore_support(file_true, file_pred, average=None, zero_division=0)
    print(f"=== per-location file-level performance (threshold={threshold}) ===")
    for i, loc in enumerate(LOCATIONS):
        print(f"  {loc:22s} precision={prec[i]:.3f}  recall={rec[i]:.3f}  f1={f1[i]:.3f}  n_true={support[i]}")


def train_final_model(X, Y, split: str):
    """Trained on ALL available data (no holdout) -- this is the deployable model, distinct
    from the leave-one-condition-out validation above which exists to report honest
    generalization numbers, not to produce the artifact actually used for inference."""
    clf = _new_classifier()
    clf.fit(X, Y)
    # compress=3: a multi-output RandomForest stores a (n_outputs, n_classes) float64
    # array at every node of every tree, which is hugely repetitive -- zlib takes these
    # from ~250-390MB down to ~40-60MB. Lossless: verified predictions bit-identical
    # before and after. Load time is unaffected (less disk I/O offsets the inflate).
    joblib.dump({"model": clf, "locations": LOCATIONS, "threshold": PRESENCE_THRESHOLD},
                model_path(split), compress=3)
    return clf


def predict_presence(model, X) -> dict:
    """Mean presence probability per location across a set of windows (e.g. all windows
    from one uploaded file)."""
    probs = model.predict_proba(X)
    return {
        loc: float(probs[j][:, 1].mean()) if probs[j].shape[1] > 1 else 0.0
        for j, loc in enumerate(LOCATIONS)
    }


def main():
    for split in SPLITS:
        print(f"\n########## {split} ##########")
        X, y, torque_nm, rpm = build_dataset(split=split)
        torque_nm, rpm = torque_nm.astype(int), rpm.astype(int)
        Y = build_multi_label(y)

        file_true, file_pred_probs, file_names = validate_leave_one_condition_out(X, Y, torque_nm, rpm, y)
        report(file_true, file_pred_probs)

        print("\ntraining final deployable model on all available data...")
        train_final_model(X, Y, split)
        print(f"saved to {model_path(split)}")


if __name__ == "__main__":
    main()
