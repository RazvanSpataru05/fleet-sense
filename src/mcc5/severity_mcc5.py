"""
Layer 2, stage 2: bearing fault severity (high vs low), for the three bearing locations
where the dataset actually has both severity levels recorded (bearing_outer, bearing_inner,
bearing_ball). Deliberately NOT attempted for the other 6 locations:

  - bend, rotor_bar (broken_bar), dynamic_eccentricity, voltage_unbalance: the dataset only
    ever recorded ONE severity level for these. There's no H/L pair to learn a distinction
    from -- a data limitation, not something more modeling effort can fix.
  - static_eccentricity, winding: DO have H/L labels, but a dedicated severity classifier
    (same approach as below) tested at 50% and 58% leave-one-condition-out accuracy --
    chance level. Reusing the presence-detection probability as a severity proxy tested even
    worse (confirmed empirically: it doesn't consistently track true severity direction at
    all for these two, sometimes scoring L higher than H). No severity call is made for
    these; they surface as presence-only.

Each bearing location gets its own binary classifier (H vs L), trained only on that
location's own PURE single-fault files (bearing_outer_H/L, etc.) -- not the 4 combo files
each of bearing_outer/bearing_inner also appears in (e.g. "winding_H_and_bearing_outer_H"),
since a combo's severity signature for that location hasn't been tested and might differ
from a pure single fault's. Those combo files could plausibly be added later as extra
"H" training examples (they're always the H variant in this dataset) to shore up bearing_
outer/bearing_inner's small sample size -- noted here as a real option, not implemented,
since it's untested and bearing_ball has no combo files to match it with anyway.

Validated leave-one-condition-out (n=12 files per location: 6 conditions x 2 severities):
  bearing_ball    83% file-level accuracy
  bearing_inner   83%
  bearing_outer   75%
A real, consistent pattern across all three bearing locations -- not one lucky number --
but still a small sample, so the exact percentage carries real uncertainty.

Provides diagnose_file(), the full two-stage inference entrypoint meant to be reusable
directly by a future web app: given an uploaded recording, run presence detection first,
then severity for any bearing location that comes back present, and return a plain list of
detected issues.
"""
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

from envelope_dataset_mcc5 import windows_for_file, artifacts_dir
from classifier_mcc5 import build_dataset, SPLIT, CONDITIONS
from presence_mcc5 import PRESENCE_THRESHOLD, MODEL_PATH as PRESENCE_MODEL_PATH, predict_presence

BEARING_SEVERITY_LOCATIONS = {
    "bearing_outer": ("bearing_outer_H", "bearing_outer_L"),
    "bearing_inner": ("bearing_inner_H", "bearing_inner_L"),
    "bearing_ball": ("bearing_ball_H", "bearing_ball_L"),
}

SEVERITY_MODEL_PATH = artifacts_dir(SPLIT) / "severity_models.pkl"


def _new_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


def _location_data(X, y, torque_nm, rpm, h_name, l_name):
    mask = (y == h_name) | (y == l_name)
    y_bin = (y[mask] == h_name).astype(int)  # 1 = high severity, 0 = low
    return X[mask], y_bin, torque_nm[mask], rpm[mask]


def validate_severity(X, y, torque_nm, rpm):
    """Leave-one-condition-out per bearing location, matching classifier_mcc5.py's and
    presence_mcc5.py's validation style. Returns {location: file_level_accuracy}."""
    results = {}
    for location, (h_name, l_name) in BEARING_SEVERITY_LOCATIONS.items():
        Xs, ys, ts, rs = _location_data(X, y, torque_nm, rpm, h_name, l_name)
        file_true, file_pred = [], []
        for test_t, test_r in CONDITIONS:
            test_mask = (ts == test_t) & (rs == test_r)
            train_mask = ~test_mask
            clf = _new_classifier()
            clf.fit(Xs[train_mask], ys[train_mask])
            probs = clf.predict_proba(Xs[test_mask])[:, 1]
            for label in (0, 1):
                lm = ys[test_mask] == label
                if lm.sum() == 0:
                    continue
                file_true.append(label)
                file_pred.append(int(probs[lm].mean() >= 0.5))
        file_true, file_pred = np.array(file_true), np.array(file_pred)
        acc = (file_true == file_pred).mean()
        results[location] = (acc, len(file_true))
    return results


def train_final_models(X, y, torque_nm, rpm):
    """Trained on ALL available pure H/L files per location (no holdout) -- these are the
    deployable models; validate_severity() above exists only to report honest generalization
    numbers separately."""
    models = {}
    for location, (h_name, l_name) in BEARING_SEVERITY_LOCATIONS.items():
        Xs, ys, _, _ = _location_data(X, y, torque_nm, rpm, h_name, l_name)
        clf = _new_classifier()
        clf.fit(Xs, ys)
        models[location] = clf
    joblib.dump(models, SEVERITY_MODEL_PATH)
    return models


def diagnose_file(csv_path, presence_model=None, severity_models=None):
    """Full inference entrypoint: extract features from one uploaded recording, run
    presence detection across all 9 locations, then severity for any bearing location
    that comes back present. Returns a plain list of detected-issue dicts, ready to hand
    to a UI layer -- no plotting or coloring decisions made here."""
    if presence_model is None:
        presence_model = joblib.load(PRESENCE_MODEL_PATH)["model"]
    if severity_models is None:
        severity_models = joblib.load(SEVERITY_MODEL_PATH)

    rows = windows_for_file(csv_path)
    X = np.array(rows)

    presence = predict_presence(presence_model, X)

    issues = []
    for location, confidence in sorted(presence.items(), key=lambda kv: -kv[1]):
        if confidence < PRESENCE_THRESHOLD:
            continue
        issue = {"location": location, "presence_confidence": round(confidence, 3)}
        if location in severity_models:
            p_high = float(severity_models[location].predict_proba(X)[:, 1].mean())
            issue["severity"] = "high" if p_high >= 0.5 else "low"
            issue["severity_confidence"] = round(p_high if p_high >= 0.5 else 1 - p_high, 3)
        else:
            issue["severity"] = "not assessable (no severity model for this location)"
        issues.append(issue)

    return issues if issues else [{"location": "none", "presence_confidence": None, "severity": "healthy"}]


def main():
    X, y, torque_nm, rpm = build_dataset()
    torque_nm, rpm = torque_nm.astype(int), rpm.astype(int)

    print("=== severity validation (leave-one-condition-out) ===")
    for location, (acc, n) in validate_severity(X, y, torque_nm, rpm).items():
        print(f"  {location:15s} accuracy={acc:.3f}  (n={n})")

    print("\ntraining final deployable severity models on all available data...")
    train_final_models(X, y, torque_nm, rpm)
    print(f"saved to {SEVERITY_MODEL_PATH}")


if __name__ == "__main__":
    main()
