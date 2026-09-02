"""
Layer 2, stage 2: bearing fault severity, for the three bearing locations
where the dataset actually has both severity levels recorded.
Not attempted for the other 6 locations.

Each bearing location gets its own binary classifier, trained only on that
location's own PURE single-fault files, not the 4 combo files each of bearing_outer/bearing_inner
also appears in, since a combo's severity signature for that location hasn't been tested and might differ
from a pure single fault's.

Validated leave-one-condition-out:
  bearing_ball    83% file-level accuracy
  bearing_inner   83%
  bearing_outer   75%

A real, consistent pattern across all three bearing locations: not one lucky number
but still a small sample, so the exact percentage carries real uncertainty.
"""
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

from envelope_dataset_mcc5 import artifacts_dir
from classifier_mcc5 import build_dataset, CONDITIONS
from presence_mcc5 import SPLITS, PRESENCE_THRESHOLD, model_path as presence_model_path, predict_presence

BEARING_SEVERITY_LOCATIONS = {
    "bearing_outer": ("bearing_outer_H", "bearing_outer_L"),
    "bearing_inner": ("bearing_inner_H", "bearing_inner_L"),
    "bearing_ball": ("bearing_ball_H", "bearing_ball_L"),
}


def severity_model_path(split: str):
    return artifacts_dir(split) / "severity_models.pkl"


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


def train_final_models(X, y, torque_nm, rpm, split: str):
    """Trained on all available pure H/L files per location. These are the
    deployable models; validate_severity() above exists only to report honest generalization
    numbers separately."""
    models = {}
    for location, (h_name, l_name) in BEARING_SEVERITY_LOCATIONS.items():
        Xs, ys, _, _ = _location_data(X, y, torque_nm, rpm, h_name, l_name)
        clf = _new_classifier()
        clf.fit(Xs, ys)
        models[location] = clf
    joblib.dump(models, severity_model_path(split), compress=3) 
    return models


def diagnose_features(X, split: str, presence_model=None, severity_models=None) -> list:
    """Core Layer 2 logic given an already-extracted feature matrix, so an orchestrator
    that also needs X for Layer 1 (see pipeline_mcc5.py) does not extract features twice
    for the same file. Returns a plain list of detected-issue dicts, empty if nothing
    crosses the presence threshold -- no "healthy" sentinel, since that framing depends on
    what Layer 1 says too."""
    if presence_model is None:
        presence_model = joblib.load(presence_model_path(split))["model"]
    if severity_models is None:
        severity_models = joblib.load(severity_model_path(split))

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

    return issues

def main():
    for split in SPLITS:
        print(f"\n########## {split} ##########")
        X, y, torque_nm, rpm = build_dataset(split=split)
        torque_nm, rpm = torque_nm.astype(int), rpm.astype(int)

        print("=== severity validation (leave-one-condition-out) ===")
        for location, (acc, n) in validate_severity(X, y, torque_nm, rpm).items():
            print(f"  {location:15s} accuracy={acc:.3f}  (n={n})")

        print("\ntraining final deployable severity models on all available data...")
        train_final_models(X, y, torque_nm, rpm, split)
        print(f"saved to {severity_model_path(split)}")


if __name__ == "__main__":
    main()
