"""
Layer 2 as a real supervised classifier, trained on labeled fault data across all 6 real
recording conditions per fault type (2 torque levels x 3 RPMs) -- not just the single
condition (20Nm/1000rpm) the rest of this pipeline started scoped to.

Why: diagnose_mcc5.py's reconstruction-error-ratio heuristic was hand-tuned against
exactly one file per fault type -- every threshold was fragile because there was nothing
to average over (confirmed directly: mixed-fault validation showed real combo signal and
single-fault noise landing in the same narrow band, with no single threshold able to
separate them). This dataset actually has 6 independent real recordings per fault type per
split; the rest of this project used only 1 of them. Expanding to all 6 needed one
prerequisite fix: shaft_hz_for_file's fundamental-frequency search was hardcoded for
1000rpm and gave nonsense at 2000/3000rpm -- now RPM-adaptive (see envelope_dataset_mcc5.py
-- confirmed detection lands within 0.4% of nominal at all 3 RPMs after the fix).

Torque/RPM are kept as explicit input features (already the tail of the window feature
vector) so the model can learn condition-specific patterns instead of being confounded by
them, rather than trying to force condition-invariant features.

Validated with leave-one-condition-out: each of the 6 (torque, rpm) conditions is held out
in turn as the test set (for every class at once), trained on the other 5. This guarantees
no window from a held-out file's recording ever appears in training, and tests
generalization to a genuinely unseen operating condition -- not a random split of
overlapping windows from the same file, which would leak.

Reports both window-level accuracy (what the classifier gets right per 0.5s slice) and
file-level accuracy (majority vote across a whole recording's windows) -- the latter is
what actually matters for the real use case: diagnosing an uploaded recording, not a
single isolated window.
"""
import numpy as np
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from load_mcc5 import list_files, list_fault_types
from envelope_dataset_mcc5 import windows_for_file, artifacts_dir

SPLIT = "torque_circulation"
CONDITIONS = [(t, r) for t in (20, 40) for r in (1000, 2000, 3000)]


def cache_file(split: str):
    cache_dir = artifacts_dir(split) / "classifier_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "multi_condition_dataset.npz"


# kept for backwards compatibility with existing callers that import these directly
CACHE_DIR = artifacts_dir(SPLIT) / "classifier_cache"
CACHE_FILE = cache_file(SPLIT)


def build_dataset(split=SPLIT, force=False):
    cf = cache_file(split)
    if cf.exists() and not force:
        data = np.load(cf, allow_pickle=True)
        return data["X"], data["y"], data["torque_nm"], data["rpm"]

    X_list, y_list, torque_list, rpm_list = [], [], [], []
    fault_types = list_fault_types()
    for fault in fault_types:
        for torque_nm, rpm in CONDITIONS:
            files = list_files(fault=fault, split=split, torque_nm=torque_nm, rpm=rpm)
            if not files:
                print(f"WARNING: missing {fault} at {torque_nm}Nm/{rpm}rpm")
                continue
            rows = windows_for_file(files[0])
            X_list.extend(rows)
            y_list.extend([fault] * len(rows))
            torque_list.extend([torque_nm] * len(rows))
            rpm_list.extend([rpm] * len(rows))
        print(f"extracted: {fault}")

    X = np.array(X_list)
    y = np.array(y_list)
    torque_nm = np.array(torque_list)
    rpm = np.array(rpm_list)
    np.savez(cf, X=X, y=y, torque_nm=torque_nm, rpm=rpm)
    return X, y, torque_nm, rpm


def leave_one_condition_out(X, y, torque_nm, rpm):
    """Each fold holds out ALL windows from one (torque, rpm) condition -- across every
    fault class at once -- as test, training on the other 5 conditions' windows. No
    window from a held-out file ever appears in training."""
    window_true, window_pred = [], []
    file_true, file_pred = [], []  # one entry per (fault, held-out condition) file

    for test_torque, test_rpm in CONDITIONS:
        test_mask = (torque_nm == test_torque) & (rpm == test_rpm)
        train_mask = ~test_mask

        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
        clf.fit(X[train_mask], y[train_mask])
        pred = clf.predict(X[test_mask])

        window_true.extend(y[test_mask])
        window_pred.extend(pred)

        window_acc = (pred == y[test_mask]).mean()

        # file-level majority vote: one file per fault type at this held-out condition
        y_test = y[test_mask]
        file_hits = 0
        file_total = 0
        for fault in sorted(set(y_test)):
            fault_mask = y_test == fault
            votes = Counter(pred[fault_mask])
            majority = votes.most_common(1)[0][0]
            file_true.append(fault)
            file_pred.append(majority)
            file_total += 1
            file_hits += int(majority == fault)

        print(f"held out {test_torque}Nm/{test_rpm}rpm: window_acc={window_acc:.3f}  "
              f"file_acc={file_hits}/{file_total}={file_hits/file_total:.3f}")

    return (np.array(window_true), np.array(window_pred)), (np.array(file_true), np.array(file_pred))


def main():
    X, y, torque_nm, rpm = build_dataset()
    print(f"\ntotal windows: {len(y)}, classes: {len(set(y))}\n")

    (w_true, w_pred), (f_true, f_pred) = leave_one_condition_out(X, y, torque_nm, rpm)

    print(f"\n=== overall window-level accuracy: {(w_true == w_pred).mean():.3f} ===")
    print(f"=== overall file-level (majority vote) accuracy: {(f_true == f_pred).mean():.3f} ===\n")

    print("=== per-class report (file-level, majority vote across all 6 held-out conditions) ===")
    print(classification_report(f_true, f_pred, zero_division=0))

    out_dir = artifacts_dir(SPLIT)
    joblib.dump({"window_true": w_true, "window_pred": w_pred, "file_true": f_true, "file_pred": f_pred},
                out_dir / "classifier_cache" / "loco_results.pkl")


if __name__ == "__main__":
    main()
