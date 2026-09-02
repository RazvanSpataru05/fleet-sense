"""
The real Layer 1 to Layer 2 entrypoint: check_motor is what a real deployment
would actually call for one uploaded recording. Chains:

  1. Regime detection: Which control regime this file is running under, which selects
     which pair of models to use. Refuses to proceed if ambiguous rather than guessing.

  2. Layer 1 anomaly scoring: is this file's overall signature elevated relative to a
     healthy baseline at its own auto-detected operating condition.

  3. Layer 2 diagnosis: Which specific location(s), and severity were assessable.

Design decision on how Layer 1 and Layer 2 combine: Layer 2 is NOT locked behind a
confident Layer 1 anomaly call. Layer 1 only catches ~72-73% of real faults a hard gate
would silently downgrade the other ~27-28% to "no issue detected", discarding
exactly the cases where Layer 2's more targeted per-location classifiers might still catch
something Layer 1's single aggregate reconstruction-error score misses. Both layers always
run instead, and the report combines both signals rather than dropping one:

  - Layer 2 finds >=1 location: report it/them, annotated with whether Layer 1
    independently agrees this file is anomalous overall;
  - Layer 2 finds nothing, but Layer 1 flags anomaly: "anomaly detected, no specific
    cause identified". The honest state for known blind-spot faults or a genuinely novel
    fault Layer 2 was never trained to recognize;
  - Neither flags anything: "no issue detected"

Runs upload_validation_mcc5.validate_upload() first. The model is never handed a file
that hasn't passed those checks. A rejected file returns verdict="rejected"
with the specific reason, not an exception the caller has to know to catch.
"""
import joblib
import numpy as np

from regime_detector import detect_regime
from envelope_dataset_mcc5 import (windows_for_file_blind, detect_condition_from_data,
                                   artifacts_dir, display_spectrum)
from anomaly_mcc5 import reconstruction_error, MIN_SIGNAL_RATIO
from severity_mcc5 import diagnose_features
from upload_validation_mcc5 import validate_upload, UploadValidationError


def check_motor(csv_path, declared_sample_rate: float = None) -> dict:
    try:
        validation = validate_upload(csv_path, declared_sample_rate=declared_sample_rate)
    except UploadValidationError as e:
        return {"verdict": "rejected", "reason": str(e)}
    df = validation["df"]

    regime = detect_regime(df)
    if not regime["confident"]:
        return {
            "verdict": "cannot_process",
            "reason": f"control regime could not be confidently determined "
                      f"(torque_cv={regime['torque_cv']:.3f}, ambiguous between "
                      f"speed_circulation and torque_circulation)",
            "regime_detection": regime,
        }
    split = regime["regime"]

    condition = detect_condition_from_data(df)
    rows = windows_for_file_blind(csv_path, df=df)
    X = np.array(rows)

    out_dir = artifacts_dir(split)
    anomaly_model = joblib.load(out_dir / "anomaly_model_multi_condition.pkl")
    anomaly_scaler = joblib.load(out_dir / "anomaly_scaler_multi_condition.pkl")
    baselines = joblib.load(out_dir / "anomaly_healthy_baselines.pkl")

    baseline_key = (condition["torque_nm"], condition["rpm"])
    healthy_baseline = baselines[baseline_key]
    file_error = reconstruction_error(anomaly_model, anomaly_scaler.transform(X)).mean()
    anomaly_ratio = float(file_error / healthy_baseline)
    layer1_anomalous = anomaly_ratio >= MIN_SIGNAL_RATIO

    issues = diagnose_features(X, split)

    if validation["reduced_confidence_rotor_bar"]:
        for issue in issues:
            if issue["location"] == "rotor_bar":
                issue["caveat"] = (
                    f"recording is {validation['duration_sec']:.0f}s long -- rotor-bar "
                    f"detection specifically needs close to the full ~90s reference length "
                    f"to be reliable (confirmed by truncation testing), so this result "
                    f"carries less confidence than the other locations"
                )

    if issues:
        verdict = "issues_detected"
    elif layer1_anomalous:
        verdict = "anomaly_detected_unattributed"
    else:
        verdict = "no_issue_detected"

    return {
        "verdict": verdict,
        "regime": split,
        "regime_confident": True,
        "condition_detected": condition,
        "duration_sec": round(validation["duration_sec"], 1),
        "sample_rate": round(validation["sample_rate"], 1),
        "reduced_confidence_rotor_bar": validation["reduced_confidence_rotor_bar"],
        "layer1_anomaly_ratio": round(anomaly_ratio, 3),
        "layer1_anomalous": layer1_anomalous,
        "issues": issues,
        "spectrum": display_spectrum(df, condition["rpm"]),
    }


if __name__ == "__main__":
    import sys
    result = check_motor(sys.argv[1])
    for k, v in result.items():
        print(f"{k}: {v}")
