"""
Detect which control regime a recording belongs to (speed_circulation = constant-torque
control, vs torque_circulation = constant-speed control), using the torque channel's own
coefficient of variation (std relative to its mean level).

This is NOT based on comparing speed's tightness to torque's tightness within a file --
that was the first idea, and it failed: the "speed" channel is a raw keyphase pulse
signal (mostly near-zero with occasional spikes when a shaft reference point passes the
sensor), not a smooth measurement, so a mean-relative variability on it is meaningless
and always looks "loose" regardless of the true regime.

What actually works, checked directly against all 48 known files (24 fault types x 2
splits) at the 20Nm/1000rpm condition: torque's own CV clusters tightly around ~0.115
for speed_circulation and ~0.31 for torque_circulation, with a wide empty gap between
them. Two known files (bearing_inner_L and bearing_outer_H, both torque_circulation)
fall outside their expected cluster -- flagged honestly below, not hidden.
"""
import numpy as np

# Calibrated from the two known healthy files' own torque CV, not tuned on fault data.
SPEED_CIRCULATION_REFERENCE_CV = 0.1150   # constant-torque control -> torque held tight
TORQUE_CIRCULATION_REFERENCE_CV = 0.3107  # constant-speed control -> torque floats more

# The gap between the two clusters runs roughly 0.13-0.28 in what we've seen; anything
# landing in there -- or notably outside either cluster's own spread -- doesn't clearly
# match either known regime.
AMBIGUOUS_LOW = 0.15
AMBIGUOUS_HIGH = 0.25


def torque_cv(df) -> float:
    return float(df["torque"].std() / abs(df["torque"].mean()))


def detect_regime(df) -> dict:
    cv = torque_cv(df)

    if AMBIGUOUS_LOW <= cv <= AMBIGUOUS_HIGH:
        regime = "ambiguous"
    elif cv < AMBIGUOUS_LOW:
        regime = "speed_circulation"
    else:
        regime = "torque_circulation"

    dist_speed = abs(cv - SPEED_CIRCULATION_REFERENCE_CV)
    dist_torque = abs(cv - TORQUE_CIRCULATION_REFERENCE_CV)

    return {
        "torque_cv": cv,
        "regime": regime,
        "confident": regime != "ambiguous",
        "distance_to_speed_circulation": dist_speed,
        "distance_to_torque_circulation": dist_torque,
    }


def _validate_against_known_files():
    from load_mcc5 import list_fault_types, list_files, load_recording

    condition = {"torque_nm": 20, "rpm": 1000}
    expected_regime = {"speed_circulation": "speed_circulation", "torque_circulation": "torque_circulation"}

    correct, total, misclassified = 0, 0, []
    for fault in list_fault_types():
        for f in list_files(fault=fault, **condition):
            true_split = "speed_circulation" if "speed_circulation" in f.name else "torque_circulation"
            result = detect_regime(load_recording(f))
            total += 1
            if result["regime"] == expected_regime[true_split]:
                correct += 1
            else:
                misclassified.append((fault, true_split, result))

    print(f"Accuracy against known files: {correct}/{total} ({100*correct/total:.1f}%)")
    if misclassified:
        print("\nMisclassified:")
        for fault, true_split, result in misclassified:
            print(f"  {fault:40s} true={true_split:18s} detected={result['regime']:18s} torque_cv={result['torque_cv']:.4f}")


def _splice_test():
    """Synthetic stress test only -- not a claim about real transitional physics.

    First attempt (kept here as a documented dead end): splicing half of a 1000rpm
    speed_circulation recording to half of a 2000rpm speed_circulation recording did
    NOT trigger ambiguity (torque_cv=0.0938, confidently classified) -- because both
    halves are still constant-torque-controlled, so torque stays well-behaved in both
    halves regardless of the RPM change. That tested "different speed, same regime",
    not a regime conflict.

    Real test: splice across the two *regimes* -- half constant-torque-controlled,
    half constant-speed-controlled -- which should actually produce an inconsistent
    torque-variability signature within one file."""
    import pandas as pd
    from load_mcc5 import list_files, load_recording

    f_speed = list_files(fault="health", split="speed_circulation", torque_nm=20, rpm=1000)[0]
    f_torque = list_files(fault="health", split="torque_circulation", torque_nm=20, rpm=1000)[0]
    df_speed = load_recording(f_speed)
    df_torque = load_recording(f_torque)

    half = min(len(df_speed), len(df_torque)) // 2
    spliced = pd.concat([df_speed.iloc[:half], df_torque.iloc[half:2 * half]], ignore_index=True)

    result = detect_regime(spliced)
    print(f"\nSplice test (half speed_circulation + half torque_circulation, same 20Nm/1000rpm label):")
    print(f"  torque_cv={result['torque_cv']:.4f}  regime={result['regime']}  confident={result['confident']}")


if __name__ == "__main__":
    _validate_against_known_files()
    _splice_test()
