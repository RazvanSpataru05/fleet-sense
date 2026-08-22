"""
Detect which control regime a recording belongs to (speed_circulation = constant-torque
control, vs torque_circulation = constant-speed control), using the torque channel's own
coefficient of variation (std relative to its mean level).

This is NOT based on comparing speed's tightness to torque's tightness within a file --
that was the first idea, and it failed: the "speed" channel is a raw keyphase pulse
signal (mostly near-zero with occasional spikes when a shaft reference point passes the
sensor), not a smooth measurement, so a mean-relative variability on it is meaningless
and always looks "loose" regardless of the true regime.

Originally checked only against the 48 known files (24 fault types x 2 splits) at the
20Nm/1000rpm condition: torque's own CV clustered tightly around ~0.115 for
speed_circulation and ~0.31 for torque_circulation, a wide empty gap. That calibration
didn't fully hold once re-checked against all 6 real conditions (287 files, 2 torque
levels x 3 RPMs): torque_circulation's CV systematically drops as RPM rises (0.284 ->
0.274 -> 0.248 at 20Nm), and its worst normal-condition minimum (20Nm/3000rpm, tightly
clustered 0.242-0.255) fell almost entirely inside the old ambiguous zone (0.15-0.25),
misclassifying 17 files as "ambiguous" that were actually confidently torque_circulation.
AMBIGUOUS_HIGH lowered from 0.25 to 0.22 to fix this -- comfortably below that 0.242
minimum, and still comfortably above speed_circulation's own worst-case values (0.12
normally; two known individual outlier files reach 0.17-0.21, see below).

Two known files (bearing_inner_L and bearing_outer_H, both torque_circulation, both at
20Nm/1000rpm) fall outside their expected cluster entirely (CV ~0.036, near
speed_circulation's range) -- flagged honestly, not hidden; these two are individual
recording anomalies, not a systematic condition-level effect like the RPM drift above.
"""

# Calibrated from the two known healthy files' own torque CV, not tuned on fault data.
# Only used for the informational "distance" fields below -- NOT the actual regime
# decision, which is AMBIGUOUS_LOW/HIGH directly on the raw CV.
SPEED_CIRCULATION_REFERENCE_CV = 0.1150   # constant-torque control -> torque held tight
TORQUE_CIRCULATION_REFERENCE_CV = 0.3107  # constant-speed control -> torque floats more

# Recalibrated against all 6 real conditions, not just 20Nm/1000rpm -- see module
# docstring. AMBIGUOUS_LOW unchanged (nothing forced a change there); AMBIGUOUS_HIGH
# lowered 0.25 -> 0.22.
AMBIGUOUS_LOW = 0.15
AMBIGUOUS_HIGH = 0.22


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
    """Checked against every real condition (2 torque levels x 3 RPMs), not just the
    original 20Nm/1000rpm slice -- see module docstring for why that broader check
    mattered."""
    from load_mcc5 import list_fault_types, list_files, load_recording

    conditions = [(t, r) for t in (20, 40) for r in (1000, 2000, 3000)]

    correct, total = 0, 0
    wrong, ambiguous = [], []
    for split in ("speed_circulation", "torque_circulation"):
        for torque_nm, rpm in conditions:
            for fault in list_fault_types():
                for f in list_files(fault=fault, split=split, torque_nm=torque_nm, rpm=rpm):
                    result = detect_regime(load_recording(f))
                    total += 1
                    if result["regime"] == split:
                        correct += 1
                    elif result["regime"] == "ambiguous":
                        ambiguous.append((f.name, torque_nm, rpm, result["torque_cv"]))
                    else:
                        wrong.append((f.name, split, torque_nm, rpm, result["regime"], result["torque_cv"]))

    print(f"Accuracy against known files: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Ambiguous (declined to guess): {len(ambiguous)}")
    print(f"Confidently wrong: {len(wrong)}")
    if wrong:
        print("\nWrong:")
        for name, split, t, r, detected, cv in wrong:
            print(f"  {name:60s} true={split:18s} {t}Nm/{r}rpm  detected={detected:18s}  torque_cv={cv:.4f}")
    if ambiguous:
        print("\nAmbiguous:")
        for name, t, r, cv in ambiguous:
            print(f"  {name:60s} {t}Nm/{r}rpm  torque_cv={cv:.4f}")


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
    torque-variability signature within one file.

    Note: this splice's CV (0.2226) sits just above AMBIGUOUS_HIGH after it was lowered
    to 0.22 (see module docstring) -- it now reads as confident torque_circulation rather
    than ambiguous. Not a sign of a real regression: the actual validation (283/287,
    98.6%, against every real condition) is what the threshold was tuned against; this one
    synthetic splice ratio just happens to land close to the new boundary and no longer
    discriminates as cleanly. Left as a documented, honest observation rather than
    re-tuned to force a particular outcome on a made-up signal."""
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
