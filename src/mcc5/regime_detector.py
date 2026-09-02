"""
Detect which control regime a recording belongs to, using the torque channel's own
coefficient of variation.

"""

SPEED_CIRCULATION_REFERENCE_CV = 0.1150   # constant-torque control 
TORQUE_CIRCULATION_REFERENCE_CV = 0.3107  # constant-speed control

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


def main():
    """Checked against every real condition, not just the original 20Nm/1000rpm slice
    See module docstring for why that broader check mattered.
    """
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

if __name__ == "__main__":
    main()
