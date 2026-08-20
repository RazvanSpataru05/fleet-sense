"""
Layer 2: fault localization via per-feature reconstruction error.

Layer 1 only answers "is this anomalous". This looks at *which* features the
Autoencoder reconstructs worst, grouped into physically meaningful families, to guess
*what kind* of fault it is -- reusing the same trained model, no new model needed.

Feature layout per phase block (33 cols x 3 phases = 99), then 1 cross-phase dim (99),
then 4 condition dims (100:104):
  [0:25]  wide-band envelope spectrum bins (generic -- no known formula for these faults
          yet: dynamic eccentricity, winding)
  [25:28] stats: rms, peak, kurtosis (also generic)
  [28:31] targeted: bpfo, bpfi, bsf (physics-grounded bearing fault frequencies)
  [31]    targeted: rotor bar twice-slip-frequency sideband
  [32]    targeted: static eccentricity 2x-rotation-frequency envelope peak
  [99]    targeted: rotation-invariant negative-sequence current magnitude (cross-phase,
          one value not per-phase -- see envelope_dataset_mcc5.py for why)

Bearing, rotor-bar, static-eccentricity, and negative-sequence (voltage unbalance) faults
have a targeted formula built, so location/type diagnosis is only attempted for those. A
first pass just took argmax(family ratios) and it was unreliable beyond bearing faults --
it can't tell a genuinely narrow, localized spike (real fault signature) apart from a
broad disturbance that elevates every family at once (voltage unbalance, winding) or from
plain noise (weak/no real signal). Fixed with two checks instead of one:
  1. is there a real signal at all (max ratio clears a minimum bar)?
  2. is that elevation concentrated in one family relative to the *generic* baseline, or
     just riding along with a broad disturbance that inflates everything (generic
     features included)?
Only when both hold do we attempt a fault-type call.
"""
import numpy as np
import joblib

from load_mcc5 import list_files
from envelope_dataset_mcc5 import CONDITION, artifacts_dir, build_matrix

SPLIT = "torque_circulation"  # the regime where real bearing-fault signal was confirmed

PHASE_BLOCK = 33
N_PHASES = 3

FAMILIES = {
    "wideband": [p * PHASE_BLOCK + i for p in range(N_PHASES) for i in range(0, 25)],
    "stats": [p * PHASE_BLOCK + i for p in range(N_PHASES) for i in range(25, 28)],
    "bpfo": [p * PHASE_BLOCK + 28 for p in range(N_PHASES)],
    "bpfi": [p * PHASE_BLOCK + 29 for p in range(N_PHASES)],
    "bsf": [p * PHASE_BLOCK + 30 for p in range(N_PHASES)],
    "rotor_bar": [p * PHASE_BLOCK + 31 for p in range(N_PHASES)],
    "eccentricity": [p * PHASE_BLOCK + 32 for p in range(N_PHASES)],
    "neg_seq": [N_PHASES * PHASE_BLOCK],  # single cross-phase value, not per-phase
}

BEARING_LOCATION = {"bpfo": "outer race", "bpfi": "inner race", "bsf": "ball"}
FAULT_TYPE_NAME = {
    **BEARING_LOCATION,
    "rotor_bar": "broken rotor bar",
    "eccentricity": "static eccentricity",
    "neg_seq": "voltage unbalance / winding imbalance",
}

TARGETED_FAMILIES = ["bpfo", "bpfi", "bsf", "rotor_bar", "eccentricity", "neg_seq"]
GENERIC_FAMILIES = ["wideband", "stats"]

MIN_SIGNAL_RATIO = 1.5    # below this, nothing is elevated enough to say anything at all

DEFAULT_FAMILY_VS_GENERIC_THRESHOLD = 2.2  # a targeted family must beat the *generic*
# baseline (not the other targeted families) by this much to earn a specific fault-type
# call. Calibrated against the validated fault set: clears bpfo=2.86 (bearing_outer_H,
# correct) and bpfi=2.56 (bearing_inner_H, correct) while rejecting bpfo=2.02
# (bearing_ball_H, where bsf=1.64 is the true family -- letting bpfo win here would be a
# new wrong call, not just an inherited one).

FAMILY_VS_GENERIC_THRESHOLD = {
    # eccentricity needs its own, higher bar: unlike bpfo/bpfi/bsf, its healthy floor isn't
    # unusually tiny, but almost *any* unrelated disturbance nudges it 2-3.7x (bend=3.41,
    # bearing_ball_H=3.67, bearing_outer_L=2.42) -- that's collateral jitter, not real
    # eccentricity. Real static eccentricity separates cleanly at high severity (9-11x) but
    # only reaches ~4.2x at low severity, uncomfortably close to that 3.67 noise ceiling --
    # rather than thread a fragile needle between one L-severity data point and the noise
    # floor, 6.0 keeps the H-severity calls confident and lets L-severity honestly fall to
    # "unsure", consistent with the severity-dependent detectability seen everywhere else
    # in this project.
    "eccentricity": 6.0,
}

GENERIC_AVG_CEILING = 9.0  # if the *generic* baseline itself is this elevated, the file is
# too broadly/severely disturbed to trust any single family's ratio as the true cause --
# this is what actually fixes voltage_unbalance_L getting misattributed to "eccentricity"
# (its eccentricity ratio is 46.78x generic, dwarfing real eccentricity's own 9-11x, simply
# because voltage unbalance is a catastrophic global disturbance, not because eccentricity
# is the true mechanism). Single faults all show generic_avg <= 7.21 (winding_H) except
# voltage_unbalance_L's 30+ -- a wide gap.
#
# NOTE this ceiling is NOT what's currently limiting real two-fault combo files (see
# diagnose_mixed_mcc5.py) -- most of them have a perfectly modest generic_avg (1-7.5) and
# clear this bar easily. Their real blocker is FAMILY_VS_GENERIC_THRESHOLD itself: compounding
# two real faults dilutes each individual family's excess-over-generic signal (e.g.
# broken_bar_and_bearing_inner_H's true bpfi sits at 2.00, dynamic_eccentricity_and_
# bearing_inner_H's true bpfi at 2.13 -- both just under the 2.2 bar), and that diluted
# range overlaps with genuine single-fault noise (bearing_ball_H's spurious bpfo=2.02,
# which 2.2 was specifically calibrated to exclude). There is currently no single threshold
# that admits the former without readmitting the latter -- a real, unresolved limitation,
# not a tuning gap. Only bearing_outer_H_and_inner_H is actually blocked by this ceiling
# (generic_avg=37.77, exceeding even voltage_unbalance_L's 30.22) -- and even without the
# ceiling, its own true bpfo ranks last among all six targeted families, so it wouldn't
# resolve correctly regardless.

EXPECTED_FAMILY = {
    "bearing_outer_H": "bpfo", "bearing_outer_L": "bpfo",
    "bearing_inner_H": "bpfi", "bearing_inner_L": "bpfi",
    "bearing_ball_H": "bsf", "bearing_ball_L": "bsf",
    "broken_bar": "rotor_bar",
    "static_eccentricity_H": "eccentricity", "static_eccentricity_L": "eccentricity",
    "voltage_unbalance_L": "neg_seq",
}
OTHER_FAULTS = ["bend", "dynamic_eccentricity", "winding_H"]


def per_feature_error(model, X) -> np.ndarray:
    X_hat = model.predict(X)
    return (X - X_hat) ** 2  # NOT averaged -- keep every feature's own error


def family_ratios(mean_sq_error: np.ndarray, healthy_family_means: dict) -> dict:
    return {
        name: mean_sq_error[idx].mean() / healthy_family_means[name]
        for name, idx in FAMILIES.items()
    }


def candidate_families(ratios: dict) -> list:
    """Every targeted family that independently clears its own bar against the *generic*
    baseline -- not just the single best one. diagnose() uses only the top of this list;
    a multi-fault file can have more than one real candidate, so this is exposed
    separately for mixed-fault validation (see diagnose_mixed_mcc5.py)."""
    generic_avg = sum(ratios[f] for f in GENERIC_FAMILIES) / len(GENERIC_FAMILIES)

    if generic_avg >= GENERIC_AVG_CEILING:
        # the whole file is too broadly disturbed to trust any specific family's ratio --
        # don't even look for a candidate, this is a catastrophic/global anomaly by itself.
        return []

    return sorted(
        (
            (family, ratios[family]) for family in TARGETED_FAMILIES
            if ratios[family] >= MIN_SIGNAL_RATIO
            and ratios[family] / generic_avg >= FAMILY_VS_GENERIC_THRESHOLD.get(family, DEFAULT_FAMILY_VS_GENERIC_THRESHOLD)
        ),
        key=lambda x: -x[1],
    )


def diagnose(ratios: dict) -> dict:
    """Each physically-grounded family is checked independently against the *generic*
    baseline (wideband/stats), not against every other family in one flat argmax. That
    argmax approach was tried first and broke on voltage_unbalance_L: it's a broad, severe
    disturbance that inflates every family at once (13x-42x), including bpfo/bsf, so it
    beat the physically-correct neg_seq family on raw ratio alone and got confidently
    misattributed as a bearing fault. Checking each targeted family against the generic
    level instead means a family only wins if it stands out *above the disturbance itself*
    -- for voltage_unbalance_L the generic baseline is so inflated (avg~30x) that no
    targeted family clears it, so it correctly falls through to broad_anomaly rather than
    picking whichever targeted family happened to be biggest."""
    generic_avg = sum(ratios[f] for f in GENERIC_FAMILIES) / len(GENERIC_FAMILIES)

    max_ratio = max(ratios.values())
    top_family = max(ratios, key=ratios.get)

    candidates = candidate_families(ratios)

    if not candidates:
        if max_ratio < MIN_SIGNAL_RATIO:
            return {"verdict": "no_signal", "top_family": top_family, "max_ratio": max_ratio, "dominance": None}
        return {"verdict": "broad_anomaly_not_bearing_specific", "top_family": top_family,
                "max_ratio": max_ratio, "dominance": max_ratio / generic_avg}

    top_family, top_ratio = max(candidates, key=lambda x: x[1])
    return {"verdict": FAULT_TYPE_NAME[top_family], "top_family": top_family,
            "max_ratio": top_ratio, "dominance": top_ratio / generic_avg}


def evaluate_fault(fault: str, scaler, model, healthy_family_means: dict) -> dict:
    files = list_files(fault=fault, split=SPLIT, **CONDITION)
    rows = np.array(build_matrix(files))
    fault_err = per_feature_error(model, scaler.transform(rows)).mean(axis=0)
    ratios = family_ratios(fault_err, healthy_family_means)
    return diagnose(ratios), ratios


def main():
    out_dir = artifacts_dir(SPLIT)
    scaler = joblib.load(out_dir / "scaler_envelope.pkl")
    model = joblib.load(out_dir / "autoencoder_mcc5.pkl")

    X_val = np.load(out_dir / "X_val_envelope.npy")
    healthy_err = per_feature_error(model, X_val).mean(axis=0)
    healthy_family_means = {name: healthy_err[idx].mean() for name, idx in FAMILIES.items()}

    print("=== Faults with a targeted formula (bearing location + rotor bar) ===")
    print(f"{'fault':20s} {'expected':>18s} {'verdict':>28s} {'max_ratio':>10s} {'dominance':>10s}")
    n_correct = n_ambiguous = n_wrong = 0
    for fault, expected_family in EXPECTED_FAMILY.items():
        result, ratios = evaluate_fault(fault, scaler, model, healthy_family_means)
        expected_verdict = FAULT_TYPE_NAME[expected_family]
        dom_str = f"{result['dominance']:.2f}" if result["dominance"] is not None else "n/a"
        if result["verdict"] == expected_verdict:
            tag, n_correct = "CORRECT", n_correct + 1
        elif result["verdict"] in ("no_signal", "broad_anomaly_not_bearing_specific"):
            tag, n_ambiguous = "HONEST-UNSURE", n_ambiguous + 1
        else:
            tag, n_wrong = "WRONG", n_wrong + 1
        print(f"{fault:20s} {expected_verdict:>18s} {result['verdict']:>28s} {result['max_ratio']:10.2f} {dom_str:>10s}  [{tag}]")
    print(f"\ncorrect={n_correct}  honestly-unsure={n_ambiguous}  confidently-wrong={n_wrong}  (out of {len(EXPECTED_FAMILY)})")

    print("\n=== Faults with NO targeted formula (should not get a confident fault-type call) ===")
    print(f"{'fault':25s} {'verdict':>28s} {'max_ratio':>10s} {'dominance':>10s}")
    n_correctly_unattributed = n_falsely_attributed = 0
    known_verdicts = set(FAULT_TYPE_NAME.values())
    for fault in OTHER_FAULTS:
        result, ratios = evaluate_fault(fault, scaler, model, healthy_family_means)
        dom_str = f"{result['dominance']:.2f}" if result["dominance"] is not None else "n/a"
        falsely_attributed = result["verdict"] in known_verdicts
        n_falsely_attributed += falsely_attributed
        n_correctly_unattributed += not falsely_attributed
        tag = "FALSE ATTRIBUTION" if falsely_attributed else "correctly unattributed"
        print(f"{fault:25s} {result['verdict']:>28s} {result['max_ratio']:10.2f} {dom_str:>10s}  [{tag}]")
    print(f"\ncorrectly unattributed={n_correctly_unattributed}  falsely attributed={n_falsely_attributed}  (out of {len(OTHER_FAULTS)})")


if __name__ == "__main__":
    main()
