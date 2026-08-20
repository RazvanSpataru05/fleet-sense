"""
Mixed-fault validation: does the Layer 2 diagnosis still make sense when TWO real faults
are present in the same recording at once? Uses the dataset's own combo files (e.g.
"bearing_outer_H_and_inner_H") -- real recordings with two induced faults simultaneously,
not synthetic/fabricated data.

diagnose_mcc5.py's diagnose() only reports the single best family, which isn't the right
question for a combo file -- a two-fault file can legitimately have two real candidates at
once. This instead uses candidate_families() (every family that independently clears its
own bar), and checks it against each combo's true components.

Not every component of a combo has a working detector yet: dynamic eccentricity has no
formula (no rotor-slot-count data available) and winding shows no reliable reconstruction-
error signal (confirmed in the single-fault validation, despite a real raw negative-
sequence elevation for voltage_unbalance specifically -- winding itself showed none). Those
components are marked "not currently detectable" and are not counted as misses -- only the
components we already have a validated formula for are scored.
"""
import numpy as np
import joblib

from load_mcc5 import list_files
from envelope_dataset_mcc5 import CONDITION, artifacts_dir, build_matrix
from diagnose_mcc5 import (
    SPLIT, FAMILIES, FAULT_TYPE_NAME, per_feature_error, family_ratios, candidate_families,
)

# For each combo fault: which families we'd expect to see elevated, and whether we
# currently have a validated detector for that component at all. Components with no
# detector aren't scored -- they're listed for honesty, not treated as a miss.
COMBO_FAULTS = {
    "bearing_outer_H_and_inner_H": {"bpfo": True, "bpfi": True},
    "broken_bar_and_bearing_inner_H": {"rotor_bar": True, "bpfi": True},
    "broken_bar_and_bearing_outer_H": {"rotor_bar": True, "bpfo": True},
    "dynamic_eccentricity_and_bearing_inner_H": {"dynamic_eccentricity": False, "bpfi": True},
    "dynamic_eccentricity_and_bearing_outer_H": {"dynamic_eccentricity": False, "bpfo": True},
    "static_eccentricity_H_and_bearing_inner_H": {"eccentricity": True, "bpfi": True},
    "static_eccentricity_H_and_bearing_outer_H": {"eccentricity": True, "bpfo": True},
    "winding_H_and_bearing_inner_H": {"winding": False, "bpfi": True},
    "winding_H_and_bearing_outer_H": {"winding": False, "bpfo": True},
}


def evaluate_combo(fault: str, scaler, model, healthy_family_means: dict) -> dict:
    files = list_files(fault=fault, split=SPLIT, **CONDITION)
    rows = np.array(build_matrix(files))
    err = per_feature_error(model, scaler.transform(rows)).mean(axis=0)
    ratios = family_ratios(err, healthy_family_means)
    candidates = candidate_families(ratios)
    return candidates, ratios


def main():
    out_dir = artifacts_dir(SPLIT)
    scaler = joblib.load(out_dir / "scaler_envelope.pkl")
    model = joblib.load(out_dir / "autoencoder_mcc5.pkl")

    X_val = np.load(out_dir / "X_val_envelope.npy")
    healthy_err = per_feature_error(model, X_val).mean(axis=0)
    healthy_family_means = {name: healthy_err[idx].mean() for name, idx in FAMILIES.items()}

    n_files = 0
    n_detectable_components = n_components_hit = 0
    n_false_positive_components = 0

    for fault, components in COMBO_FAULTS.items():
        candidates, ratios = evaluate_combo(fault, scaler, model, healthy_family_means)
        candidate_names = {c[0] for c in candidates}
        n_files += 1

        detectable = {fam for fam, has_formula in components.items() if has_formula}
        undetectable = {fam for fam, has_formula in components.items() if not has_formula}

        hits = detectable & candidate_names
        misses = detectable - candidate_names
        false_positives = candidate_names - detectable  # candidates that aren't a true component at all

        n_detectable_components += len(detectable)
        n_components_hit += len(hits)
        n_false_positive_components += len(false_positives)

        print(f"{fault}")
        print(f"  true components:         {sorted(components.keys())}"
              + (f"  (no detector yet: {sorted(undetectable)})" if undetectable else ""))
        cand_str = ", ".join(f"{f}={r:.2f}" for f, r in candidates) if candidates else "(none)"
        print(f"  candidates found:        {cand_str}")
        print(f"  detectable components hit: {sorted(hits)}   missed: {sorted(misses)}")
        if false_positives:
            print(f"  FALSE POSITIVE component(s): {sorted(false_positives)}  <-- unrelated family incorrectly implicated")
        tag = "CLEAN" if not misses and not false_positives else ("PARTIAL" if hits else "MISSED")
        if false_positives:
            tag = "FALSE POSITIVE"
        print(f"  [{tag}]\n")

    print(f"=== summary: {n_files} combo files, {n_detectable_components} scorable components "
          f"(components with no detector yet are excluded) ===")
    print(f"detectable components correctly flagged: {n_components_hit}/{n_detectable_components}")
    print(f"combo files with a false-positive component: {n_false_positive_components}")


if __name__ == "__main__":
    main()
