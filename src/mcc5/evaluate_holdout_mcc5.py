"""
Blind evaluation on the ten held-out recordings.

These ten files were physically removed from the dataset before the final models were
trained, so nothing here has been seen during training -- unlike the leave-one-condition-out
figures reported elsewhere, this is a genuine holdout. Small (10 files), so treat it as an
illustration rather than a precise accuracy estimate.

The files were renamed test_1..test_10 and moved to dataset/mcc5-thu evaluate_dataset/.
Their true identities are recorded below. They were recovered independently from the data
itself -- each of the ten has a unique (regime, torque, rpm) signature, and the regime
detector plus condition auto-detection matched all ten correctly, which is itself a result
worth reporting.

Scoring is per LOCATION, not per file: a two-fault recording where we find one of the two
is a partial result, not a pass and not a failure, and is counted as such.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maintenance"))

from pipeline_mcc5 import check_motor          # noqa: E402
from presence_mcc5 import fault_set            # noqa: E402
from costs import estimate as estimate_costs   # noqa: E402

HOLDOUT_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "mcc5-thu evaluate_dataset"

# test_N -> (true fault label, expected regime, expected severity for bearing locations)
TRUTH = {
    "test_1":  ("bearing_outer_H",                            "torque_circulation", "high"),
    "test_2":  ("bearing_inner_L",                            "torque_circulation", "low"),
    "test_3":  ("static_eccentricity_H",                      "torque_circulation", None),
    "test_4":  ("voltage_unbalance_L",                        "torque_circulation", None),
    "test_5":  ("broken_bar",                                 "torque_circulation", None),
    "test_6":  ("bearing_outer_L",                            "speed_circulation",  "low"),
    "test_7":  ("static_eccentricity_H_and_bearing_outer_H",  "speed_circulation",  "high"),
    "test_8":  ("broken_bar_and_bearing_inner_H",             "speed_circulation",  "high"),
    "test_9":  ("bearing_inner_H",                            "speed_circulation",  "high"),
    "test_10": ("winding_H",                                  "speed_circulation",  None),
}

BEARING = {"bearing_outer", "bearing_inner", "bearing_ball"}


def main():
    tot_expected = tot_found = tot_false = 0
    regime_ok = severity_ok = severity_n = 0
    rows = []

    for name in sorted(TRUTH, key=lambda s: int(s.split("_")[1])):
        label, exp_regime, exp_sev = TRUTH[name]
        expected = fault_set(label)

        result = check_motor(HOLDOUT_DIR / f"{name}.csv")
        if result["verdict"] in ("rejected", "error", "cannot_process"):
            rows.append((name, label, expected, set(), "-", result["verdict"], None))
            tot_expected += len(expected)
            continue

        found = {i["location"] for i in result["issues"]}
        hits, false_pos = expected & found, found - expected
        tot_expected += len(expected)
        tot_found += len(hits)
        tot_false += len(false_pos)
        regime_ok += int(result["regime"] == exp_regime)

        sev = "-"
        if exp_sev:
            bearing_issues = [i for i in result["issues"] if i["location"] in BEARING
                              and i["location"] in expected]
            if bearing_issues:
                severity_n += 1
                got = bearing_issues[0].get("severity")
                sev = f"{got} (exp {exp_sev})"
                severity_ok += int(got == exp_sev)

        costs = estimate_costs(result["issues"])
        band = costs["summary"]["motor_repair"]["cost_eur"] if costs["summary"] and costs["summary"]["motor_repair"] else None
        rows.append((name, label, expected, found, sev, result["verdict"], band))

    print(f"{'file':8s} {'true fault':44s} {'expected':34s} {'detected':34s} {'severity':22s} cost")
    for name, label, expected, found, sev, verdict, band in rows:
        hits = expected & found
        mark = "OK " if hits == expected and not (found - expected) else ("PART" if hits else "MISS")
        cost = f"EUR {band['min']}-{band['max']}" if band else "-"
        print(f"{name:8s} {label[:44]:44s} {','.join(sorted(expected))[:34]:34s} "
              f"{(','.join(sorted(found)) or '(none)')[:34]:34s} {sev:22s} {cost}   [{mark}]")

    print()
    print(f"locations correctly identified : {tot_found}/{tot_expected}")
    print(f"false locations reported       : {tot_false}")
    print(f"regime routed correctly        : {regime_ok}/{len(TRUTH)}")
    if severity_n:
        print(f"bearing severity correct       : {severity_ok}/{severity_n}")


if __name__ == "__main__":
    main()
