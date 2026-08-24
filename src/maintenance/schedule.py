"""
Layer 3, scheduling: given a set of analysed recordings the plant manager has chosen to
include, rank them into a maintenance plan.

This is a TRANSPARENT SCORING RULE, not a learned model, and that is deliberate. Ranking
maintenance work properly would need failure-progression data -- how long a low-severity
bearing fault takes to become a failure, what that failure costs in lost production. We
have none of that, so any model claiming "this fails in six weeks" would be fabricated.
What we do have is severity, confidence, whether two independent layers agree, and a
sourced repair cost. Those are enough to rank sensibly, and every position in the ranking
can be explained to the engineer who has to act on it -- which matters more in a plant
than a marginally better ordering nobody can audit.

The four factors, and why each is defensible:

  SEVERITY      The only direct statement about how bad the fault is. Only assessable for
                bearing faults; everything else grades as "not assessable", which scores
                between low and high rather than being treated as harmless.

  CONFIDENCE    How sure the presence model is. Note these are RandomForest vote
                fractions, not calibrated probabilities, so they are used to ORDER work,
                never multiplied into an expected cost.

  CORROBORATION Whether Layer 1 independently flagged the recording as abnormal. Two
                methods agreeing on different features is stronger evidence than one.

  COST LEVERAGE A cheap repair on a serious finding is a better use of a maintenance slot
                than an expensive one on a weak finding. This is the only factor that
                looks at money, and it deliberately favours acting early on the cheap end.

Scores land roughly on 0-100 but are ordinal: the gap between 60 and 50 carries no
physical meaning, and the plan says so.
"""

# Weighting principle: severity is the only DIRECT statement about how bad the fault is.
# Everything else is supporting evidence, so the severity spread (35 points) has to be
# wide enough that circumstantial factors rarely overturn it. A first cut used a 20-point
# severity spread against 60 points of supporting factors, and a machine with three
# ungraded findings outranked one with a CONFIRMED high-severity bearing fault -- which is
# the wrong advice to give a maintenance planner.
SEVERITY_POINTS = {
    "high": 50,
    # "Detected, severity not assessable" is not "mild". We know a fault is present and we
    # cannot grade it, so unknown risk sits above a confirmed-mild finding, not below it.
    "unknown": 30,
    "low": 15,
}

CONFIDENCE_POINTS = 20         # scaled by the strongest finding's confidence
CORROBORATION_POINTS = 12      # both layers independently flagged the machine
EXTRA_FINDING_POINTS = 4       # per additional distinct finding, capped below
MAX_EXTRA_FINDING_POINTS = 8
LEVERAGE_POINTS = 8            # cheap fix, serious finding
UNLOCALISED_POINTS = 30        # abnormal, but Layer 2 could not say where

# A repair at or under this is "cheap enough that deferring it is rarely worth the risk".
# Anchored on the sourced bands: a bearing job is EUR 24-68, a motor replacement 145-405.
CHEAP_REPAIR_EUR = 100


def _severity_key(issue: dict) -> str:
    sev = (issue.get("severity") or "").strip().lower()
    return sev if sev in ("high", "low") else "unknown"


def score_analysis(result: dict) -> dict:
    """result: one stored check_motor() payload, including its 'costs' block.

    Returns the score plus the reasons behind it, so the UI can show why something ranked
    where it did instead of presenting a bare number."""
    issues = result.get("issues") or []
    verdict = result.get("verdict")
    reasons = []

    if not issues and verdict != "anomaly_detected_unattributed":
        return {"score": 0, "reasons": ["No fault detected."], "action": "No action required."}

    # --- an abnormal machine with no localised cause still needs looking at ---
    if not issues:
        return {
            "score": UNLOCALISED_POINTS,
            "reasons": ["Layer 1 flagged this recording as abnormal, but no specific "
                        "cause could be localised."],
            "action": "Inspect on site: the analysis cannot say which component.",
        }

    severities = [_severity_key(i) for i in issues]
    best_severity = ("high" if "high" in severities
                     else "unknown" if "unknown" in severities else "low")
    score = SEVERITY_POINTS[best_severity]
    reasons.append({"high": "High-severity fault confirmed.",
                    "unknown": "Fault detected; severity could not be graded.",
                    "low": "Low-severity fault."}[best_severity])

    top_confidence = max((i.get("presence_confidence") or 0) for i in issues)
    score += CONFIDENCE_POINTS * top_confidence
    reasons.append(f"Strongest finding at {round(top_confidence * 100)}% confidence.")

    if result.get("layer1_anomalous"):
        score += CORROBORATION_POINTS
        reasons.append("Anomaly detection independently agrees the machine is abnormal.")
    else:
        reasons.append("Anomaly detection did not flag this recording: the finding rests "
                       "on per-location analysis alone.")

    extra = min((len(issues) - 1) * EXTRA_FINDING_POINTS, MAX_EXTRA_FINDING_POINTS)
    if extra:
        score += extra
        reasons.append(f"{len(issues)} separate findings on one machine.")

    band = ((result.get("costs") or {}).get("summary") or {}).get("motor_repair")
    cost = band["cost_eur"] if band else None
    if cost and cost["max"] <= CHEAP_REPAIR_EUR and best_severity in ("high", "unknown"):
        score += LEVERAGE_POINTS
        reasons.append(f"Inexpensive to correct (EUR {cost['min']}-{cost['max']}) for the "
                       f"severity involved: good value for a maintenance slot.")

    if best_severity == "high":
        action = "Schedule at the next available maintenance window."
    elif best_severity == "unknown":
        action = "Inspect to confirm, then schedule."
    else:
        action = "Fold into routine servicing."

    return {"score": round(score, 1), "reasons": reasons, "action": action}


def build_plan(entries: list, budget_eur: float) -> dict:
    """entries: [{"id", "label", "result"}] -- the recordings the user selected.
    budget_eur: what the plant actually has to spend on motor repair this cycle.

    Work is committed against the WORST CASE of each cost band, not the best. Our figures
    are ranges (a bearing job is EUR 24-68), and a plan built on the optimistic end
    overruns the moment a shop quotes high -- which is precisely when a maintenance
    manager gets caught out. Fitting at the maximum means the plan survives every job
    coming in expensive, and underspend becomes headroom instead of a shortfall.

    Allocation is GREEDY IN PRIORITY ORDER, not an optimal knapsack packing. A knapsack
    solver would sometimes drop the most urgent machine in order to fit two cheaper ones
    and score better overall -- which is indefensible advice when the dropped machine is
    the one with a confirmed high-severity fault. Instead each job is taken in priority
    order if it fits, and skipped (not stopped at) if it does not, so cheaper lower-priority
    work still fills the remaining budget.

    Supply-side electrical work IS charged against this budget -- a manager entering a
    figure means money, not "motor work only, plus an unbounded amount of electrical".
    It stays labelled as a different trade, but the money is one pot. Motor work is
    charged FIRST and supply work takes what remains: motor condition is the domain this
    tool actually measures, so it has first call on the budget.

    That work is charged ONCE for the whole plan, not per machine. Voltage unbalance is a
    supply-side fault: if the feed is unbalanced every motor on it sees the same fault, and
    one electrician visit fixes the root cause for all of them -- the same reasoning that
    collapses three bearing detections into one bearing job. The caveat is that FleetSense
    has no plant topology, so it cannot know whether two flagged machines actually share a
    feed. The affected machines are listed so a manager who knows they are on separate
    transformers can budget one investigation per feed."""
    ranked = []
    for e in entries:
        scored = score_analysis(e["result"])
        band = ((e["result"].get("costs") or {}).get("summary") or {}).get("motor_repair")
        other = ((e["result"].get("costs") or {}).get("summary") or {}).get("other_work")
        ranked.append({**e, **scored,
                       "cost_eur": band["cost_eur"] if band else None,
                       "other_eur": other["cost_eur"] if other else None})

    ranked.sort(key=lambda r: (-r["score"], r["label"].lower()))
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i

    remaining = float(budget_eur)
    scheduled, deferred, no_action, awaiting_supply = [], [], [], []

    # --- motor work first: this is the domain the tool operates in, so it has first call
    # on the budget. Supply-side work is charged from whatever is left. ---
    for r in ranked:
        if r["score"] == 0:
            no_action.append(r)
            continue

        # A machine whose only remedy is the shared supply job has no motor cost of its
        # own, so a plain "does it fit" test would always pass and list it as scheduled
        # even if the electrical work turns out to be unaffordable. Held back until that
        # job's fate is known.
        if not r.get("cost_eur") and r.get("other_eur"):
            r["remedy"] = "electrical"
            awaiting_supply.append(r)
            continue

        worst = (r["cost_eur"] or {}).get("max", 0)
        if worst <= remaining:
            remaining -= worst
            scheduled.append(r)
        else:
            # Recorded so the plan can say what was left out and what it would take,
            # rather than silently dropping machines that still need attention.
            r["shortfall_eur"] = round(worst - remaining, 2)
            deferred.append(r)

    # --- then supply-side work, from the remainder ---
    affected = [r for r in ranked if r.get("other_eur")]
    electrical = None
    if affected:
        band = {"min": max(r["other_eur"]["min"] for r in affected),
                "max": max(r["other_eur"]["max"] for r in affected)}
        electrical = {"cost_eur": band,
                      "machines": [r["label"] for r in affected],
                      "scheduled": band["max"] <= remaining}
        if electrical["scheduled"]:
            remaining -= band["max"]
        else:
            electrical["shortfall_eur"] = round(band["max"] - remaining, 2)

    for r in awaiting_supply:
        if electrical["scheduled"]:
            scheduled.append(r)
        else:
            r["shortfall_eur"] = electrical["shortfall_eur"]
            deferred.append(r)

    # Held-back machines were appended after the loop, so restore priority order.
    scheduled.sort(key=lambda r: r["rank"])
    deferred.sort(key=lambda r: r["rank"])

    def total(rows, key):
        lo = sum(x[key]["min"] for x in rows if x.get(key))
        hi = sum(x[key]["max"] for x in rows if x.get(key))
        return {"min": lo, "max": hi} if hi else None

    # Two recordings of the same machine both generate work here; the manager chose the
    # selection, so this warns rather than silently dropping one.
    seen, duplicates = {}, []
    for r in ranked:
        key = r["label"].strip().lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            duplicates.append(r["label"])

    committed = total(scheduled, "cost_eur") or {"min": 0, "max": 0}
    if electrical and electrical["scheduled"]:
        committed = {"min": committed["min"] + electrical["cost_eur"]["min"],
                     "max": committed["max"] + electrical["cost_eur"]["max"]}

    return {
        "entries": ranked,
        "scheduled": scheduled,
        "deferred": deferred,
        "no_action": no_action,
        "electrical": electrical,
        "needing_action": len(scheduled) + len(deferred),
        "budget_eur": round(float(budget_eur), 2),
        "committed": committed,
        "remaining_eur": round(remaining, 2),
        "deferred_total": total(deferred, "cost_eur"),
        "duplicate_labels": duplicates,
    }
