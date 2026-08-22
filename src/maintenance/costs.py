"""
Layer 3, cost estimation: turns FleetSense's detected issues into repair cost bands.

Deliberately knows nothing about signals, datasets or models -- it consumes only the
`issues` list that pipeline_mcc5.check_motor() returns, so it stays usable if the
diagnosis layer is ever retrained or swapped for a different dataset. The dependency
runs one way: the app calls the diagnosis pipeline, then calls this. Nothing here
imports from src/mcc5.

All figures live in cost_reference.json, not in this file -- a maintenance engineer
should be able to reprice a bearing without editing Python.

Three things this has to get right, none of which a naive per-issue lookup does:

  1. DEDUPE. bearing_outer / bearing_inner / bearing_ball are three detections of one
     physical bearing. They map to a single job, so a file flagging all three is one
     bearing replacement, not three.

  2. ABSORPTION. Some jobs already contain others: the published stator rewind includes
     new bearings and varnishing, so a rewind plus a separate bearing job would bill the
     bearings twice.

  3. CATEGORY SPLIT. voltage_unbalance is not a motor repair -- it is upstream
     switchgear work for a different trade and a different budget line. Folding it into
     "motor repair cost" would send someone to fix the wrong thing.

On severity: severity does NOT scale the repair price. Replacing a bearing costs the
same whether the fault is graded low or high -- you fit the same bearing either way.
What severity legitimately drives is URGENCY: how long you can defer the work before
risking a failure that costs far more than the repair. So severity maps to an urgency
band here, not to a cost multiplier. Inventing a "high severity = 1.6x price" factor
would be fabricated precision, and it would be the one number in the whole project with
no source behind it.
"""
import json
from pathlib import Path

REFERENCE_PATH = Path(__file__).resolve().parent / "cost_reference.json"

# Severity -> how long the work can reasonably be deferred. Qualitative on purpose:
# turning this into "fails in 43 days" would imply remaining-useful-life modelling that
# FleetSense does not do -- it reports the state of a recording, not a time to failure.
URGENCY = {
    "high": {
        "band": "prompt",
        "guidance": "Schedule at the next available maintenance window.",
    },
    "low": {
        "band": "planned",
        "guidance": "Monitor and fold into routine servicing.",
    },
    "unknown": {
        "band": "assess",
        "guidance": "Severity could not be graded for this fault type -- inspect to confirm before deciding.",
    },
}


def load_reference(path: Path = REFERENCE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _severity_key(issue: dict) -> str:
    """check_motor() emits severity as 'high'/'low', or a long 'not assessable ...'
    string for the six locations with no severity model."""
    sev = (issue.get("severity") or "").strip().lower()
    return sev if sev in ("high", "low") else "unknown"


def estimate(issues: list, reference: dict = None) -> dict:
    """issues: the list from check_motor()['issues'].

    Returns per-issue cost annotations (keyed by location, in the order given) plus a
    resolved summary. Returns empty structures for an empty issue list rather than
    zeroes, so callers can distinguish "healthy" from "costs nothing to fix"."""
    ref = reference or load_reference()
    jobs, locations = ref["repair_jobs"], ref["locations"]

    if not issues:
        return {"per_issue": [], "summary": None}

    # --- which job does each detected location call for ---
    known = [i for i in issues if i.get("location") in locations]
    job_to_locations = {}
    for issue in known:
        job_id = locations[issue["location"]]["job"]
        job_to_locations.setdefault(job_id, []).append(issue["location"])

    selected = set(job_to_locations)

    # --- absorption: drop jobs already contained in a larger selected job ---
    absorbed = {}
    for job_id in list(selected):
        for sub in jobs[job_id].get("absorbs", []):
            if sub in selected:
                selected.discard(sub)
                absorbed[sub] = job_id

    # --- per-issue annotations ---
    per_issue = []
    for issue in issues:
        loc = issue.get("location")
        if loc not in locations:
            per_issue.append({"location": loc, "costed": False,
                              "reason": "No cost reference for this location."})
            continue

        loc_entry = locations[loc]
        job_id = loc_entry["job"]
        job = jobs[job_id]
        sev = _severity_key(issue)

        shared_with = [l for l in job_to_locations[job_id] if l != loc]
        covered_by = absorbed.get(job_id)

        entry = {
            "location": loc,
            "costed": True,
            "job": job_id,
            "job_label": job["label"],
            "action": job["action"],
            "category": job["category"],
            "cost_eur": job["cost_eur"],
            "confidence": job["confidence"],
            "counts_toward_total": job_id in selected,
            # Same physical job reached from more than one detection -- the UI must say
            # so, or a reader will mentally add the identical figure twice.
            "shared_with": shared_with,
            # This job's cost is already inside another selected job's price.
            "covered_by": covered_by,
            "covered_by_label": jobs[covered_by]["label"] if covered_by else None,
            "urgency": URGENCY[sev]["band"],
            "urgency_guidance": URGENCY[sev]["guidance"],
        }
        for optional in ("confidence_note", "economic_alternative", "economic_note", "important", "note"):
            value = job.get(optional) or loc_entry.get(optional)
            if value:
                entry[optional] = value
        if entry.get("economic_alternative"):
            entry["economic_alternative_cost_eur"] = jobs[entry["economic_alternative"]]["cost_eur"]
            entry["economic_alternative_label"] = jobs[entry["economic_alternative"]]["label"]
        per_issue.append(entry)

    # --- totals, split by category ---
    def band(job_ids):
        return {"min": sum(jobs[j]["cost_eur"]["min"] for j in job_ids),
                "max": sum(jobs[j]["cost_eur"]["max"] for j in job_ids)}

    motor_jobs = sorted(j for j in selected if jobs[j]["category"] != "electrical_supply")
    other_jobs = sorted(j for j in selected if jobs[j]["category"] == "electrical_supply")

    lowest_confidence = None
    if selected:
        order = ["low", "medium", "medium_high", "high"]
        lowest_confidence = min((jobs[j]["confidence"] for j in selected),
                                key=lambda c: order.index(c) if c in order else 0)

    summary = {
        "currency": ref["meta"]["currency"],
        "motor_repair": {"jobs": motor_jobs, "cost_eur": band(motor_jobs)} if motor_jobs else None,
        "other_work": {"jobs": other_jobs, "cost_eur": band(other_jobs)} if other_jobs else None,
        "jobs": [{"job": j, "label": jobs[j]["label"], "action": jobs[j]["action"],
                  "category": jobs[j]["category"], "cost_eur": jobs[j]["cost_eur"],
                  "confidence": jobs[j]["confidence"]} for j in motor_jobs + other_jobs],
        "lowest_confidence": lowest_confidence,
        "market": ref["meta"]["market"],
        "market_warning": ref["meta"]["market_warning"],
    }
    return {"per_issue": per_issue, "summary": summary}
