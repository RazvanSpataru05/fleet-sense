"""
FleetSense web app. A recording goes through validation and the Layer 1/2 pipeline
(pipeline_mcc5.check_motor); costs (Layer 3) and the budget scheduler are composed on top
of the result here rather than inside the model code, so the diagnosis stays cost-unaware
and the cost layer stays model-unaware.

ML code lives entirely in src/mcc5 and is imported, never duplicated, so the two could be
containerized separately if that ever became useful.

Recordings are grouped by motor for display: the fleet page, the motor history page and
the maintenance plan all key off _motor_key, so one machine is one row wherever it appears.
"""
import os
import sys
import tempfile
from pathlib import Path

from flask import (Flask, abort, request, jsonify, render_template, redirect, url_for,
                   flash)
from flask_login import (LoginManager, current_user, login_required, login_user,
                         logout_user)

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC / "mcc5"))
sys.path.insert(0, str(SRC / "maintenance"))

from pipeline_mcc5 import check_motor  # noqa: E402
from costs import estimate as estimate_costs  # noqa: E402
from schedule import build_plan  # noqa: E402
# Layers 2 and 3 are composed here rather than inside check_motor: the diagnosis code
# stays dataset-specific and cost-unaware, and the cost layer stays model-unaware.
from models import db, User, Analysis  # noqa: E402
from faults import FAULTS  # noqa: E402

app = Flask(__name__)

# Config comes from the environment so the same image runs locally and on AWS.
# DATABASE_URL unset -> local SQLite file; set to a mysql+pymysql://... URL -> RDS.
# Pinned to this file's own directory rather than app.instance_path. Flask derives that
# path from the import name, which resolves differently for `python app/app.py` than for
# importing the module -- so the project silently kept TWO SQLite files, app/instance/ and
# instance/, and whichever one the app wrote to was invisible to the other. Symptom was a
# login that worked one day and failed the next against an empty database. An explicit path
# has no such ambiguity. Only affects local development; DATABASE_URL wins when set, which
# is always the case in the container.
LOCAL_DB = Path(__file__).resolve().parent / "instance" / "fleetsense.db"
LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", f"sqlite:///{LOCAL_DB}")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Reconnect rather than hand out a connection RDS has already dropped.
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 280}

# A genuine recording is ~113 MB. Flask accepts unlimited bodies by default, which behind a
# public URL is an easy way to fill the container's disk -- 200 MB leaves headroom over a
# real file while putting a ceiling on it.
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# A fixed fallback keeps local development frictionless, but it MUST be overridden in
# any real deployment -- with a known secret, anyone can forge a session cookie.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-insecure-key")
if app.secret_key == "dev-only-insecure-key" and not app.debug:
    print("WARNING: SECRET_KEY is unset; sessions are forgeable. Set it before deploying.")

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Sign in to analyse recordings."


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()


LOCATION_LABELS = {
    "bearing_outer": "Bearing (outer race)",
    "bearing_inner": "Bearing (inner race)",
    "bearing_ball": "Bearing (rolling element)",
    "rotor_bar": "Broken rotor bar",
    "static_eccentricity": "Static eccentricity",
    "dynamic_eccentricity": "Dynamic eccentricity",
    "winding": "Stator winding",
    "voltage_unbalance": "Supply voltage unbalance",
    "bend": "Bent shaft",
}


def _cost_band(analysis):
    """Motor-repair band for the fleet list. Supply-side work is excluded here on
    purpose -- it is a different budget line and folding it in would misstate the
    maintenance cost of the machine itself."""
    summary = (analysis.result or {}).get("costs", {}).get("summary")
    if not summary or not summary.get("motor_repair"):
        return None
    return summary["motor_repair"]["cost_eur"]


def _issue_cost(analysis, location):
    """The estimated band for ONE location in an analysis, as the string the UI shows.

    _cost_band above is the whole-analysis motor-repair total; this is the per-finding
    figure that sits behind an individual chip, which is what a question about "this
    fault" should be answered with."""
    for entry in ((analysis.result or {}).get("costs") or {}).get("per_issue") or []:
        if entry.get("location") == location and entry.get("cost_eur"):
            band = entry["cost_eur"]
            return f"€{band['min']}–{band['max']}"
    return None


def _not_plannable(analysis):
    """Why this recording cannot enter a plan, or None if it can.

    A reason rather than a bare bool: a motor that drops out of planning should say why.
    Silently omitting a machine the manager knows is faulty is the failure mode worth
    designing against -- they would simply not notice it was missing."""
    if analysis.issues:
        return None
    if analysis.verdict == "anomaly_detected_unattributed":
        # No located fault, but "inspect on site" is still work the scheduler can plan.
        return None
    if analysis.verdict in ("cannot_process", "rejected", "error"):
        return "latest recording could not be processed — re-measure"
    return "no findings in the latest recording"


_SEVERITY_RANK = {"none": 0, "unknown": 1, "low": 2, "high": 3}


def _motor_key(label):
    """Grouping key for a motor name.

    The name is typed by hand on every upload, so one machine arrives as "Pump Motor 3",
    "Pump motor 3" and "Pump  Motor 3". Capitalisation and runs of whitespace are the two
    things people vary without meaning to, so those are the only things normalised away --
    deciding that "Pump Motor 3" and "Pump Motor #3" are one machine would guess wrong in
    both directions. The datalist on the upload form is what keeps names identical in the
    first place; this is the net underneath it.

    Grouping here rather than rewriting the label on save means recordings already in the
    database collapse correctly with no migration, and the text the user actually typed
    survives to be displayed.
    """
    return " ".join(label.split()).casefold()


def _trend(history):
    """"worse" / "better" / "stable" from the two most recent recordings, or None.

    One recording has no trend, and reporting one would invent a direction from a single
    point. This is what the roster exists for: a machine that read healthy last week and
    does not now matters more than one that has always read "low"."""
    if len(history) < 2:
        return None
    now, before = (_SEVERITY_RANK[h.worst_severity] for h in history[:2])
    return "worse" if now > before else ("better" if now < before else "stable")


def _group_by_motor(analyses):
    """One entry per machine, from a newest-first log of recordings.

    `analyses` must arrive newest-first: the first row seen for a key supplies both that
    motor's current state and the spelling shown for it, so re-analysing with different
    capitalisation quietly acts as a rename."""
    motors = {}
    for a in analyses:
        key = _motor_key(a.motor_label)
        if key not in motors:
            # Case is kept as last typed -- "PUMP MOTOR 3" may well be deliberate -- but
            # runs of whitespace are collapsed, since a double space is a slip every time
            # and rendering it back looks like the page is broken.
            motors[key] = {"key": key, "name": " ".join(a.motor_label.split()),
                           "latest": a, "history": []}
        motors[key]["history"].append(a)
    for motor in motors.values():
        motor["count"] = len(motor["history"])
        motor["trend"] = _trend(motor["history"])
    # Most recently analysed first. Severity-first ordering was tried and read as arbitrary:
    # a motor analysed a minute ago could sit in the middle of the list because two others
    # scored worse, so the list did not respond to what the user had just done. Recency is
    # predictable, and severity is still obvious at a glance from the tone stripe and chips.
    # Name breaks ties so the order is stable when timestamps collide.
    return sorted(motors.values(),
                  key=lambda m: (-m["latest"].created_at.timestamp(), m["key"]))


@app.route("/")
@login_required
def fleet():
    """The landing page: one row per machine, not one per file. The same motor is analysed
    repeatedly, and grouping those recordings is what turns a log into a roster -- a plant
    manager asks which machines need attention, never which files were uploaded."""
    analyses = (current_user.analyses
                .order_by(Analysis.created_at.desc())
                .all())
    return render_template("fleet.html", motors=_group_by_motor(analyses),
                           location_label=lambda k: LOCATION_LABELS.get(k, k),
                           cost_band=_cost_band)


@app.route("/motor")
@login_required
def motor_detail():
    """Every recording of one machine, newest first.

    The name arrives as a query parameter rather than a path segment because labels
    contain spaces and slashes, and it is matched on the normalised key so a link still
    resolves after the spelling of the name changes."""
    key = _motor_key(request.args.get("name", ""))
    analyses = (current_user.analyses
                .order_by(Analysis.created_at.desc())
                .all())
    # Built from the user's OWN analyses, so a key belonging to somebody else simply fails
    # to match rather than needing a separate ownership check.
    motors = _group_by_motor(analyses)
    motor = next((m for m in motors if m["key"] == key), None)
    if motor is None:
        abort(404)
    # Everything except this motor: the rename field offers them for merging, and the
    # browser uses the counts to say what a merge would actually combine.
    others = [{"key": m["key"], "name": m["name"], "count": m["count"]}
              for m in motors if m["key"] != key]

    # How often each location has been flagged across this motor's recordings. This is what
    # keeps the detail panel from being a glossary: reference text describes a fault type,
    # while "flagged in 3 of 4 recordings since 18 Aug" describes this machine. Iterated
    # oldest-first so `first` is genuinely the earliest occurrence.
    recurrence = {}
    for a in reversed(motor["history"]):
        for issue in a.issues:
            location = issue.get("location")
            if location not in FAULTS:
                continue
            seen = recurrence.setdefault(
                location, {"count": 0, "first": a.created_at.strftime("%d %b %Y")})
            seen["count"] += 1

    return render_template("motor.html", motor=motor, others=others,
                           faults=FAULTS, recurrence=recurrence,
                           location_label=lambda k: LOCATION_LABELS.get(k, k),
                           location_labels=LOCATION_LABELS,
                           cost_band=_cost_band, issue_cost=_issue_cost)


@app.route("/maintenance", methods=["GET", "POST"])
@login_required
def maintenance():
    """Selection then plan, one row per machine rather than one per recording.

    You repair a motor, not a file, so selection is per motor and only its NEWEST recording
    can enter a plan. Budgeting from a superseded reading is the mistake condition
    monitoring exists to prevent, and restricting it here also makes the old double-count
    warning ("if they are the same machine, deselect the older one") structurally
    impossible instead of something the reader has to remember to act on.

    The manager still decides which machines are in scope -- some are not their
    responsibility -- so nothing is auto-selected away.
    """
    analyses = (current_user.analyses
                .order_by(Analysis.created_at.desc())
                .all())
    motors = _group_by_motor(analyses)
    for motor in motors:
        motor["blocked"] = _not_plannable(motor["latest"])

    # Only a motor's newest recording is ever plannable. Enforced here and not just by
    # rendering the older ones without a checkbox, because the form is not the only thing
    # that can POST to this endpoint.
    selectable = {m["latest"].id for m in motors if not m["blocked"]}

    plan, budget_error, budget_value = None, None, ""
    if request.method == "POST":
        budget_value = request.form.get("budget", "").strip()
        # Validated here, not just in the browser -- the client-side attributes are a
        # convenience, and anything can POST to this endpoint directly.
        try:
            budget = float(budget_value.replace(",", "."))
        except ValueError:
            budget_error = "Enter the available budget as a number, e.g. 500."
        else:
            if budget <= 0:
                budget_error = "The budget must be greater than zero."
            else:
                # Intersected with `selectable`, so an id for an older recording -- or one
                # belonging to somebody else, which is not in this user's motors at all --
                # is dropped rather than planned.
                chosen = set(request.form.getlist("include", type=int)) & selectable
                selected = [a for a in analyses if a.id in chosen]
                plan = build_plan([{"id": a.id, "label": a.motor_label, "result": a.result}
                                   for a in selected], budget)

    return render_template("maintenance.html", motors=motors, plan=plan,
                           budget_error=budget_error, budget_value=budget_value,
                           location_label=lambda k: LOCATION_LABELS.get(k, k))


@app.route("/analysis/<int:analysis_id>")
@login_required
def view_analysis(analysis_id):
    """A stored analysis, rendered by the same code that renders a fresh one.

    Nothing is recomputed. Analysis.result holds the complete pipeline output -- the exact
    JSON /api/analyze returns -- so the results panel, the cost blocks, the 3D viewer and
    the fault panel all work from the database alone. The recording itself is never needed;
    the viewer only ever sees the issues list.
    """
    analysis = db.session.get(Analysis, analysis_id)
    # 404 rather than 403 when it belongs to somebody else, matching delete_analysis:
    # replying "forbidden" would confirm the id exists.
    if analysis is None or analysis.user_id != current_user.id:
        abort(404)

    analyses = (current_user.analyses
                .order_by(Analysis.created_at.desc())
                .all())
    return render_template("analyse.html",
                           motor_names=[m["name"] for m in _group_by_motor(analyses)],
                           location_labels=LOCATION_LABELS, faults=FAULTS,
                           stored=analysis)


@app.route("/motor/rename", methods=["POST"])
@login_required
def rename_motor():
    """Rewrite the label on every recording of one machine.

    Renaming onto a name that already exists IS the merge: grouping is by normalised
    label, so once the rows carry the same text they are the same motor. "I mistyped this"
    and "these two are the same machine" are the same mistake seen from either side, so
    they get one control rather than two.

    This is an UPDATE of motor_label on the affected rows and nothing else -- no row is
    created or removed, ids and results are untouched, and there is no schema change.
    """
    analyses = (current_user.analyses
                .order_by(Analysis.created_at.desc())
                .all())
    old_raw = request.form.get("name", "")
    old_key = _motor_key(old_raw)
    # Collapsed the same way the roster collapses it for display, so a stray double space
    # cannot create a name that renders differently from what was typed.
    new_name = " ".join(request.form.get("new_name", "").split())

    rows = [a for a in analyses if _motor_key(a.motor_label) == old_key]
    # Selected from the user's OWN analyses, so a name belonging to somebody else finds
    # nothing rather than needing a separate ownership check.
    if not rows:
        abort(404)

    back = redirect(url_for("motor_detail", name=old_raw))
    if not new_name:
        flash("Enter a name for the motor.", "error")
        return back
    if len(new_name) > 120:
        flash("Motor names are limited to 120 characters.", "error")
        return back

    # Counted before the update, while the two groups are still distinguishable.
    absorbed = len([a for a in analyses
                    if _motor_key(a.motor_label) == _motor_key(new_name)
                    and _motor_key(a.motor_label) != old_key])

    for a in rows:
        a.motor_label = new_name
    db.session.commit()

    if absorbed:
        flash(f"Merged {len(rows)} recording{'' if len(rows) == 1 else 's'} into "
              f"{new_name}, which now has {len(rows) + absorbed}.", "notice")
    elif _motor_key(old_raw) != _motor_key(new_name) or old_raw != new_name:
        flash(f"Renamed to {new_name}.", "notice")
    return redirect(url_for("motor_detail", name=new_name))


@app.route("/analysis/<int:analysis_id>/delete", methods=["POST"])
@login_required
def delete_analysis(analysis_id):
    analysis = db.session.get(Analysis, analysis_id)
    # 404 rather than 403 when it belongs to somebody else: replying "forbidden" would
    # confirm that the id exists, which is information another account should not get.
    if analysis is None or analysis.user_id != current_user.id:
        abort(404)
    db.session.delete(analysis)
    db.session.commit()
    return redirect(url_for("fleet"))


@app.route("/analyze")
@login_required
def analyse_page():
    analyses = (current_user.analyses
                .order_by(Analysis.created_at.desc())
                .all())
    # Offered to the name field as a datalist: picking the existing spelling is what stops
    # one machine from splitting into three, and it is better than normalising the drift
    # away afterwards.
    return render_template("analyse.html",
                           motor_names=[m["name"] for m in _group_by_motor(analyses)],
                           location_labels=LOCATION_LABELS, faults=FAULTS)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("fleet"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        factory = request.form.get("factory_name", "").strip()

        if not email or not password or not factory:
            flash("All fields are required.")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.")
        elif db.session.query(User).filter_by(email=email).first():
            flash("An account with that email already exists.")
        else:
            user = User(email=email, factory_name=factory)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for("fleet"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("fleet"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = db.session.query(User).filter_by(email=email).first()
        # Same message either way: distinguishing them tells an attacker which emails
        # are registered.
        if user and user.check_password(request.form.get("password", "")):
            login_user(user, remember=True)
            return redirect(request.args.get("next") or url_for("fleet"))
        flash("Incorrect email or password.")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


ALLOWED_EXTENSIONS = {".csv", ".txt"}


@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"verdict": "rejected", "reason": "No file provided."}), 400
    source_name = uploaded.filename

    ext = Path(source_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"verdict": "rejected",
                         "reason": f"Unsupported file type {ext!r}. Accepted: "
                                   f"{', '.join(sorted(ALLOWED_EXTENSIONS))}."}), 400

    declared_sample_rate = None
    raw_rate = request.form.get("sample_rate", "").strip()
    if raw_rate:
        try:
            declared_sample_rate = float(raw_rate)
        except ValueError:
            return jsonify({"verdict": "rejected",
                             "reason": f"Sample rate must be a number, got: {raw_rate!r}"}), 400

    # NamedTemporaryFile(delete=False), not the usual context-manager form -- on Windows
    # the file stays exclusively locked while that handle is open, so a second open()
    # (uploaded.save() below) fails with PermissionError. Close it immediately, save into
    # the now-unlocked path, then clean up manually.
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.close()
    try:
        uploaded.save(tmp.name)
        try:
            result = check_motor(tmp.name, declared_sample_rate=declared_sample_rate)
        except Exception as e:
            return jsonify({"verdict": "error", "reason": f"Unexpected error while processing file: {e}"}), 500
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    if result["verdict"] not in ("rejected", "error", "cannot_process"):
        result["costs"] = estimate_costs(result.get("issues", []))

        # Only completed analyses are kept. Rejections are input errors, not fleet
        # history -- storing them would clutter the motor list with non-events.
        label = (request.form.get("motor_label", "").strip()
                 or Path(source_name).stem)
        record = Analysis.from_result(current_user.id, label[:120],
                                      source_name[:255], result)
        db.session.add(record)
        db.session.commit()
        result["analysis_id"] = record.id

    status_code = 200 if result["verdict"] not in ("rejected", "error") else 400
    return jsonify(result), status_code


if __name__ == "__main__":
    app.run(port=5000, debug=True)
