"""
FleetSense web app -- first version. Upload a recording, run it through validation and
the real Layer 1/2 pipeline (pipeline_mcc5.check_motor), get the result back as JSON.

No aesthetics, no health checks, no load balancing yet -- explicitly deferred until the
cloud deployment phase. ML code stays entirely in src/mcc5, imported here rather than
duplicated, so the two can be containerized separately later if that ends up making sense.
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

app = Flask(__name__)

# Config comes from the environment so the same image runs locally and on AWS.
# DATABASE_URL unset -> local SQLite file; set to a mysql+pymysql://... URL -> RDS.
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", f"sqlite:///{Path(app.instance_path) / 'fleetsense.db'}")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Reconnect rather than hand out a connection RDS has already dropped.
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 280}

# A fixed fallback keeps local development frictionless, but it MUST be overridden in
# any real deployment -- with a known secret, anyone can forge a session cookie.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-insecure-key")
if app.secret_key == "dev-only-insecure-key" and not app.debug:
    print("WARNING: SECRET_KEY is unset; sessions are forgeable. Set it before deploying.")

Path(app.instance_path).mkdir(parents=True, exist_ok=True)
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
    "bearing_outer": "Bearing — outer race",
    "bearing_inner": "Bearing — inner race",
    "bearing_ball": "Bearing — rolling element",
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


@app.route("/")
@login_required
def fleet():
    """The landing page: this site's analysis history, newest first. The same motor can be
    analysed repeatedly, so this is a log of recordings rather than a roster of machines."""
    analyses = (current_user.analyses
                .order_by(Analysis.created_at.desc())
                .all())
    return render_template("fleet.html", analyses=analyses,
                           location_label=lambda k: LOCATION_LABELS.get(k, k),
                           cost_band=_cost_band)


@app.route("/maintenance", methods=["GET", "POST"])
@login_required
def maintenance():
    """Selection then plan. The manager decides which recordings are in scope -- some
    machines are not their responsibility, and the same motor may appear more than once
    with only one recording worth acting on. Nothing is auto-selected away."""
    analyses = (current_user.analyses
                .order_by(Analysis.created_at.desc())
                .all())

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
                chosen = set(request.form.getlist("include", type=int))
                # Filtered from the user's OWN analyses, so a posted id belonging to
                # somebody else never matches rather than needing a separate check.
                selected = [a for a in analyses if a.id in chosen]
                plan = build_plan([{"id": a.id, "label": a.motor_label, "result": a.result}
                                   for a in selected], budget)

    return render_template("maintenance.html", analyses=analyses, plan=plan,
                           budget_error=budget_error, budget_value=budget_value,
                           location_label=lambda k: LOCATION_LABELS.get(k, k))


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


@app.route("/analyse")
@login_required
def analyse_page():
    return render_template("analyse.html")


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
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"verdict": "rejected", "reason": "No file provided."}), 400

    uploaded = request.files["file"]

    ext = Path(uploaded.filename).suffix.lower()
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
                 or Path(uploaded.filename).stem)
        record = Analysis.from_result(current_user.id, label[:120],
                                      uploaded.filename[:255], result)
        db.session.add(record)
        db.session.commit()
        result["analysis_id"] = record.id

    status_code = 200 if result["verdict"] not in ("rejected", "error") else 400
    return jsonify(result), status_code


if __name__ == "__main__":
    app.run(port=5000, debug=True)
