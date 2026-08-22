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

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import (LoginManager, current_user, login_required, login_user,
                         logout_user)

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC / "mcc5"))
sys.path.insert(0, str(SRC / "maintenance"))

from pipeline_mcc5 import check_motor  # noqa: E402
from costs import estimate as estimate_costs  # noqa: E402
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


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
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
            return redirect(url_for("index"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = db.session.query(User).filter_by(email=email).first()
        # Same message either way: distinguishing them tells an attacker which emails
        # are registered.
        if user and user.check_password(request.form.get("password", "")):
            login_user(user, remember=True)
            return redirect(request.args.get("next") or url_for("index"))
        flash("Incorrect email or password.")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


ALLOWED_EXTENSIONS = {".csv", ".txt"}


@app.route("/analyze", methods=["POST"])
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
