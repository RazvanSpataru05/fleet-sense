"""
FleetSense web app -- first version. Upload a recording, run it through validation and
the real Layer 1/2 pipeline (pipeline_mcc5.check_motor), get the result back as JSON.

No aesthetics, no health checks, no load balancing yet -- explicitly deferred until the
cloud deployment phase. ML code stays entirely in src/mcc5, imported here rather than
duplicated, so the two can be containerized separately later if that ends up making sense.
"""
import sys
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template

MCC5_DIR = Path(__file__).resolve().parent.parent / "src" / "mcc5"
sys.path.insert(0, str(MCC5_DIR))

from pipeline_mcc5 import check_motor  # noqa: E402

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


ALLOWED_EXTENSIONS = {".csv", ".txt"}


@app.route("/analyze", methods=["POST"])
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

    status_code = 200 if result["verdict"] not in ("rejected", "error") else 400
    return jsonify(result), status_code


if __name__ == "__main__":
    app.run(port=5000, debug=True)
