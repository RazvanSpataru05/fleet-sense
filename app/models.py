"""
Database models for FleetSense.

The product model is a plant manager with an account for their own site, uploading
recordings for the motors they are responsible for. So analyses belong to a user, and the
fleet view is "the motors this account has analysed".

Portability note: every String column carries an explicit length. SQLite ignores lengths,
but MySQL refuses to create an indexed VARCHAR without one -- declaring them here means the
same models run unchanged on local SQLite and on RDS MySQL, selected purely by
DATABASE_URL. The JSON column maps to native JSON on MySQL 5.7+ and to text on SQLite,
handled transparently by SQLAlchemy.
"""
from datetime import datetime, timezone

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


def _utcnow():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    # The site this account monitors -- shown in the fleet header.
    factory_name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow)

    analyses = db.relationship("Analysis", back_populates="user",
                               cascade="all, delete-orphan", lazy="dynamic")

    def set_password(self, password: str) -> None:
        # Werkzeug's default is a salted PBKDF2 hash; the plaintext is never stored.
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Analysis(db.Model):
    """One analysed recording. The full pipeline result is kept as JSON so the fleet view
    can re-render exactly what the user saw, while the columns alongside it exist so the
    list can be sorted and filtered without deserialising every row."""
    __tablename__ = "analyses"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow, index=True)

    # What the user calls this machine, e.g. "Line 3 feed pump". Defaults to the filename.
    motor_label = db.Column(db.String(120), nullable=False)
    filename = db.Column(db.String(255), nullable=False)

    verdict = db.Column(db.String(40), nullable=False, index=True)
    regime = db.Column(db.String(40), nullable=True)
    torque_nm = db.Column(db.Integer, nullable=True)
    rpm = db.Column(db.Integer, nullable=True)
    layer1_anomalous = db.Column(db.Boolean, nullable=True)
    layer1_anomaly_ratio = db.Column(db.Float, nullable=True)

    # Full check_motor() output plus the cost estimate, exactly as returned to the browser.
    result = db.Column(db.JSON, nullable=False)

    user = db.relationship("User", back_populates="analyses")

    @classmethod
    def from_result(cls, user_id: int, motor_label: str, filename: str, result: dict) -> "Analysis":
        cond = result.get("condition_detected") or {}
        return cls(
            user_id=user_id,
            motor_label=motor_label,
            filename=filename,
            verdict=result.get("verdict", "unknown"),
            regime=result.get("regime"),
            torque_nm=cond.get("torque_nm"),
            rpm=cond.get("rpm"),
            layer1_anomalous=result.get("layer1_anomalous"),
            layer1_anomaly_ratio=result.get("layer1_anomaly_ratio"),
            result=result,
        )

    @property
    def issues(self) -> list:
        return (self.result or {}).get("issues", []) or []

    @property
    def worst_severity(self) -> str:
        """high > low > unknown > none -- used for fleet ordering and colour.

        'none' means genuinely nothing found. A recording where Layer 1 flagged an
        anomaly that Layer 2 could not localise is NOT healthy, so it grades as
        'unknown' -- showing it green would tell a plant manager the machine is fine
        when the system is actually saying it looks abnormal but cannot say where."""
        sevs = {i.get("severity") for i in self.issues}
        if "high" in sevs:
            return "high"
        if "low" in sevs:
            return "low"
        if self.issues or self.verdict == "anomaly_detected_unattributed":
            return "unknown"
        return "none"
