"""
Validates a real, arbitrary uploaded CSV before it's handed to the pipeline -- the
"background checks" step of the app's upload flow. This is deliberately separate from
pipeline_mcc5.check_motor(): a file either passes these checks and gets a validated,
schema-normalized DataFrame, or it's rejected with a specific, user-facing reason. The
model is never run on something that hasn't passed this gate.

What's actually checked, and why:

1. Column shape. Every model/feature in this project only ever reads 4 columns --
   current_a, current_b, current_c, torque (see load_mcc5.REQUIRED_COLUMNS, confirmed by
   grepping every module). "time", "speed", and the 3 vibration channels are in the raw
   dataset but genuinely unused anywhere -- a real upload without vibration sensors (the
   whole point of current-based monitoring vs needing vibration hardware) should not be
   rejected over columns nothing reads. Two shapes are accepted:
     - Headered: any column order/subset, matched by name, as long as the 4 required
       names are present (case/whitespace-insensitive).
     - Headerless: must match the raw dataset's exact 9-column positional layout, since
       without names there's no other way to know which column is which.

2. Sample rate. Every frequency-domain feature (window sizing, FFT resolution, bin
   grouping) is built around FS=12800Hz -- not a flexible parameter, a hard assumption
   baked into how features are computed. Getting this wrong doesn't degrade results, it
   silently produces confidently wrong ones. Current/torque values alone can't reveal a
   file's true sample rate (there's no time base encoded in them) -- there are exactly two
   valid sources: a "time" column (rate = 1/mean(diff(time))), or an explicit declaration
   from the uploader if "time" is absent. No default is silently assumed in that case.

3. Minimum length. Checked directly by truncating real recordings: shaft_hz and condition
   detection stabilize well before 60s, but rotor_bar_magnitude_for_file's target frequency
   (~0.13Hz) never stabilizes until close to the full ~90s recording -- confirmed
   empirically (40-230% error at 10-75s, not a smooth degradation). 60s is the hard
   minimum; anything from 60s up to ~90s is accepted but flagged as reduced-confidence
   specifically for rotor-bar/broken-bar detection, not rejected outright.

4. Basic sanity: not empty, no NaN/inf in the columns that are actually used.
"""
import numpy as np
import pandas as pd

from load_mcc5 import FS, COLUMNS, REQUIRED_COLUMNS

MIN_DURATION_SEC = 60.0
FULL_REFERENCE_DURATION_SEC = 90.0  # what the models were actually trained/validated on
SAMPLE_RATE_TOLERANCE = 0.01  # relative; a real recording's rate should match almost exactly


class UploadValidationError(Exception):
    """Raised with a specific, user-facing reason -- meant to be caught by the app layer
    and shown directly, not a generic failure."""
    pass


def _looks_like_header(first_row_tokens: list) -> bool:
    for tok in first_row_tokens:
        try:
            float(tok)
        except ValueError:
            return True
    return False


def load_uploaded_csv(csv_path) -> pd.DataFrame:
    """Loads an arbitrary uploaded CSV, handling both accepted shapes (see module
    docstring). Raises UploadValidationError with a specific reason for anything that
    fits neither -- never silently guesses."""
    with open(csv_path, "r") as f:
        first_line = f.readline()
    first_row_tokens = [t.strip() for t in first_line.strip().split(",")]

    if _looks_like_header(first_row_tokens):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise UploadValidationError(
                f"Missing required column(s): {', '.join(missing)}. A file with a header "
                f"row must include these exact column names: {', '.join(REQUIRED_COLUMNS)} "
                f"(other columns are fine to include or omit -- only these 4 are actually "
                f"used)."
            )
        return df

    n_cols = len(first_row_tokens)
    if n_cols != len(COLUMNS):
        raise UploadValidationError(
            f"File has no header row and {n_cols} column(s), but a headerless file must "
            f"have exactly {len(COLUMNS)} columns in this fixed order: {', '.join(COLUMNS)} "
            f"(that's the only way to know which column is which without names). Add a "
            f"header row instead if you want a different column set/order -- only "
            f"{', '.join(REQUIRED_COLUMNS)} are actually required."
        )
    return pd.read_csv(csv_path, header=None, names=COLUMNS)


def resolve_sample_rate(df: pd.DataFrame, declared_sample_rate: float = None) -> tuple:
    if "time" in df.columns:
        dt = np.diff(df["time"].values)
        sample_rate = float(1.0 / np.mean(dt))
        source = "time_column"
    elif declared_sample_rate is not None:
        sample_rate = float(declared_sample_rate)
        source = "user_declared"
    else:
        raise UploadValidationError(
            f"Cannot determine this file's sample rate: it has no 'time' column, and none "
            f"was provided. Please confirm the recording's sample rate before proceeding "
            f"(the models expect {FS}Hz -- a different rate will silently produce wrong "
            f"results, not just less accurate ones)."
        )

    relative_error = abs(sample_rate - FS) / FS
    if relative_error > SAMPLE_RATE_TOLERANCE:
        raise UploadValidationError(
            f"This recording's sample rate ({sample_rate:.1f}Hz, from {source}) doesn't "
            f"match what the models were trained on ({FS}Hz). Processing it anyway would "
            f"produce confidently wrong results -- rejected rather than silently degraded."
        )
    return sample_rate, source


def validate_upload(csv_path, declared_sample_rate: float = None) -> dict:
    """Returns a dict with the validated df and metadata on success. Raises
    UploadValidationError with a specific reason on any failure -- callers (the API layer)
    should catch this and surface result.args[0] directly to the user."""
    df = load_uploaded_csv(csv_path)

    if len(df) == 0:
        raise UploadValidationError("File is empty.")

    required = df[REQUIRED_COLUMNS]
    if required.isnull().any().any():
        raise UploadValidationError(
            "One or more required columns (current_a, current_b, current_c, torque) "
            "contain missing values."
        )
    if not np.isfinite(required.values).all():
        raise UploadValidationError(
            "One or more required columns contain non-finite values (inf/-inf), not just "
            "missing data -- this file looks corrupted."
        )

    sample_rate, sample_rate_source = resolve_sample_rate(df, declared_sample_rate)

    n_samples = len(df)
    duration_sec = n_samples / sample_rate

    if duration_sec < MIN_DURATION_SEC:
        raise UploadValidationError(
            f"Recording is only {duration_sec:.1f}s long; at least {MIN_DURATION_SEC:.0f}s "
            f"is needed for reliable analysis. Checked directly by truncating real "
            f"recordings: shorter ones produce unstable results, especially for the "
            f"rotor-bar/broken-bar fault signature."
        )

    return {
        "valid": True,
        "df": df,
        "n_samples": n_samples,
        "sample_rate": sample_rate,
        "sample_rate_source": sample_rate_source,
        "duration_sec": duration_sec,
        "reduced_confidence_rotor_bar": duration_sec < FULL_REFERENCE_DURATION_SEC * 0.95,
    }
