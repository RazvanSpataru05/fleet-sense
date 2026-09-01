"""
Envelope-based feature extraction for the MCC5-THU dataset.

The dataset's own paper (Chen, Liu, Li, Zou, He & Zhou, Data in Brief 2026) gives the
exact bearing used (SKF 6205 2Z-C3) and its fault-frequency ratios directly: BPFO =
3.585x shaft frequency, BPFI = 5.415x, BSF = 2.357x. These were cross-checked against
the standard bearing-geometry formula using the paper's own pitch/ball diameter and
ball count and matched exactly, confirming both the geometry and a 0 degree contact
angle. We add these as three precise, targeted features per phase, on top of the wide
envelope-spectrum band from the first pass -- the targeted features should carry the
real signal for bearing-related faults, while the wide band is kept for fault types
that don't have a known formula yet (rotor bar, eccentricity, winding).

Originally scoped to a single operating condition (20Nm, 1000rpm) -- mixing conditions
dilutes the anomaly signal. Later extended to work across all 6 real conditions (2 torque levels x 3 RPMs) once RPM-
adaptive frequency search (see FUNDAMENTAL_SEARCH_MARGIN_HZ) made that safe -- see
classifier_mcc5.py, presence_mcc5.py, severity_mcc5.py, and anomaly_mcc5.py, which all
extract features across every condition rather than just one.
"""
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

from load_mcc5 import FS, load_recording, parse_filename

BASE_DIR = Path(__file__).resolve().parent


def artifacts_dir(split: str) -> Path:
    """Each split gets its own model/scaler/artifacts -- comparing a speed_circulation
    fault file against a torque_circulation healthy baseline (or vice versa) mixes in
    a systematic split-level difference unrelated to fault status (confirmed: healthy
    files score very differently by split alone). Keeping everything split-scoped
    avoids that confound entirely instead of trying to correct for it after the fact."""
    return BASE_DIR / "artifacts" / "envelope" / split


WINDOW_SEC = 0.5
OVERLAP = 0.5
WINDOW_SAMPLES = int(WINDOW_SEC * FS)
STRIDE_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))

FUNDAMENTAL_SEARCH_MARGIN_HZ = 5.0  # +/- around the nominal electrical frequency for this
# motor's RPM (2-pole: f_e ~= rpm/60, tiny slip). Was a fixed (10,30)Hz band -- only valid
# at 1000rpm (~16.7Hz); gave nonsense (negative/impossible slip) at 2000/3000rpm since
# their true fundamentals (~33.3Hz, ~50Hz) fall outside that fixed window entirely. Made
# adaptive per-file using the RPM already in every filename so all 3 RPMs (and both
# torque levels) become usable, not just the one condition this pipeline started scoped
# to. Checked against all 3 RPMs directly: detected frequency lands within 0.4% of
# nominal every time, comfortably inside a +/-5Hz window.
ENVELOPE_BAND = (0.0, 200.0)  # Hz, ~12x shaft frequency -- wide enough for any common bearing's BPFO/BPFI
BIN_GROUP = 4  # average-pool this many native (2Hz) bins together -> ~8Hz effective resolution.
# Only 359 windows exist at this single condition; native resolution would give 101
# bins/phase x 3 phases = 303 dims against ~287 training
# rows, close to 1:1 and a serious overfitting risk. Pooling trades resolution for a
# dimensionality the sample count can actually support.

# SKF 6205 2Z-C3, from the dataset's own paper (Table 7) -- fault frequency as a
# multiple of shaft rotational frequency. Cross-checked against the standard geometry
# formula using the paper's own pitch diameter (39.04mm), ball diameter (7.94mm), ball
# count (9), assuming 0 degree contact angle (standard for a deep-groove bearing) --
# matched all three ratios exactly.
BEARING_FAULT_RATIOS = {"bpfo": 3.585, "bpfi": 5.415, "bsf": 2.357}
TARGETED_FAULT_WINDOW_HZ = 2.0  # +/- search window around each target frequency

CURRENT_CHANNELS = ["current_a", "current_b", "current_c"]


def window_signal(signal, window_samples=WINDOW_SAMPLES, stride_samples=STRIDE_SAMPLES):
    n = len(signal)
    starts = range(0, n - window_samples + 1, stride_samples)
    return [signal[s:s + window_samples] for s in starts]


def compute_fft(signal, fs=FS):
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = np.abs(fft_vals) / n
    return freqs, magnitude


def detect_fundamental_hz(freqs, magnitude, band) -> float:
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    band_freqs, band_mag = freqs[mask], magnitude[mask]
    return float(band_freqs[np.argmax(band_mag)])


def fundamental_search_band(nominal_rpm: float, margin_hz: float = FUNDAMENTAL_SEARCH_MARGIN_HZ) -> tuple:
    nominal_hz = nominal_rpm / 60.0  # 2-pole motor: f_e ~= mechanical frequency
    return (nominal_hz - margin_hz, nominal_hz + margin_hz)


def shaft_hz_for_file(df, nominal_rpm: float) -> float:
    """Detected once per file from the full ~90s recording, not per 0.5s window --
    a short window's FFT frequency resolution is coarse enough that fundamental
    detection got noticeably unstable in spot checks (dropped to 10-15Hz instead of
    ~16.7Hz in some 10s slices). The full recording gives ~0.011Hz resolution and a
    stable estimate that's reused for every window in that file. nominal_rpm (from the
    filename, or auto-detected -- see detect_condition_from_data) picks the search band
    -- see fundamental_search_band."""
    freqs, magnitude = compute_fft(df["current_a"].values)
    return detect_fundamental_hz(freqs, magnitude, band=fundamental_search_band(nominal_rpm))


# Real deployment can't trust a filename to already know the operating condition -- a
# genuine upload won't be named "fault_40Nm_2000rpm.csv". These auto-detect torque/RPM
# straight from the recording's own data instead, so windows_for_file never has to parse
# a filename for anything except building our OWN labeled training set.
BLIND_SEARCH_BAND = (10.0, 60.0)  # wide enough to contain all 3 known RPMs' nominal
# frequencies (16.7, 33.3, 50.0 Hz) with margin, so the fundamental can be found without
# knowing the RPM first -- unlike fundamental_search_band, which needs it already known.
KNOWN_NOMINAL_RPMS = (1000, 2000, 3000)

# The "torque" column is NOT recorded in Nm -- it's some other, unlabeled unit (a per-unit
# fraction of rated torque, most likely). Checked directly against every file's true label:
# abs(torque.mean()) clusters at ~0.17-0.24 for real 20Nm runs and ~0.40-0.61 for real 40Nm
# runs, in BOTH splits, with a wide empty gap between -- so a single threshold in the middle
# separates them cleanly. abs() specifically because some recording sessions show a flipped
# sign convention on this channel (same kind of per-session wiring/convention inconsistency
# already found and fixed for negative-sequence current) -- without abs(), health-only checks
# looked clean but the full 24-fault-type sweep showed real overlap purely from sign flips.
# One single file across all 288 (bearing_outer_H_torque_circulation_20Nm_3000rpm, recorded
# ~7 weeks after every other file at that exact condition) reads 0.734 -- 4x its 23 same-
# condition siblings (all 0.17-0.19) -- a flagged, isolated data anomaly in the raw dataset,
# not something this threshold is tuned to paper over.
TORQUE_DETECTION_THRESHOLD = 0.3


def detect_condition_from_data(df) -> dict:
    """Two-pass frequency detection: a wide blind search first, just to work out WHICH of
    the known RPMs this recording is running at, then the existing precise narrow-band
    search (shaft_hz_for_file) centered on that RPM for the same clean estimate a labeled
    file would get. Torque uses the recording's own "torque" column directly -- see
    TORQUE_DETECTION_THRESHOLD above for why it's a threshold on the magnitude, not a
    literal Nm readout."""
    freqs, magnitude = compute_fft(df["current_a"].values)
    f_e_rough = detect_fundamental_hz(freqs, magnitude, band=BLIND_SEARCH_BAND)

    nominal_hz_by_rpm = {rpm: rpm / 60.0 for rpm in KNOWN_NOMINAL_RPMS}
    inferred_rpm = min(nominal_hz_by_rpm, key=lambda rpm: abs(nominal_hz_by_rpm[rpm] - f_e_rough))

    torque_measured = float(df["torque"].mean())
    inferred_torque_nm = 40 if abs(torque_measured) >= TORQUE_DETECTION_THRESHOLD else 20

    return {
        "rpm": inferred_rpm,
        "torque_nm": inferred_torque_nm,
        "detected_f_e_rough": f_e_rough,
        "torque_measured": torque_measured,
    }


def bearing_fault_frequencies(shaft_hz: float, ratios=BEARING_FAULT_RATIOS) -> dict:
    return {name: ratio * shaft_hz for name, ratio in ratios.items()}


def targeted_fault_magnitude(freqs, magnitude, fault_hz, window_hz=TARGETED_FAULT_WINDOW_HZ):
    mask = (freqs >= fault_hz - window_hz) & (freqs <= fault_hz + window_hz)
    return float(magnitude[mask].max()) if mask.any() else 0.0


# Broken rotor bar: classic signature is a "twice-slip-frequency" sideband at
# f_e * (1 +/- 2*k*s) in the RAW spectrum -- which, after envelope demodulation
# (the same trick used for bearings), collapses to a single peak at 2*k*s*f_e
# directly, since that's the separation between the carrier and its sidebands.
# Only k=1 for a first pass. This target frequency is tiny (~0.13Hz at our scoped
# condition) -- far too low to resolve in a 0.5s window (native resolution ~2Hz,
# smaller than one bin), so unlike BPFO/BPFI/BSF this has to be computed once from
# the FULL ~90s recording, not per-window, then replicated as a per-file constant.
ROTOR_BAR_SEARCH_WINDOW_HZ = 0.1  # generous relative to the target's own tiny scale,
# to tolerate slip-estimate imprecision, but still far from DC


def estimate_slip(f_e: float, nominal_rpm: float) -> float:
    """Slip = (synchronous - actual) / synchronous. For this 2-pole motor, synchronous
    mechanical frequency in Hz equals f_e directly (poles=2 means pole-pairs=1, so
    n_sync = 60*f_e -> f_sync(Hz) = f_e). Nominal rpm from the filename stands in for
    actual mechanical speed -- an approximation, not a direct measurement, since the
    keyphase channel itself was too noisy to reliably derive true RPM from earlier."""
    f_mech_nominal = nominal_rpm / 60.0
    return (f_e - f_mech_nominal) / f_e


def rotor_bar_frequency(f_e: float, slip: float, k: int = 1) -> float:
    return 2 * k * slip * f_e


def rotor_bar_magnitude_for_file(df, f_e: float, nominal_rpm: float) -> dict:
    """Computed once from the FULL recording per phase -- see module docstring on why
    this can't be a per-window feature the way BPFO/BPFI/BSF are."""
    slip = estimate_slip(f_e, nominal_rpm)
    target_hz = rotor_bar_frequency(f_e, slip, k=1)
    result = {}
    for ch in CURRENT_CHANNELS:
        env = envelope_signal(df[ch].values)
        freqs, magnitude = compute_fft(env)
        result[ch] = targeted_fault_magnitude(freqs, magnitude, target_hz, window_hz=ROTOR_BAR_SEARCH_WINDOW_HZ)
    return result


# Stator winding faults: classic MCSA signature is elevated NEGATIVE-SEQUENCE current --
# unlike BPFO/BPFI/BSF/rotor-bar, this isn't about a modulation frequency at all. It's
# a cross-phase quantity: decompose the three phase currents' complex value AT the
# fundamental frequency itself (not the envelope) using the standard symmetrical-
# component transform. A healthy, balanced three-phase system has ~zero negative-
# sequence current; any three-phase imbalance raises it. That's also the known caveat
# going in: an unbalanced SUPPLY voltage produces the same signature as a winding short,
# so this feature may not distinguish winding_H from voltage_unbalance_L -- to be
# checked empirically, not assumed.
#
# First implementation used a fixed "a"/"a^2" assignment for positive vs negative
# sequence and got nonsense: healthy files showed a LARGE "negative sequence" while
# voltage_unbalance_L showed a small one. Root cause, confirmed by comparing the phase
# angle of current_b relative to current_a across files: which physical channel leads
# by 120 degrees vs lags by 120 degrees is NOT consistent across recordings in this
# dataset (health/winding/bearing_ball/broken_bar go one way, voltage_unbalance/
# bearing_outer/bearing_inner go the other) -- an inconsistent phase-labeling/wiring
# convention across recording sessions, not a physical finding. Fixed by computing both
# candidate sequence assignments per file and always taking the SMALLER one as "negative
# sequence" -- for a real motor, most of the current is always in whichever sequence is
# actually dominant, so the smaller of the two is the rotation-invariant imbalance
# measure regardless of which way this particular file happens to be wired.
NEGATIVE_SEQUENCE_SEARCH_WINDOW_HZ = 1.0  # around f_e, for the fundamental's own phasor
PHASE_ROTATION = np.exp(1j * 2 * np.pi / 3)  # the "a" operator in symmetrical components


def phasor_at_frequency(signal, fs, target_hz, window_hz=NEGATIVE_SEQUENCE_SEARCH_WINDOW_HZ):
    """Like targeted_fault_magnitude, but keeps the complex value (magnitude AND phase)
    -- symmetrical components need phase, not just magnitude."""
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mask = (freqs >= target_hz - window_hz) & (freqs <= target_hz + window_hz)
    if not mask.any():
        return 0j
    band_vals = fft_vals[mask]
    peak_idx = np.argmax(np.abs(band_vals))
    return band_vals[peak_idx] / n


def negative_sequence_magnitude_for_file(df, f_e: float) -> float:
    """Computed once from the FULL recording, combining all three phases into one
    number -- unlike the other targeted features, this is inherently a cross-phase
    quantity, not something computed independently per phase. See module note above
    on why this takes the smaller of the two candidate sequence assignments."""
    phasors = {ch: phasor_at_frequency(df[ch].values, FS, f_e) for ch in CURRENT_CHANNELS}
    a = PHASE_ROTATION
    seq_1 = (phasors["current_a"] + a * phasors["current_b"] + a ** 2 * phasors["current_c"]) / 3
    seq_2 = (phasors["current_a"] + a ** 2 * phasors["current_b"] + a * phasors["current_c"]) / 3
    return float(min(np.abs(seq_1), np.abs(seq_2)))


# Eccentricity: a genuinely uneven air gap modulates the stator current at multiples of
# the mechanical rotation frequency. No rotor-slot-count-based formula was attempted here
# (the standard rigorous approach -- rotor slot harmonics -- needs the rotor's slot count,
# which isn't available from this dataset or its paper); instead this uses the simpler,
# well-established generic signature: an envelope-spectrum peak at 2x rotation frequency.
# Checked empirically (raw sanity check, before wiring in) across every fault type: this
# cleanly separates static_eccentricity_H/L (and their bearing-combo variants) from every
# mechanical fault and from healthy. It does NOT separate dynamic_eccentricity at all --
# scanned 1x through 5x harmonics and dynamic_eccentricity tracked healthy closely at every
# one, so no formula is claimed for it and it has no targeted feature -- unlike broken_bar,
# which since gained one (rotor_bar_frequency, the twice-slip sideband) and now validates
# at F1 0.65. Dynamic eccentricity rests on the wide envelope band alone, and scores
# accordingly: F1 0.15.
# Also note: voltage_unbalance_L shows an even LARGER peak here than real eccentricity --
# expected, since it's a broad disturbance that inflates nearly everything -- so this alone
# doesn't discriminate eccentricity from voltage imbalance. That separation is left to the
# supervised per-location classifier in presence_mcc5.py, which sees every feature at once
# rather than thresholding this one.
STATIC_ECCENTRICITY_HARMONIC = 2  # multiple of shaft frequency


def static_eccentricity_frequency(shaft_hz: float, harmonic: int = STATIC_ECCENTRICITY_HARMONIC) -> float:
    return harmonic * shaft_hz


def envelope_signal(signal):
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    return envelope - envelope.mean()


def envelope_spectrum_bins(freqs, magnitude, band=ENVELOPE_BAND, bin_group=BIN_GROUP):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    values = magnitude[mask]
    n_groups = len(values) // bin_group
    trimmed = values[: n_groups * bin_group]
    # max, not mean: a real fault signature is a narrow spike, and averaging it with
    # neighboring quiet bins divides its visible height by ~bin_group, diluting exactly
    # the signal we're trying to detect. Max preserves peak height under the same
    # dimensionality reduction.
    return trimmed.reshape(n_groups, bin_group).max(axis=1)


def envelope_stats(envelope) -> dict:
    return {
        "rms": float(np.sqrt(np.mean(envelope ** 2))),
        "peak": float(np.max(np.abs(envelope))),
        "kurtosis": float(kurtosis(envelope)),
    }


def windows_for_file(csv_path: Path, torque_nm: int = None, rpm: int = None, df=None) -> list:
    """torque_nm/rpm default to parsing the filename (used everywhere in this project
    to build our OWN labeled training/validation sets, where that's legitimate ground
    truth). Pass them explicitly -- e.g. from detect_condition_from_data() via
    windows_for_file_blind() -- to diagnose a real, unlabeled recording instead.

    df: pass an already-loaded/validated DataFrame (e.g. from upload_validation_mcc5) to
    skip re-loading via load_recording(), which only understands the raw dataset's fixed
    9-column positional format -- a real upload validated with a header/name-matched
    schema needs its already-resolved DataFrame used as-is, not reloaded from scratch."""
    if df is None:
        df = load_recording(csv_path)
    if torque_nm is None or rpm is None:
        meta = parse_filename(csv_path)
        torque_nm = meta["torque_nm"] if torque_nm is None else torque_nm
        rpm = meta["rpm"] if rpm is None else rpm

    shaft_hz = shaft_hz_for_file(df, rpm)
    fault_freqs = bearing_fault_frequencies(shaft_hz)
    rotor_bar_mag = rotor_bar_magnitude_for_file(df, shaft_hz, rpm)
    neg_seq_mag = negative_sequence_magnitude_for_file(df, shaft_hz)
    eccentricity_hz = static_eccentricity_frequency(shaft_hz)

    windows_per_channel = {ch: window_signal(df[ch].values) for ch in CURRENT_CHANNELS}
    torque_measured = float(df["torque"].mean())

    n_windows = len(windows_per_channel[CURRENT_CHANNELS[0]])
    rows = []
    for i in range(n_windows):
        # Skip degenerate windows (raw signal is exactly flat -- a trailing
        # data-acquisition artifact seen at the very end of one recording).
        # Kurtosis of a zero-variance signal is mathematically undefined (NaN).
        if any(windows_per_channel[ch][i].std() < 1e-9 for ch in CURRENT_CHANNELS):
            continue

        row = []
        for ch in CURRENT_CHANNELS:
            w = windows_per_channel[ch][i]
            env = envelope_signal(w)
            freqs, magnitude = compute_fft(env)

            row.extend(envelope_spectrum_bins(freqs, magnitude))

            stats = envelope_stats(env)
            row.extend([stats["rms"], stats["peak"], stats["kurtosis"]])

            for fault_hz in fault_freqs.values():
                row.append(targeted_fault_magnitude(freqs, magnitude, fault_hz))

            row.append(rotor_bar_mag[ch])  # per-file constant, not per-window (see above)
            row.append(targeted_fault_magnitude(freqs, magnitude, eccentricity_hz))
        row.append(neg_seq_mag)  # cross-phase, per-file constant -- one value, not per-channel
        row.extend([torque_nm, rpm, torque_measured, shaft_hz])
        rows.append(row)
    return rows


def windows_for_file_blind(csv_path: Path, df=None) -> list:
    """Real-inference entrypoint: don't parse the filename for anything. Auto-detects
    torque/RPM from the recording's own data (detect_condition_from_data), then reuses
    the exact same feature pipeline every labeled training file goes through. Pass an
    already-loaded/validated df (see upload_validation_mcc5) to avoid re-loading via
    load_recording(), which only understands the raw dataset's fixed column layout."""
    if df is None:
        df = load_recording(csv_path)
    condition = detect_condition_from_data(df)
    return windows_for_file(csv_path, torque_nm=condition["torque_nm"], rpm=condition["rpm"], df=df)


# Bounded so the payload cannot grow with recording length: a 90 s file gives 359 windows,
# and anything beyond this is averaged down in time rather than stored column by column.
SPECTROGRAM_MAX_COLUMNS = 200
SPECTROGRAM_FLOOR_DB = -60.0


def display_spectrum(df, nominal_rpm: float, band=ENVELOPE_BAND) -> dict:
    """The envelope spectrum behind a diagnosis, for showing rather than for modelling.

    Nothing in the pipeline consumes this. It exists so the app can show what the targeted
    features are sampled from: the same envelope spectrum, with the fault frequencies
    computed from measured shaft speed and the bearing geometry marked on it.

    Returns both views of the same per-window transform:

      * the mean spectrum -- one curve, averaged over every window and phase
      * a spectrogram -- the same data before that averaging, so time structure survives.
        This matters for the circulation splits, where speed or torque varies during the
        recording and the whole spectrum shifts with it; averaging hides exactly that.

    Two deliberate differences from the feature path. Native 2 Hz resolution rather than
    the BIN_GROUP-pooled ~8 Hz the classifier sees, because at that width a peak and its
    marker fall in the same bin. And phases averaged together, where the features keep them
    separate, because one curve is what a reader can interpret.

    Magnitudes are normalised to the recording's own maximum: absolute scale depends on
    motor current, which would make the axis meaningless across recordings.
    """
    shaft_hz = shaft_hz_for_file(df, nominal_rpm)
    fault_freqs = bearing_fault_frequencies(shaft_hz)
    slip = estimate_slip(shaft_hz, nominal_rpm)

    # One row per window, averaged across the three phases as it is built.
    per_window, freqs_kept = [], None
    windows = {ch: window_signal(df[ch].values) for ch in CURRENT_CHANNELS}
    n_windows = min(len(w) for w in windows.values())

    for i in range(n_windows):
        acc, used = None, 0
        for ch in CURRENT_CHANNELS:
            w = windows[ch][i]
            if w.std() < 1e-9:      # same degenerate-window guard the feature path uses
                continue
            freqs, magnitude = compute_fft(envelope_signal(w))
            keep = (freqs >= band[0]) & (freqs <= band[1])
            if freqs_kept is None:
                freqs_kept = freqs[keep]
            acc = magnitude[keep] if acc is None else acc + magnitude[keep]
            used += 1
        if used:
            per_window.append(acc / used)

    if not per_window:
        return None

    matrix = np.array(per_window)                 # (windows, freq bins)
    peak = float(matrix.max()) or 1.0
    mean = matrix.mean(axis=0)
    resolution = float(freqs_kept[1] - freqs_kept[0]) if len(freqs_kept) > 1 else 0.0

    # location keys match LOCATION_LABELS in the app, so it can highlight the markers whose
    # fault was actually reported and mute the rest.
    markers = [
        {"label": "Shaft", "hz": round(float(shaft_hz), 2), "location": None},
        {"label": "BPFO", "hz": round(float(fault_freqs["bpfo"]), 2), "location": "bearing_outer"},
        {"label": "BPFI", "hz": round(float(fault_freqs["bpfi"]), 2), "location": "bearing_inner"},
        {"label": "BSF", "hz": round(float(fault_freqs["bsf"]), 2), "location": "bearing_ball"},
        {"label": "2x shaft", "hz": round(float(static_eccentricity_frequency(shaft_hz)), 2),
         "location": "static_eccentricity"},
        {"label": "2sf", "hz": round(float(rotor_bar_frequency(shaft_hz, slip)), 2),
         "location": "rotor_bar"},
    ]
    # Markers below one bin cannot be told apart from DC, so they are dropped rather than
    # drawn on the axis where they would read as a rendering fault. In practice this is the
    # rotor-bar sideband: 2sf_e depends on slip, and slip here comes from the LABELLED rpm
    # rather than a measured speed, so it lands near zero (~0.09 Hz at 2000 rpm). The
    # feature path samples it regardless; this only governs what is worth plotting.
    visible = [m for m in markers
               if band[0] <= m["hz"] <= band[1] and m["hz"] >= resolution]

    return {
        "freq_hz": [round(float(f), 1) for f in freqs_kept],
        "magnitude": [round(float(m / peak), 4) for m in mean],
        "resolution_hz": round(resolution, 2) if resolution else None,
        "windows_averaged": len(per_window) * len(CURRENT_CHANNELS),
        "markers": visible,
        "spectrogram": _spectrogram_payload(matrix, peak),
    }


def _spectrogram_payload(matrix, peak: float) -> dict:
    """Time-frequency matrix as base64 bytes rather than a JSON array of numbers.

    36,000 floats would be ~250 KB of JSON text stored on every analysis row and sent to
    the browser again on every view. Quantising dB to a byte and base64-ing the result is
    ~48 KB for the same picture, and a spectrogram is a picture -- 256 levels is more than
    the eye resolves in a colour ramp.
    """
    import base64

    n_windows = matrix.shape[0]
    if n_windows > SPECTROGRAM_MAX_COLUMNS:
        # Average adjacent windows down rather than dropping them, so a transient is dimmed
        # instead of disappearing entirely depending on where it fell.
        group = int(np.ceil(n_windows / SPECTROGRAM_MAX_COLUMNS))
        trimmed = matrix[:(n_windows // group) * group]
        matrix = trimmed.reshape(-1, group, matrix.shape[1]).mean(axis=1)

    db = 20 * np.log10(np.maximum(matrix / peak, 1e-6))
    clipped = np.clip(db, SPECTROGRAM_FLOOR_DB, 0.0)
    levels = ((clipped - SPECTROGRAM_FLOOR_DB) / -SPECTROGRAM_FLOOR_DB * 255).astype(np.uint8)

    return {
        "columns": int(levels.shape[0]),
        "rows": int(levels.shape[1]),
        "floor_db": SPECTROGRAM_FLOOR_DB,
        "seconds": round(n_windows * STRIDE_SAMPLES / FS, 1),
        # Row-major: one column of the image (a time slice) at a time, low frequency first.
        "levels_b64": base64.b64encode(levels.tobytes()).decode("ascii"),
    }
