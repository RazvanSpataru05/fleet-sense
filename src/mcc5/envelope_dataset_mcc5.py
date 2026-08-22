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

Originally scoped to a single operating condition (20Nm, 1000rpm), for the same reason
Paderborn was scoped to one condition -- mixing conditions dilutes the anomaly signal.
Later extended to work across all 6 real conditions (2 torque levels x 3 RPMs) once RPM-
adaptive frequency search (see FUNDAMENTAL_SEARCH_MARGIN_HZ) made that safe -- see
classifier_mcc5.py, presence_mcc5.py, severity_mcc5.py, and anomaly_mcc5.py, which all
extract features across every condition rather than just one.
"""
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

from load_mcc5 import FS, list_files, load_recording, parse_filename

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
# Only 359 windows exist at this single condition (no repeated files like Paderborn);
# native resolution would give 101 bins/phase x 3 phases = 303 dims against ~287 training
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
KNOWN_TORQUE_LEVELS_NM = (20, 40)

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
# one, so no formula is claimed for it; it stays undiagnosable for now, same as broken_bar.
# Also note: voltage_unbalance_L shows an even LARGER peak here than real eccentricity --
# expected, since it's a broad disturbance that inflates nearly everything -- so this alone
# doesn't discriminate eccentricity from voltage imbalance; that's left to the per-family
# vs-generic-baseline gating in diagnose_mcc5.py, the same mechanism already relied on to
# keep voltage_unbalance_L from hijacking the neg_seq family.
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


def windows_for_file(csv_path: Path, torque_nm: int = None, rpm: int = None) -> list:
    """torque_nm/rpm default to parsing the filename (used everywhere in this project
    to build our OWN labeled training/validation sets, where that's legitimate ground
    truth). Pass them explicitly -- e.g. from detect_condition_from_data() via
    windows_for_file_blind() -- to diagnose a real, unlabeled recording instead."""
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


def windows_for_file_blind(csv_path: Path) -> list:
    """Real-inference entrypoint: don't parse the filename for anything. Auto-detects
    torque/RPM from the recording's own data (detect_condition_from_data), then reuses
    the exact same feature pipeline every labeled training file goes through."""
    df = load_recording(csv_path)
    condition = detect_condition_from_data(df)
    return windows_for_file(csv_path, torque_nm=condition["torque_nm"], rpm=condition["rpm"])


def build_matrix(files) -> np.ndarray:
    rows = []
    for f in files:
        rows.extend(windows_for_file(f))
    return np.array(rows)


