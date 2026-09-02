"""
Loader for the MCC5-THU multi-fault induction motor benchmark dataset.
Single physical 2.2kW motor, every fault type on the same rig. What lets us build
one unified detector across fault families instead of separate profiles.

Covers both splits: "speed_circulation" and "torque_circulation".
Despite the names, both splits' files are steady-state at the condition in their filename,
and the same fault+condition combo often exists in both, giving us two independent
recordings per condition instead of one.
"""
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR.parent.parent / "dataset"
SPLIT_DIRS = {
    "speed_circulation": DATASET_ROOT / "MCC5-THU Motor_speed_circulation",
    "torque_circulation": DATASET_ROOT / "MCC5-THU Motor_torque_circulation",
}

FS = 12800 
COLUMNS = ["time", "speed", "torque", "vib_x", "vib_y", "vib_z", "current_a", "current_b", "current_c"]

REQUIRED_COLUMNS = ["current_a", "current_b", "current_c", "torque"]

FILENAME_RE = re.compile(
    r"^(?P<fault>.+)_(?P<split>speed_circulation|torque_circulation)_"
    r"(?P<torque>\d+)Nm_(?P<rpm>\d+)rpm(?:_\d+d?)?$"
)


def parse_filename(csv_path: Path) -> dict:
    match = FILENAME_RE.match(csv_path.stem)
    if not match:
        raise ValueError(f"Filename doesn't match expected pattern: {csv_path.name}")
    return {
        "fault": match.group("fault"),
        "split": match.group("split"),
        "torque_nm": int(match.group("torque")),
        "rpm": int(match.group("rpm")),
    }

EXCLUDED_FILES = {"bearing_outer_H_torque_circulation_20Nm_3000rpm_250821175544.csv"}


def list_files(fault: str = None, torque_nm: int = None, rpm: int = None, split: str = None) -> list:
    dirs = SPLIT_DIRS.values() if split is None else [SPLIT_DIRS[split]]
    files = sorted(f for d in dirs for f in d.glob("*.csv"))
    result = []
    for f in files:
        if f.name in EXCLUDED_FILES:
            continue
        meta = parse_filename(f)
        if fault is not None and meta["fault"] != fault:
            continue
        if torque_nm is not None and meta["torque_nm"] != torque_nm:
            continue
        if rpm is not None and meta["rpm"] != rpm:
            continue
        result.append(f)
    return result


def list_fault_types() -> list:
    faults = {parse_filename(f)["fault"] for d in SPLIT_DIRS.values() for f in d.glob("*.csv")}
    return sorted(faults)


def load_recording(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, header=None, names=COLUMNS)


if __name__ == "__main__":
    files = list_files()
    print(f"{len(files)} files across both splits")
    print(f"{len(list_fault_types())} fault types: {list_fault_types()}")

    files_at_condition = list_files(fault="health", torque_nm=20, rpm=1000)
    print(f"\nhealthy files at 20Nm/1000rpm: {len(files_at_condition)}")
    for f in files_at_condition:
        print(f"  {f.parent.name}/{f.name}")
