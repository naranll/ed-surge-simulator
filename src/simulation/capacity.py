"""ONE source of truth for hospital capacity: configs/capacity.json (written by scripts/calibrate.py, full mode).

Standard library only. The file holds the calibrated reference hospital (beds, doctors, triage nurses) with the
method and data it was fitted to, and the patient-volume shares used to scale the Small / Medium presets.
A different file can be used with the environment variable ED_CAPACITY_FILE.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

CAPACITY_PATH = Path(__file__).resolve().parents[2] / "configs" / "capacity.json"
_ENV = "ED_CAPACITY_FILE"
_CACHE: dict = {}


def path() -> Path:
    return Path(os.environ.get(_ENV) or CAPACITY_PATH)


def load() -> dict:
    p = path()
    key = str(p)
    if key not in _CACHE:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Create it with: python scripts/calibrate.py   (full mode, not --quick)")
        data = json.loads(p.read_text(encoding="utf-8"))
        ref = data.get("reference_hospital", {})
        for k in ("beds", "doctors", "triage_nurses"):
            if not isinstance(ref.get(k), int) or ref[k] < 1:
                raise ValueError(f"{p}: reference_hospital.{k} must be a whole number >= 1")
        _CACHE[key] = data
    return _CACHE[key]


def reference() -> dict:
    """{"beds", "doctors", "triage_nurses"} of the calibrated reference hospital."""
    ref = load()["reference_hospital"]
    return {k: ref[k] for k in ("beds", "doctors", "triage_nurses")}


def preset_specs() -> dict:
    """{name: {beds, doctors, triage_nurses, volume_scale}}. Smaller presets = reference x volume share, rounded UP."""
    data, ref = load(), reference()
    out = {}
    for name, share in data["preset_volume_shares"]["values"].items():
        if not 0.0 < share <= 1.0:
            raise ValueError(f"preset volume share for {name} must be in (0, 1]")
        out[name] = {k: max(1, math.ceil(v * share - 1e-9)) for k, v in ref.items()}
        out[name]["volume_scale"] = share
    return out
