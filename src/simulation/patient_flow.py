"""Patient-flow link: give simulated patients admission and care time from REAL-data tables.

Reads models/patient_flow/admission_lookup.json (written by scripts/train_patient_flow.py from NHAMCS 2022).
Standard library only. Off by default: nothing calls this unless a HospitalConfig has patient_flow=True,
and the old behaviour stays available (legacy switch: ED_USE_LEGACY_PARAMS=1 turns it off everywhere).

For each patient (keyed by its own random stream, so the same patient always gets the same draw):
  1. an age band is drawn from the real age mix of that triage level,
  2. admission is drawn from the real admission rate for (triage level, age band),
  3. total care time is drawn from a log-normal fitted to LOV - WAITTIME for (level, admitted / discharged),
  4. bed stay after the doctor consult = care time - consult time (never below MIN_HOLD).
LOV - WAITTIME is a PROXY (includes tests and boarding), not pure service time.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Optional, Sequence

from src.simulation.scenarios import USE_LEGACY_PARAMS, Arrival

LOOKUP_PATH = Path(__file__).resolve().parents[2] / "models" / "patient_flow" / "admission_lookup.json"
MIN_HOLD = 5.0          # ASSUMPTION: a bed is never held less than 5 minutes after the consult
MAX_CARE = 1440.0       # ASSUMPTION: total care time is capped at 24 hours
_CACHE: dict = {}


def available(path: Optional[Path] = None) -> bool:
    return (Path(path) if path else LOOKUP_PATH).exists() and not USE_LEGACY_PARAMS


def load_lookup(path: Optional[Path] = None) -> dict:
    p = Path(path) if path else Path(os.environ.get("ED_PATIENT_FLOW_LOOKUP", LOOKUP_PATH))
    key = str(p)
    if key not in _CACHE:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run scripts/train_patient_flow.py first")
        _CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
    return _CACHE[key]


def _pick(rng: random.Random, shares: dict) -> str:
    u, acc, last = rng.random(), 0.0, None
    for band, s in shares.items():
        acc += s
        last = band
        if u < acc:
            return band
    return last


def apply_patient_flow(arrivals: Sequence[Arrival], lookup: Optional[dict] = None) -> list:
    """Return new Arrival tuples with admit and hold re-drawn from the real tables."""
    lk = lookup or load_lookup()
    out = []
    for a in arrivals:
        rng = random.Random(f"flow:{a.pid}:{a.t:.4f}:{a.ats}")
        lvl = str(a.ats)
        band = _pick(rng, lk["age_share_given_triage"][lvl])
        admitted = rng.random() < lk["admission"][lvl]["bands"][band]["rate"]
        cell = lk["care_time_log"].get(f"{lvl}_{'admitted' if admitted else 'discharged'}")
        if cell is None:     # a cell with too few real records: fall back to the other cell of the same level
            cell = lk["care_time_log"][f"{lvl}_{'discharged' if admitted else 'admitted'}"]
        care = min(MAX_CARE, math.exp(rng.gauss(cell["mu"], cell["sigma"])))
        out.append(a._replace(admit=admitted, hold=max(MIN_HOLD, care - a.consult)))
    return out
