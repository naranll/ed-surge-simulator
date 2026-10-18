"""Early-warning risk for the next 4 hours. predict_risk(snapshot, info) -> dict.

The model is trained on SIMULATED days (scripts/train_early_warning.py). It is a fast early-warning
surrogate for the simulator, not validated on a real hospital. Features are only what a hospital can see
at time t (queues, waits, beds, staff, hour) plus the forecaster's expected arrivals built from ``info``
(never the true day). Standard library for logistic models; xgboost only if the saved model needs it.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = Path(__file__).resolve().parent / "early_warning"
TARGETS = ("urgent_late", "beds_full", "lwbs")
PRETTY = {"urgent_late": "urgent patients seen late", "beds_full": "beds full (above 95%)",
          "lwbs": "patients leaving without being seen"}
HORIZON_MIN = 240.0

# Every feature a hospital can know at time t. Nothing here comes from the true day or from the future.
FEATURES = (
    [f"queue_l{l}" for l in range(1, 6)] + [f"longest_wait_l{l}" for l in range(1, 6)] +
    ["queue_total", "bed_occupancy", "boarders", "doctors_busy_frac", "nurses_busy_frac",
     "staff_doctors", "staff_nurses", "beds_total", "hour_of_day", "minutes_since_start", "arrivals_last_hour",
     "forecast_next4h", "forecast_ratio", "surge_known", "burst_possible"])


def forecast_next4h(info: dict, now_min: float, volume_scale: float = 1.0) -> tuple:
    """(expected arrivals in the next 4 h, ratio to the normal-day expectation). Built from ``info`` only."""
    from src.simulation.registry import forecast      # the one scenario registry (same as the old forecaster for its scenarios)
    from src.simulation.scenarios import BASE_VOLUME, HOURLY
    h0, frac = int(now_min // 60) % 24, (now_min % 60) / 60.0
    f = forecast(info, h0, 6)
    w = [1 - frac, 1, 1, 1, frac]
    exp = volume_scale * sum(f[i] * w[i] for i in range(5))
    base = volume_scale * BASE_VOLUME * sum(HOURLY[(h0 + i) % 24] * w[i] for i in range(5))
    return exp, (exp / base if base > 0 else 1.0)


def build_features(sim, info: dict, volume_scale: float = 1.0) -> dict:
    """Feature dict from a live EDSim (or a snapshot of one) and the advance info."""
    now = sim.now
    q = {l: [] for l in range(1, 6)}
    boarders = last_hour = 0
    for p in sim.pat.values():
        if p.st == "future":
            continue
        if now - 60.0 < p.t <= now:
            last_hour += 1
        if p.st in ("tri_q", "bed_q", "doc_q"):
            q[p.ats].append(now - p.t)
        elif p.st == "board" and p.admit:
            boarders += 1
    f = {}
    for l in range(1, 6):
        f[f"queue_l{l}"] = len(q[l])
        f[f"longest_wait_l{l}"] = max(q[l], default=0.0)
    f["queue_total"] = sum(len(v) for v in q.values())
    f["bed_occupancy"] = sim.busy["b"] / max(sim.cap["b"], 1)
    f["boarders"] = boarders
    f["doctors_busy_frac"] = sim.busy["d"] / max(sim.cap["d"], 1)
    f["nurses_busy_frac"] = sim.busy["n"] / max(sim.cap["n"], 1)
    f["staff_doctors"], f["staff_nurses"], f["beds_total"] = sim.cap["d"], sim.cap["n"], sim.cap["b"]
    f["hour_of_day"] = (now / 60.0) % 24
    f["minutes_since_start"] = now
    f["arrivals_last_hour"] = last_hour
    f["forecast_next4h"], f["forecast_ratio"] = forecast_next4h(info, now, volume_scale)
    f["surge_known"] = 1.0 if info.get("surge_known") else 0.0
    f["burst_possible"] = 1.0 if info.get("burst_possible") else 0.0
    return f


# ------------------------------------------------------------------ loading
_LOADED: dict = {}


def _load(model_dir: Path) -> Optional[dict]:
    key = str(model_dir)
    if key in _LOADED:
        return _LOADED[key]
    meta_path = model_dir / "early_warning_meta.json"
    if not meta_path.exists():
        _LOADED[key] = None
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    models = {}
    for t in TARGETS:
        spec = meta["models"][t]
        path = model_dir / spec["file"]
        if spec["kind"] == "linear":
            models[t] = ("linear", json.loads(path.read_text(encoding="utf-8")))
        else:
            import xgboost as xgb      # ImportError is handled by predict_risk
            b = xgb.Booster()
            b.load_model(str(path))
            models[t] = ("xgb", b)
    _LOADED[key] = {"meta": meta, "models": models}
    return _LOADED[key]


def _prob(kind, model, row: list) -> float:
    if kind == "linear":
        z = model["intercept"]
        for x, m, s, c in zip(row, model["mean"], model["scale"], model["coef"]):
            z += c * ((x - m) / (s if s else 1.0))
        return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
    import numpy as np
    import xgboost as xgb
    return float(model.predict(xgb.DMatrix(np.array([row], dtype=float), feature_names=list(FEATURES)))[0])


def predict_risk(snapshot, info: dict, volume_scale: float = 1.0, model_dir: Optional[Path] = None) -> dict:
    """Three risks for the next 4 hours, the resource likely to break first, and a plain-English line.

    Returns {"risk_urgent_late", "risk_beds_full", "risk_lwbs", "breaks_first", "message", "available"}.
    If the model files or a library are missing, ``available`` is False and the message says why.
    """
    empty = {"risk_urgent_late": None, "risk_beds_full": None, "risk_lwbs": None, "breaks_first": None, "available": False}
    try:
        bundle = _load(Path(model_dir) if model_dir else MODEL_DIR)
    except ImportError as err:
        return {**empty, "message": f"Early-warning model not available: a library is missing ({err})."}
    if bundle is None:
        return {**empty, "message": "Early-warning model not trained yet. Run: python scripts/train_early_warning.py"}
    feats = build_features(snapshot, info, volume_scale)
    row = [float(feats[k]) for k in FEATURES]
    p = {t: _prob(*bundle["models"][t], row) for t in TARGETS}
    thr = bundle["meta"].get("alarm_thresholds", {})
    # "breaks first" = the target whose risk is highest relative to its own alarm line (alarm line = 5% false alarms on normal days)
    ratio = {t: p[t] / max(thr.get(t, 0.5), 1e-6) for t in TARGETS}
    top = max(TARGETS, key=lambda t: ratio[t])
    alarm = ratio[top] >= 1.0
    msg = (f"Watch {PRETTY[top]}: chance {100 * p[top]:.0f}% in the next 4 hours (alarm line {100 * thr.get(top, 0.5):.0f}%). "
           if alarm else "No alarm: all three risks are below their alarm lines. ")
    msg += "Simulation-based early warning, not validated on a real hospital."
    return {"risk_urgent_late": p["urgent_late"], "risk_beds_full": p["beds_full"], "risk_lwbs": p["lwbs"],
            "breaks_first": PRETTY[top] if alarm else "nothing expected to break", "alarm": alarm,
            "message": msg, "available": True}
