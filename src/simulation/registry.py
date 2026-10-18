"""THE scenario registry: one definition of every scenario, used by run.py, run_all / run_scenarios, generate_days,
the plan search and the dashboard. Standard library only.

Surge sizes are never typed here: every scenario is built from the cited components in scenarios.SURGE_RANGES
(heatwave +6% to +13%, hurricane +9% to +13%). The burst is the UNSOURCED stress test in scenarios.BURST_*.
For normal / heatwave / hurricane / hurricane_burst, generate() returns EXACTLY the same day as
scenarios.generate(name, seed, burst=...). compound = heatwave x hurricane: ASSUMPTION, effects multiply (independence).
With ED_USE_LEGACY_PARAMS=1 every key falls back to the original hand-typed profiles (including the old +35% compound).
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace
from typing import Optional, Union

from src.simulation import scenarios as S

COMPOUND_ASSUMPTION = "ASSUMPTION: effects multiply (independence)"
_BURST_NOTE = ("UNSOURCED stress test: 10-20 very urgent (level 1-2) patients arrive within one hour between 12:00 and 21:00; "
               "the hour is unknown in advance.")


def _src(c: str) -> str:
    return S.SURGE_RANGES[c]["source"] + " (" + S.SURGE_RANGES[c]["evidence_status"].replace("_", " ") + ")"


REGISTRY = {
    "normal": {"label": "Normal Monday", "components": (), "burst": False,
               "description": "Ordinary Monday. Arrival-by-hour shape and triage mix: NHAMCS 2022 (DATA). Volume 200/day: UNSOURCED."},
    "heatwave": {"label": "Heatwave", "components": ("heatwave",), "burst": False,
                 "description": ("All-cause visits +6% to +13%, known the day before. Source: " + _src("heatwave")
                                 + ". The extra visits are placed 12:00-20:00 (UNSOURCED window). The older-patient shift is not modelled here.")},
    "hurricane": {"label": "Hurricane (receiving hospital)", "components": ("hurricane",), "burst": False,
                  "description": "Sustained +9% to +13% all day, known the day before. Source: " + _src("hurricane") + "."},
    "hurricane_burst": {"label": "Hurricane + burst of very urgent patients", "components": ("hurricane",), "burst": True,
                        "description": "Hurricane day as above plus a burst. " + _BURST_NOTE},
    "compound": {"label": "Compound: heatwave and hurricane together", "components": ("heatwave", "hurricane"), "burst": False,
                 "description": ("Heatwave and hurricane components on the same day, each with its cited range. "
                                 + COMPOUND_ASSUMPTION + ". Not a historical event; a stress test.")},
}


def names() -> list:
    return list(REGISTRY)


def _check(key: str) -> dict:
    if key not in REGISTRY:
        raise ValueError(f"unknown scenario {key!r}; known: {names()}")
    return REGISTRY[key]


def label(key: str) -> str:
    return _check(key)["label"]


def description(key: str) -> str:
    return _check(key)["description"]


def info(key: str) -> dict:
    """What the hospital knows in advance. A pure function of the scenario key (never the seed or sampled sizes)."""
    r = _check(key)
    comps = r["components"]
    windows = [S.SURGE_RANGES[c]["window"] for c in comps]
    window = (min(w[0] for w in windows), max(w[1] for w in windows)) if windows else None
    return {"scenario": key, "surge_known": bool(comps), "known_from": "day_before" if comps else "n/a", "window": window,
            "burst_possible": r["burst"] or "hurricane" in comps, "components": list(comps)}


def hourly_factors(key: str, sizes: Optional[dict] = None, hourly: Optional[list] = None) -> list:
    """24 factors = product over components of hourly_multipliers(component, size). sizes default to the midpoints."""
    f = [1.0] * 24
    for c in _check(key)["components"]:
        m = (sizes or {}).get(c, S.SURGE_RANGES[c]["midpoint"])
        f = [a * b for a, b in zip(f, S.hourly_multipliers(c, m, hourly=hourly))]
    return f


def forecast(inf: dict, start_hour: int, hours: int) -> list:
    """Expected arrivals per hour from ``info`` ONLY (midpoints of the cited ranges; a burst is never forecast)."""
    for k in ("scenario", "surge_known", "window"):
        if k not in inf:
            raise ValueError(f"info is missing {k!r}")
    f = hourly_factors(inf["scenario"]) if inf["surge_known"] else [1.0] * 24
    return [S.BASE_VOLUME * S.HOURLY[(start_hour + i) % 24] * f[(start_hour + i) % 24] for i in range(hours)]


def _sizes_from(key: str, seed, intensity) -> dict:
    """Surge multiplier per component: sampled from the cited range (like generate()) or fixed by the user."""
    comps = _check(key)["components"]
    if intensity is not None:
        given = intensity if isinstance(intensity, dict) else {comps[0]: intensity} if len(comps) == 1 else None
        if given is None:
            raise ValueError(f"{key} has {len(comps)} components; pass intensity as {{component: extra share}}")
        out = {}
        for c in comps:
            r = S.SURGE_RANGES[c]
            m = 1.0 + float(given[c]) if c in given else S._multiplier_from_u(c, random.Random(f"{seed}:volume:{c}").random())
            if not r["low"] - 1e-9 <= m <= r["high"] + 1e-9:
                raise ValueError(f"{c} surge must stay between +{(r['low'] - 1) * 100:.0f}% and +{(r['high'] - 1) * 100:.0f}% (published range)")
            out[c] = m
        return out
    u = random.Random(f"{seed}:volume").random()      # the SAME first draw scenarios.generate() makes
    return {c: S._multiplier_from_u(c, u if i == 0 else random.Random(f"{seed}:volume:{c}").random()) for i, c in enumerate(comps)}


def generate(key: str, seed, burst: bool = False, intensity: Union[None, float, dict] = None):
    """The TRUE day: (list of Arrival, info). ``burst`` adds the stress-test burst on top of the key's own setting."""
    r = _check(key)
    use_burst = bool(burst or r["burst"])
    if S.USE_LEGACY_PARAMS:
        base = "hurricane" if key == "hurricane_burst" else key
        return S.generate(base, seed, burst=use_burst)[0], info(key)
    if intensity is None and len(r["components"]) <= 1:
        base = r["components"][0] if r["components"] else "normal"
        return S.generate(base, seed, burst=use_burst)[0], info(key)        # identical to the old generator
    sizes = _sizes_from(key, seed, intensity)
    sc = S._CURRENT_SCENARIOS["normal"]
    factors = hourly_factors(key, sizes, hourly=S._CURRENT_HOURLY)
    volume = S.BASE_VOLUME * S.weekday_factor("monday")
    vs = random.Random(f"{seed}:volume")
    vs.random()                                        # keep the hourly counts on the same stream position as generate()
    rows = []
    for h in range(24):
        count = S._poisson_from_u(vs.random(), volume * S._CURRENT_HOURLY[h] * factors[h])
        rows += S._hour_rows(seed, h, count, sc.acuity, sc.admit, sc.boarding_mult)
    if use_burst:
        rows += S._burst_rows(seed)
    return S._to_arrivals(rows), info(key)


def expected_daily(key: str) -> float:
    return S.BASE_VOLUME * sum(S.HOURLY[h] * hourly_factors(key)[h] for h in range(24))


def as_scenario(key: str) -> S.Scenario:
    """A Scenario object (expected hourly factors at the midpoints) for code that needs one, e.g. the briefing."""
    if S.USE_LEGACY_PARAMS:
        return S.SCENARIOS["hurricane" if key == "hurricane_burst" else key]
    return S.Scenario(key, label(key), description(key), acuity=S.BASE_ACUITY, hourly_mult=tuple(hourly_factors(key)))


def api() -> SimpleNamespace:
    """The interface the optimiser's plan_search expects, backed by this registry."""
    return SimpleNamespace(generate=generate, forecast=forecast, truth_seed=S.truth_seed, lookahead_seed=S.lookahead_seed,
                           patients_from_rates=S.patients_from_rates, list_scenarios=names, scenario_info=info)


def forecast_patients(key: str, seed, t0: float, t1: float, base_pid: int = 10 ** 6) -> list:
    """Forecast patients for [t0, t1) built from info only (whole hours, then cut to the window)."""
    h0, h1 = int(t0 // 60), int(math.ceil(t1 / 60))
    pats = S.patients_from_rates(forecast(info(key), h0, h1 - h0), seed, start_hour=h0, base_pid=base_pid)
    return [p for p in pats if t0 <= p.t < t1]
