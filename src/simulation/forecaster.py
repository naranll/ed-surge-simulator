"""Forecaster: expected arrivals per hour, built ONLY from what a hospital knows in advance.

Inputs it is allowed to use
  * the info dict from scenarios.generate()  (scenario name, surge_known, window)
  * the baseline hourly shape and BASE_VOLUME (a Monday figure)
  * the public midpoint of the published surge range (SURGE_RANGES) for that scenario

It never sees the true day, a seed, the burst flag or the surge size that the truth sampled.
It never includes a burst (the timing of a burst is unknown); info["burst_possible"] only
tells the optimiser to stay cautious. A scenario without a published surge range (for
example "compound", which is a hand-typed stress test) is forecast at the baseline.
This is a simple rule, not a trained model. Standard library only.
"""
from __future__ import annotations

from src.simulation.scenarios import BASE_VOLUME, HOURLY, SURGE_RANGES, hourly_multipliers

_REQUIRED_KEYS = ("scenario", "surge_known", "window")


def forecast(info: dict, start_hour: int, hours: int) -> list:
    """Expected arrivals for each of `hours` hours, starting at hour `start_hour`.

    The hour index wraps around every 24 hours. The result is deterministic: the same
    info always gives the same list.
    """
    if not isinstance(info, dict):
        raise ValueError("info must be the dict returned by scenarios.generate()")
    missing = [k for k in _REQUIRED_KEYS if k not in info]
    if missing:
        raise ValueError(f"info is missing {missing}")
    for label, value in (("start_hour", start_hour), ("hours", hours)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} must be a whole number, got {value!r}")
    if hours < 0:
        raise ValueError("hours must be >= 0")

    factors = [1.0] * 24
    name, window = info["scenario"], info["window"]
    if info["surge_known"] and name in SURGE_RANGES and window is not None:
        factors = hourly_multipliers(name, SURGE_RANGES[name]["midpoint"], window=tuple(window))

    out = []
    for i in range(hours):
        h = (start_hour + i) % 24
        out.append(BASE_VOLUME * HOURLY[h] * factors[h])
    return out
