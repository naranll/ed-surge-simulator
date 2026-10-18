"""Live demo for run.py (demo / rolling). Standard library only.

Replaces the old route through src/dashboard/app.py (which is left untouched). Differences from the old route:
  * capacity comes from configs/capacity.json (the calibrated reference hospital), not the 20-bed default;
  * patients get real-data admission and care times (models/patient_flow/admission_lookup.json);
  * the forecast patients used to test actions get the same treatment, so the test and the "live" day agree.
With ED_USE_LEGACY_PARAMS=1, run.py keeps using the old route unchanged.
"""
from __future__ import annotations

from typing import Iterator, Optional

from src.explainer.briefing import build_briefing
from src.optimization.optimizer import FORECAST_PID, SEEDS, Recommendation, recommend
from src.simulation.ed_simulator import EDSim
from src.simulation.hospital import HospitalConfig, PRESETS, build_sim, prepare
from src.simulation.scenarios import SCENARIOS, generate_arrivals

LIVE_NOISE = 0.08   # UNSOURCED: day-to-day noise on the hourly arrival rate (the old LiveED default, kept for comparability)


def reference_config() -> HospitalConfig:
    """The calibrated reference hospital from configs/capacity.json."""
    from src.simulation import capacity
    name = capacity.load()["reference_hospital"].get("preset_name", "Large")
    return PRESETS[name]


class LiveDay:
    """A true day advanced in real time, for the configured hospital."""

    def __init__(self, key: str, seed: int = 42, cfg: Optional[HospitalConfig] = None, noise: float = LIVE_NOISE):
        self.key, self.scenario = key, SCENARIOS[key]
        self.cfg = cfg or reference_config()
        self.arrivals = prepare(self.cfg, generate_arrivals(self.scenario, seed, 0.0, 1440.0, noise=noise))
        self.sim = build_sim(self.cfg, self.arrivals)

    @property
    def now(self) -> float:
        return self.sim.now

    def advance_to(self, t: float) -> None:
        self.sim.run_until(t)

    def snapshot(self) -> EDSim:
        return self.sim.snapshot(strip_future=True, prune=True)


def forecast_patients(live: LiveDay, seeds, hours: float = 4.0) -> dict:
    """{seed: forecast patients for the next ``hours``}, built from the scenario rates only (never the true day)."""
    t0, t1 = live.now, live.now + hours * 60.0
    return {s: prepare(live.cfg, generate_arrivals(live.scenario, s, t0, t1, base_pid=FORECAST_PID)) for s in seeds}


def recommend_now(live: LiveDay, seeds=SEEDS, workers: int = 1, hours: float = 4.0) -> Recommendation:
    return recommend(live.snapshot(), live.scenario, hours=hours, workers=workers,
                     forecasts=forecast_patients(live, seeds, hours))


def compute_view(key: str, minute: float, seed: int = 42, workers: int = 1) -> dict:
    """What run.py demo prints: the recommendation at ``minute`` and a plain briefing."""
    live = LiveDay(key, seed)
    live.advance_to(minute)
    rec = recommend_now(live, workers=workers)
    return {"scenario": live.scenario, "config": live.cfg, "rec": rec, "briefing": build_briefing(live.scenario, rec)}


def rolling_view(key: str, seed: int = 42, start: float = 0.0, until: float = 1440.0, step: float = 30.0,
                 workers: int = 1) -> Iterator[Recommendation]:
    """Every ``step`` minutes: advance the true day and recommend for the next 4 hours (actions are NOT applied)."""
    live = LiveDay(key, seed)
    t = start
    while t <= until:
        live.advance_to(t)
        yield recommend_now(live, workers=workers)
        t += step
