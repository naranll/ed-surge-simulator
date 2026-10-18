"""Live demo for run.py (demo / rolling). Standard library only.

Replaces the old route through src/dashboard/app.py (which is left untouched). Differences from the old route:
  * capacity comes from configs/capacity.json (the calibrated reference hospital), not the 20-bed default;
  * the TRUE day comes from the scenario registry (registry.generate, seed = truth_seed(--seed)), the same
    generator run_all and the plan search use; there is no extra day-to-day noise (the old LiveED 8% noise is UNSOURCED);
  * the forecast patients used to test actions come from the advance information only (registry.forecast) with
    look-ahead seeds, which never overlap truth seeds;
  * patients get real-data admission and care times (models/patient_flow/admission_lookup.json).
With ED_USE_LEGACY_PARAMS=1, run.py keeps using the old route unchanged.
"""
from __future__ import annotations

from typing import Iterator, Optional

from src.explainer.briefing import build_briefing
from src.optimization.optimizer import FORECAST_PID, Recommendation, recommend
from src.simulation import registry
from src.simulation.ed_simulator import EDSim
from src.simulation.hospital import HospitalConfig, PRESETS, build_sim, prepare
from src.simulation.scenarios import lookahead_seed, truth_seed

DEFAULT_REPLICATIONS = 5     # forecast runs per action (step 6 makes this configurable)


def reference_config() -> HospitalConfig:
    """The calibrated reference hospital from configs/capacity.json."""
    from src.simulation import capacity
    return PRESETS[capacity.load()["reference_hospital"].get("preset_name", "Large")]


class LiveDay:
    """A true day advanced in real time, for the configured hospital."""

    def __init__(self, key: str, seed: int = 42, cfg: Optional[HospitalConfig] = None):
        self.key = key
        self.scenario = registry.as_scenario(key)
        self.info = registry.info(key)
        self.cfg = cfg or reference_config()
        truth, _ = registry.generate(key, truth_seed(seed))
        self.arrivals = prepare(self.cfg, truth)
        self.sim = build_sim(self.cfg, self.arrivals)

    @property
    def now(self) -> float:
        return self.sim.now

    def advance_to(self, t: float) -> None:
        self.sim.run_until(t)

    def snapshot(self) -> EDSim:
        return self.sim.snapshot(strip_future=True, prune=True)


def forecast_patients(live: LiveDay, replications: int, hours: float = 4.0) -> dict:
    """{look-ahead seed: forecast patients for the next ``hours``}, from the advance information only (never the true day)."""
    t0, t1 = live.now, live.now + hours * 60.0
    return {lookahead_seed(i): prepare(live.cfg, registry.forecast_patients(live.key, lookahead_seed(i), t0, t1, FORECAST_PID))
            for i in range(replications)}


def recommend_now(live: LiveDay, replications: int = DEFAULT_REPLICATIONS, workers: int = 1, hours: float = 4.0) -> Recommendation:
    return recommend(live.snapshot(), live.scenario, hours=hours, workers=workers,
                     forecasts=forecast_patients(live, replications, hours))


def compute_view(key: str, minute: float, seed: int = 42, workers: int = 1, replications: int = DEFAULT_REPLICATIONS) -> dict:
    """What run.py demo prints: the recommendation at ``minute`` and a plain briefing."""
    live = LiveDay(key, seed)
    live.advance_to(minute)
    rec = recommend_now(live, replications, workers)
    return {"scenario": live.scenario, "config": live.cfg, "rec": rec, "briefing": build_briefing(live.scenario, rec),
            "description": registry.description(key), "label": registry.label(key)}


def rolling_view(key: str, seed: int = 42, start: float = 0.0, until: float = 1440.0, step: float = 30.0,
                 workers: int = 1, replications: int = DEFAULT_REPLICATIONS) -> Iterator[Recommendation]:
    """Every ``step`` minutes: advance the true day and recommend for the next 4 hours (actions are NOT applied)."""
    live = LiveDay(key, seed)
    t = start
    while t <= until:
        live.advance_to(t)
        yield recommend_now(live, replications, workers)
        t += step
