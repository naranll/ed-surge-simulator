"""Rolling look-ahead optimizer: CRN what-if simulation of candidate actions.

The first half of this file (candidate_actions ... rolling) is the original 4-hour
optimiser. It is unchanged. The second half ("VERSION 1 ADDITIONS") is new.

VERSION 1 ADDITIONS: OFFLINE PLAN SEARCH (plain-English design notes)
--------------------------------------------------------------------
A "plan" is ONE 6-hour intervention window that starts at a whole hour s and
switches on up to three levers inside that window:
  * extra clinician   : 0 or 1            (Action(doc=1), 6 staff-hours)
  * fast track        : off or on         (ATS 4-5 consults 40% shorter)
  * boarding cut      : 0%, 25% or 50%    (admitted patients leave the bed sooner)
"Do nothing" means no lever on. Every tunable number is listed, with its tag
(ASSUMPTION) and status ("unverified"), in OPTIMISER_PARAMETERS below.

Plan grid. Window starts every 3 hours: 0, 3, 6, 9, 12, 15, 18 (the last window
ends exactly at midnight). Why 3 hours: a 6-hour window moved by 3 hours overlaps
the old one by half, so a finer grid only repeats almost the same plan, while a
coarser one could miss the evening peak. 7 starts x 11 lever mixes + do nothing
= 78 plans; 78 plans x 50 seeds is about 3,900 runs of about 3 ms each. Quick mode
uses starts 0, 6, 12, 18 and boarding cuts 0% / 50% only (29 plans) and 5 seeds.

How a plan is chosen (select_plan). Only the FORECAST is used, never the true day:
rates = forecast(info) -> for each look-ahead seed i, patients_from_rates(rates,
lookahead_seed(i)) -> every plan is run on the same patients (common random
numbers, so differences between plans are not luck) -> score each run -> the best
plan by mean score wins, but only among plans that keep the fairness limit.
select_plan has no "truth" argument and never calls generate().

Score (percentage points, higher is better), per run:
    100 x (urgent share seen within the ATS target)       urgent = levels 1-3
    - COST_PENALTY_PP_PER_1000_USD x cost / 1000
    - LWBS_PENALTY_PP x (patients who left without being seen, all levels)
Fairness: the median wait of non-urgent patients (levels 4-5) may rise at most
15% over "do nothing" (plus a 1 minute slack so a median of ~0 is not hypersensitive).

How the report is made (evaluate_on_truth). Separate TRUTH seeds are used:
generate(name, truth_seed(i)) for i < n_seeds. The chosen plan and "do nothing"
run on the SAME true day (paired). The paired difference of scores is averaged and
given a 95% range: mean +/- t x s / sqrt(n), using a small Student-t table.
The plan is RECOMMENDED only if the low end of that range is above zero AND the
fairness limit holds on the truth seeds. Otherwise the recommendation is
"do nothing". Only the chosen plan and "do nothing" are run on the truth seeds.

The boarding-cut shim. The simulator has ONE fixed boarding cut (30%, the
module constant ed_simulator.EXPEDITE_FACTOR = 0.7). To test 25% and 50% without
editing the simulator, boarding_cut_shim() sets that constant to 0.75 / 0.5 for the
duration of a run and puts it back in a "finally" block (also on errors). It is not
thread-safe; processes are fine because each has its own copy of the module.
WHAT THE SIMULATOR OWNER SHOULD ADD so the shim can be deleted:
  1. Action(boarding_cut: float = 0.0)  - fraction 0..1; the engine multiplies the
     boarding time by (1 - boarding_cut) for patients whose consult ends inside the
     window (and for patients already boarding when the window opens). It replaces
     the global EXPEDITE_FACTOR / the bool "expedite".
  2. Action(delay: float = 0.0) (or "start") - minutes before the action takes
     effect; EDSim.apply() then schedules the start and the end as events at
     now + delay and now + delay + duration ("scheduled
     resource changes with delays"). Today we emulate it with run_until(start).
  3. Per-window bookkeeping: ft_from/ft_until and ex_from/ex_until/ex_factor
     stored on the EDSim, so two windows can coexist and snapshots carry them.
  4. Action.cost should scale with duration and boarding_cut.
  5. SimResult (or EDSim) could return median wait and on-time share per
     triage level; today compute_metrics() below reads EDSim.pat (read only).

Rolling loop (rolling_plan). Every 30 simulated minutes the live ED is
snapshotted (future hidden), the forecast gives the next hours, 12 candidate
actions (do nothing + 11 lever mixes, 4-hour window starting now) are run
n_seeds times from the snapshot, and an action is applied only if the 95% range of
its paired gain over "do nothing" is above zero (no false alarms). A candidate
that differs from the previous decision pays a small stability penalty. An action
is never applied while the previous one is still active (no stacking).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import heapq
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, NamedTuple, Optional, Sequence

from src.simulation import ed_simulator as _sim_module
from src.simulation.ed_simulator import COSTS, Action, EDSim, LiveED, SimResult, simulate
from src.simulation.scenarios import (ATS_TARGET, ATS_WEIGHT, SCENARIOS, Scenario,
                                      generate_arrivals)

COST_DIV = 10.0             # $ -> loss points
SEEDS = (11, 12, 13, 14, 15)  # 5 CRN seeds (forecast streams, distinct from the live truth seed)
FORECAST_PID = 10**6
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
PRETTY = {"ADD_NURSE": "Call 1 Surge Triage Nurse", "ADD_PHYSICIAN": "Call 1 On-Call MD",
          "FAST_TRACK": "Activate Fast-Track", "INPATIENT_BUFFER_EXPEDITE": "Expedite Inpatient Transfers",
          "DO_NOTHING": "Hold Current Staffing"}


def candidate_actions() -> list[Action]:
    names = ["DO_NOTHING", "ADD_NURSE", "ADD_PHYSICIAN", "FAST_TRACK", "INPATIENT_BUFFER_EXPEDITE",
             "ADD_PHYSICIAN+FAST_TRACK", "ADD_NURSE+ADD_PHYSICIAN", "ADD_PHYSICIAN+INPATIENT_BUFFER_EXPEDITE",
             "FAST_TRACK+INPATIENT_BUFFER_EXPEDITE", "ADD_PHYSICIAN+FAST_TRACK+INPATIENT_BUFFER_EXPEDITE"]
    return [Action.parse(n) for n in names]


def pretty(name: str) -> str:
    return " + ".join(PRETTY[p] for p in name.split("+"))


def clinical_loss(r: SimResult) -> float:
    return (sum(r.breaches[i] * ATS_WEIGHT[i] for i in range(1, 6))
            + 5.0 * r.median_wait + 50.0 * r.lwbs)


def total_loss(r: SimResult, cost: float) -> float:
    """Loss = sum(breaches*weight) + 5*median wait + 50*LWBS + cost/10."""
    return clinical_loss(r) + cost / COST_DIV


def _ci(xs: Sequence[float]) -> tuple[float, float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, m, m
    # CHANGED 2026-10-18: full Student-t table (t_critical). Same values as the old _T95 up to df 9;
    # the old code fell back to 1.96 above df 9, which is too narrow for e.g. 30 replications (t = 2.045).
    h = t_critical(len(xs) - 1) * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def _job(args) -> SimResult:
    snap, arr, act, hours = args
    return simulate(snap, arr, act, hours=hours)


def summarize(rs: Sequence[SimResult]) -> dict:
    n = len(rs)
    avg = lambda f: sum(f(r) for r in rs) / n
    return {
        "median_wait": avg(lambda r: r.median_wait), "lwbs": avg(lambda r: r.lwbs),
        "breaches": avg(lambda r: r.total_breaches), "n": avg(lambda r: r.n),
        "breach_by_ats": {i: avg(lambda r: r.breaches[i]) for i in range(1, 6)},
        "total_by_ats": {i: avg(lambda r: r.totals[i]) for i in range(1, 6)},
        "util": {k: avg(lambda r: r.util[k]) for k in ("nurse", "physician", "bed")},
        "provider_max": avg(lambda r: r.queues["provider_max"]),
        "provider_end": avg(lambda r: r.queues["provider_end"]),
    }


@dataclass
class Recommendation:
    t: float
    action: str
    cost: float
    loss_base: float
    loss_best: float
    wait_reduction: float
    wait_reduction_ci: tuple
    breach_reduction_pct: float
    breach_reduction_ci: tuple
    loss_reduction_ci: tuple
    pareto: list
    table: list
    base: dict
    best: dict
    state: dict
    results: dict = field(repr=False, default_factory=dict)

    def headline(self) -> str:
        if self.action == "DO_NOTHING":
            return "RECOMMENDED: Hold current staffing. No intervention beats baseline over the next 4h."
        return (f"RECOMMENDED: {pretty(self.action)}. Projected Breach Reduction: "
                f"{self.breach_reduction_pct:.0f}%, Wait Time Saved: {self.wait_reduction:.0f} min "
                f"(cost ${self.cost:,.0f})")


def recommend(snapshot: EDSim, scenario: Scenario = SCENARIOS["normal"], seeds: Sequence[int] = SEEDS,
              hours: float = 4, candidates: Optional[Sequence[Action]] = None,
              workers: int = 1, forecasts: Optional[dict] = None) -> Recommendation:
    """Evaluate every candidate on identical forecast streams (CRN) and pick the min-loss action.

    The snapshot should come from ``LiveED.snapshot()`` (no future arrivals); the
    forward stream is *forecast* from the scenario rates, never read from the live truth.
    ADDED 2026-10-18: ``forecasts`` = {seed: [Arrival]} lets a caller pass forecast patients it built itself
    (e.g. with real-data stay times); then ``seeds`` is ignored. Default None = the old behaviour.
    """
    cands = list(candidates or candidate_actions())
    if not any(a.name == "DO_NOTHING" for a in cands):
        cands.insert(0, Action())
    t0, t1 = snapshot.now, snapshot.now + hours * 60.0
    if forecasts is not None:
        seeds = list(forecasts)
        arr = dict(forecasts)
    else:
        arr = {s: generate_arrivals(scenario, s, t0, t1, base_pid=FORECAST_PID) for s in seeds}
    jobs = [(snapshot, arr[s], a, hours) for a in cands for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(_job, jobs, chunksize=max(1, len(jobs) // (workers * 4))))
    else:
        out = [_job(j) for j in jobs]
    res = {a.name: out[i * len(seeds):(i + 1) * len(seeds)] for i, a in enumerate(cands)}
    cost = {a.name: a.cost for a in cands}
    base = res["DO_NOTHING"]
    table = []
    for a in cands:
        rs = res[a.name]
        cl = sum(clinical_loss(r) for r in rs) / len(rs)
        table.append({"action": a.name, "cost": a.cost, "clinical": cl, "loss": cl + a.cost / COST_DIV,
                      **{k: v for k, v in summarize(rs).items() if k in ("median_wait", "lwbs", "breaches")}})
    best = min(table, key=lambda x: (x["loss"], x["cost"]))
    pareto = [x["action"] for x in table if not any(
        y["clinical"] <= x["clinical"] and y["cost"] <= x["cost"]
        and (y["clinical"] < x["clinical"] or y["cost"] < x["cost"]) for y in table)]
    bn, br = best["action"], res[best["action"]]
    wait = [b.median_wait - r.median_wait for b, r in zip(base, br)]
    pct = [100.0 * (b.total_breaches - r.total_breaches) / b.total_breaches if b.total_breaches else 0.0
           for b, r in zip(base, br)]
    dl = [total_loss(b, 0.0) - total_loss(r, cost[bn]) for b, r in zip(base, br)]
    return Recommendation(
        t0, bn, cost[bn], next(x["loss"] for x in table if x["action"] == "DO_NOTHING"), best["loss"],
        _ci(wait)[0], _ci(wait), _ci(pct)[0], _ci(pct), _ci(dl), pareto, table,
        summarize(base), summarize(br),
        {"t": t0, "triage_q": snapshot.cnt["tri_q"], "provider_q": snapshot.cnt["bed_q"] + snapshot.cnt["doc_q"],
         "boarders": snapshot.cnt["board"],   # legacy meaning: everyone in a bed after the consult (kept for the old briefing)
         "admitted_boarders": getattr(snapshot, "board_admitted", None), "beds_busy": snapshot.busy["b"], "beds_cap": snapshot.cap["b"],
         "docs_busy": snapshot.busy["d"], "docs_cap": snapshot.cap["d"]},
        res)


def rolling(live: LiveED, scenario: Scenario, start: float = 0.0, until: float = 1440.0,
            step: float = 30.0, apply: bool = False, **kw) -> Iterator[Recommendation]:
    """30-minute evaluation loop against the live simulator."""
    t = start
    while t <= until:
        live.advance_to(t)
        rec = recommend(live.snapshot(), scenario, **kw)
        if apply and rec.action != "DO_NOTHING":
            live.sim.apply(rec.action)
        yield rec
        t += step


# =============================================================================
# VERSION 1 ADDITIONS: offline plan search and rolling_plan.
# Everything below this line is new. Nothing above it was changed.
# =============================================================================

DAY_MIN = 1440.0                         # one day in minutes (the simulator's clock)
WINDOW_MIN = 360.0                       # one intervention window = 6 hours
START_HOURS = (0, 3, 6, 9, 12, 15, 18)   # window starts; 18 is the last that ends by midnight
QUICK_START_HOURS = (0, 6, 12, 18)       # coarse grid for --quick (windows tile the day)
CLINICIAN_OPTIONS = (0, 1)               # extra clinician: none or one
BOARDING_CUTS = (0.0, 0.25, 0.5)         # boarding time cut: none / 25% / 50%
QUICK_BOARDING_CUTS = (0.0, 0.5)
DEFAULT_SEEDS = 50                       # runs per plan
QUICK_SEEDS = 5
FAIRNESS_LIMIT = 0.15                    # non-urgent median wait may rise at most 15%
FAIRNESS_SLACK_MIN = 1.0                 # ... plus 1 minute, so a ~0 median is not over-sensitive
# CHANGED 2026-10-17 (new rule): only recommend a plan when the LOWER end of its 95% gain range is above
# this many score points (1 point = 1 percentage point of urgent patients seen on time). Before, any
# gain above 0 counted, so on a normal Monday tiny gains (+0.2 to +3.7) caused "false alarms" in 4 of 10
# repeated searches. 2.0 was picked on seed blocks 0-9, then tried on FRESH blocks 10-19: normal day
# "do nothing" 10 of 10, heatwave recommended 10 of 10, hurricane recommended 10 of 10 (small test,
# placeholder parameters). ASSUMPTION: a practical-significance threshold, not a statistical one.
MIN_BENEFIT_PP = 2.0
CLINICIAN_COST_PER_HOUR = COSTS["ADD_PHYSICIAN"] / 4.0                      # 600 per 4 h -> 150
FAST_TRACK_COST_PER_WINDOW = COSTS["FAST_TRACK"]                            # 150 per window
BOARDING_CUT_COST_PER_POINT = COSTS["INPATIENT_BUFFER_EXPEDITE"] / 30.0     # 300 for 30 points -> 10
COST_PENALTY_PP_PER_1000_USD = 7.0       # score points lost per $1000 (see note in the table)
LWBS_PENALTY_PP = 0.5                    # score points lost per patient who leaves unseen
STABILITY_PENALTY_PP = 0.5               # rolling loop: score points lost for changing plan
ROLLING_STEP_MIN = 30.0                  
ROLLING_HORIZON_MIN = 240.0              # look 4 hours ahead
ROLLING_WINDOW_MIN = 240.0               # a rolling action lasts 4 hours (same as the old COSTS)
ROLLING_SEEDS = 40                       # 40 runs per candidate


def _param(name, value, note):
    return {"name": name, "value": value, "tag": "ASSUMPTION", "status": "unverified", "note": note}


# Every tunable number of the version 1 optimiser. All are our own choices (ASSUMPTION)
# and none is verified. Keep this table in sync with the dashboard's parameter table.
OPTIMISER_PARAMETERS = [
    _param("WINDOW_MIN", WINDOW_MIN, "One 6-hour intervention window per plan."),
    _param("START_HOURS", list(START_HOURS),
           "Window starts every 3 h: a 3 h shift keeps half of the window, so a finer grid repeats "
           "almost the same plan and a coarser grid may miss the evening peak. 18 is the last start "
           "whose window ends by midnight."),
    _param("QUICK_START_HOURS", list(QUICK_START_HOURS), "Coarse grid used by --quick only."),
    _param("CLINICIAN_OPTIONS", list(CLINICIAN_OPTIONS), "Extra clinician in the window: none or one."),
    _param("BOARDING_CUTS", list(BOARDING_CUTS),
           "Boarding time cut for admitted patients. The simulator only knows a fixed 30% cut, so "
           "25% and 50% use boarding_cut_shim()."),
    _param("QUICK_BOARDING_CUTS", list(QUICK_BOARDING_CUTS), "Coarse boarding cuts used by --quick only."),
    _param("DEFAULT_SEEDS", DEFAULT_SEEDS, "Runs per plan for the look-ahead, and truth days for the report."),
    _param("QUICK_SEEDS", QUICK_SEEDS, "Seeds used by --quick (a smoke test, not a result)."),
    _param("FAIRNESS_LIMIT", FAIRNESS_LIMIT,
           "Non-urgent (levels 4-5) median wait may rise at most this fraction versus do nothing "
           "(default 15%)."),
    _param("MIN_BENEFIT_PP", MIN_BENEFIT_PP,
           "Recommend a plan only if the LOWER end of its 95% gain range is above this many score points "
           "(a practical-significance threshold; tried on fresh seeds, see the comment at the constant)."),
    _param("FAIRNESS_SLACK_MIN", FAIRNESS_SLACK_MIN,
           "Extra minutes allowed on top of the limit so a median near 0 minutes is not over-sensitive."),
    _param("CLINICIAN_COST_PER_HOUR", CLINICIAN_COST_PER_HOUR,
           "USD per extra clinician staff-hour. Derived from the existing COSTS: ADD_PHYSICIAN 600 per 4 h."),
    _param("FAST_TRACK_COST_PER_WINDOW", FAST_TRACK_COST_PER_WINDOW,
           "USD per window, from the existing COSTS['FAST_TRACK']. The simulator adds no staff for fast "
           "track, so it counts as 0 staff-hours."),
    _param("BOARDING_CUT_COST_PER_POINT", BOARDING_CUT_COST_PER_POINT,
           "USD per percentage point of boarding cut: the existing 300 for the fixed 30% cut, scaled "
           "linearly (25% = 250, 50% = 500)."),
    _param("COST_PENALTY_PP_PER_1000_USD", COST_PENALTY_PP_PER_1000_USD,
           "Score points lost per $1000. Chosen to match the old loss: an urgent breach is worth about "
           "14 loss points = $140 at COST_DIV 10, i.e. about 1 point of on-time share is worth $140."),
    _param("LWBS_PENALTY_PP", LWBS_PENALTY_PP,
           "Score points lost per patient who leaves without being seen (any level). Half an urgent "
           "breach, because an urgent LWBS already counts as a breach."),
    _param("STABILITY_PENALTY_PP", STABILITY_PENALTY_PP,
           "Rolling loop only: score points lost by a candidate that differs from the previous decision."),
    _param("ROLLING_STEP_MIN", ROLLING_STEP_MIN, "Rolling loop re-evaluates every 30 simulated minutes."),
    _param("ROLLING_HORIZON_MIN", ROLLING_HORIZON_MIN, "Rolling loop looks 4 hours ahead."),
    _param("ROLLING_WINDOW_MIN", ROLLING_WINDOW_MIN,
           "Rolling actions last 4 hours, the same length as the old COSTS (so $600 for a clinician)."),
    _param("ROLLING_SEEDS", ROLLING_SEEDS, "Look-ahead runs per candidate in the rolling loop."),
    _param("ATS_TARGET", dict(ATS_TARGET),
           "Patient data uses NHAMCS triage levels 1-5; the ATS time-to-clinician targets are applied "
           "to those levels as an assumption (from scenarios.ATS_TARGET)."),
    _param("CONFIDENCE", "95%", "Range = mean +/- t x s / sqrt(n); small Student-t table (T_975)."),
]

# Two-sided 95% Student-t values by degrees of freedom. For a df that is not listed the
# next LOWER listed df is used (a slightly wider, safer range). Above df 120 we use 1.96.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
         17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
         25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
         120: 1.980}


def t_critical(df: int) -> float:
    """95% two-sided Student-t value for ``df`` degrees of freedom (table above)."""
    if df < 1:
        raise ValueError("need at least 1 degree of freedom")
    if df > 120:
        return 1.96
    return T_975[max(k for k in T_975 if k <= df)]


def confidence_range(values: Sequence[float]) -> tuple:
    """(mean, low, high) of a 95% range for the mean. low/high are None when n < 2."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    m = sum(values) / n
    if n < 2:
        return m, None, None
    half = t_critical(n - 1) * statistics.stdev(values) / math.sqrt(n)
    return m, m - half, m + half


@dataclass(frozen=True)
class Settings:
    """The tunable numbers that change a decision. Defaults come from the constants above."""
    fairness_limit: float = FAIRNESS_LIMIT
    fairness_slack_min: float = FAIRNESS_SLACK_MIN
    cost_penalty_pp_per_1000_usd: float = COST_PENALTY_PP_PER_1000_USD
    lwbs_penalty_pp: float = LWBS_PENALTY_PP
    stability_penalty_pp: float = STABILITY_PENALTY_PP


DEFAULT_SETTINGS = Settings()


# ---- the plan ----------------------------------------------------------------
def _clock(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"


@dataclass(frozen=True)
class Plan:
    """One intervention window. All levers off = "do nothing" (always the same object value)."""
    start: float = 0.0                # minute of the day when the window opens
    clinician: int = 0                # extra clinicians (0 or 1)
    fast_track: bool = False
    boarding_cut: float = 0.0         # 0.0, 0.25, 0.5 ...
    duration: float = WINDOW_MIN      # minutes

    def __post_init__(self):
        if self.clinician < 0 or not 0.0 <= self.boarding_cut < 1.0 or self.duration <= 0:
            raise ValueError("clinician >= 0, 0 <= boarding_cut < 1 and duration > 0 are required")
        if self.is_nothing:           # one canonical "do nothing", whatever start was given
            object.__setattr__(self, "start", 0.0)
            object.__setattr__(self, "duration", WINDOW_MIN)

    @property
    def is_nothing(self) -> bool:
        return self.clinician == 0 and not self.fast_track and self.boarding_cut <= 0.0

    @property
    def levers(self) -> tuple:
        return (self.clinician, self.fast_track, self.boarding_cut)

    @property
    def staff_hours(self) -> float:
        return self.clinician * self.duration / 60.0

    @property
    def cost(self) -> float:
        return (self.clinician * CLINICIAN_COST_PER_HOUR * self.duration / 60.0
                + (FAST_TRACK_COST_PER_WINDOW if self.fast_track else 0.0)
                + BOARDING_CUT_COST_PER_POINT * 100.0 * self.boarding_cut)

    @property
    def label(self) -> str:
        if self.is_nothing:
            return "Do nothing"
        parts = ([f"+{self.clinician} clinician"] * (self.clinician > 0) + ["fast track"] * self.fast_track
                 + [f"boarding -{self.boarding_cut * 100:.0f}%"] * (self.boarding_cut > 0))
        return f"{_clock(self.start)}-{_clock(self.start + self.duration)} " + ", ".join(parts)

    def action(self) -> Action:
        """The simulator action for this plan (the boarding amount needs boarding_cut_shim)."""
        return Action(doc=self.clinician, fast_track=self.fast_track,
                      expedite=self.boarding_cut > 0.0, duration=self.duration)


DO_NOTHING = Plan()


def plan_grid(quick: bool = False) -> list:
    """Do nothing first, then every start x lever mix (at least one lever on)."""
    starts = QUICK_START_HOURS if quick else START_HOURS
    cuts = QUICK_BOARDING_CUTS if quick else BOARDING_CUTS
    plans = [DO_NOTHING]
    for hour in starts:
        for doc in CLINICIAN_OPTIONS:
            for fast in (False, True):
                for cut in cuts:
                    plan = Plan(hour * 60.0, doc, fast, cut)
                    if not plan.is_nothing:
                        plans.append(plan)
    return plans


def plan_dict(plan: Plan) -> dict:
    return {"label": plan.label, "start_hour": None if plan.is_nothing else plan.start / 60.0,
            "clinician": plan.clinician, "fast_track": plan.fast_track,
            "boarding_cut": plan.boarding_cut, "staff_hours": plan.staff_hours, "cost": plan.cost}


# ---- the boarding-cut shim ---------------------------------------------------
@contextlib.contextmanager
def boarding_cut_shim(cut: float):
    """Temporarily set the simulator's boarding factor to (1 - cut); always restore it.

    The simulator reads ed_simulator.EXPEDITE_FACTOR at call time, so changing the module
    constant changes how much Action(expedite=True) cuts. cut 0/None changes nothing.
    """
    if not cut:
        yield
        return
    if not 0.0 < cut < 1.0:
        raise ValueError("boarding cut must be between 0 and 1")
    old = _sim_module.EXPEDITE_FACTOR
    _sim_module.EXPEDITE_FACTOR = 1.0 - cut
    try:
        yield
    finally:
        _sim_module.EXPEDITE_FACTOR = old


# ---- measuring one run -------------------------------------------------------
class RunMetrics(NamedTuple):
    urgent_n: int                  # patients with triage level 1-3
    urgent_on_time: int            # ... seen within their ATS target
    nonurgent_n: int               # levels 4-5
    nonurgent_median_wait: float   # minutes (still-waiting patients are counted up to the end)
    lwbs: int                      # left without being seen, all levels

    @property
    def urgent_share(self) -> float:
        return self.urgent_on_time / self.urgent_n if self.urgent_n else 1.0


def compute_metrics(sim: EDSim, t0: float = 0.0, t_end: float = DAY_MIN) -> RunMetrics:
    """Read the finished simulator (read only). Same inclusion rule as EDSim.result():
    patients already finished before ``t0`` are left out; patients still waiting at
    ``t_end`` are counted with the wait they have so far."""
    urgent_n = on_time = lwbs = 0
    slow_waits = []
    for p in sim.pat.values():
        if p.st == "future":
            continue
        if p.t_doc is not None:
            if p.t_doc < t0:
                continue
            wait = p.t_doc - p.t
        elif p.st == "lwbs":
            if p.t_lwbs < t0:
                continue
            wait = p.t_lwbs - p.t
            lwbs += 1
        else:
            wait = t_end - p.t
        if p.ats <= 3:
            urgent_n += 1
            on_time += (wait <= ATS_TARGET[p.ats] + 1e-9) and p.st != "lwbs"
        else:
            slow_waits.append(wait)
    return RunMetrics(urgent_n, int(on_time), len(slow_waits),
                      statistics.median(slow_waits) if slow_waits else 0.0, lwbs)


def score_run(m: RunMetrics, cost: float, settings: Settings = DEFAULT_SETTINGS) -> float:
    """Score of one run in percentage points. Higher is better. LWBS counts against it."""
    return (100.0 * m.urgent_share - settings.cost_penalty_pp_per_1000_usd * cost / 1000.0
            - settings.lwbs_penalty_pp * m.lwbs)


def fairness_ok(plan_wait: float, base_wait: float, settings: Settings = DEFAULT_SETTINGS) -> bool:
    """Non-urgent median wait may not rise more than the limit (plus the small slack)."""
    return plan_wait <= base_wait * (1.0 + settings.fairness_limit) + settings.fairness_slack_min


def wait_change_pct(plan_wait: float, base_wait: float) -> float:
    if base_wait > 1e-9:
        return 100.0 * (plan_wait - base_wait) / base_wait
    return 0.0 if plan_wait <= 1e-9 else 100.0


def stability_penalty(levers: tuple, previous_levers: tuple, settings: Settings = DEFAULT_SETTINGS) -> float:
    """Rolling loop: a small penalty for changing course."""
    return 0.0 if levers == previous_levers else settings.stability_penalty_pp


# ---- running plans -----------------------------------------------------------
def run_plan(plan: Plan, patients: Sequence) -> RunMetrics:
    """One full day (00:00-24:00): the plan switches on at plan.start and ends by itself."""
    sim = EDSim()
    sim.add_arrivals(patients)
    if plan.is_nothing:
        sim.run_until(DAY_MIN)
    else:
        sim.run_until(plan.start)
        with boarding_cut_shim(plan.boarding_cut):    # factor is only read inside the window
            sim.apply(plan.action())
            sim.run_until(DAY_MIN)
    return compute_metrics(sim, 0.0, DAY_MIN)


def clean_snapshot(snap: EDSim) -> EDSim:
    """Work-around for a simulator quirk (we may not edit ed_simulator.py).

    When an "expedite" speeds up a boarder, the old DEPART event stays in the heap. If that
    patient has already left and EDSim.snapshot(prune=True) deleted them, the old event later
    hits ``self.pat[ref]`` and raises KeyError. Here we drop every event that points at a
    patient the snapshot no longer holds. Events whose ref is text (capacity changes) stay.
    Change only a COPY (LiveED.snapshot() and TruthLive.snapshot() return one).
    """
    keep = [e for e in snap.heap if isinstance(e[3], str) or e[3] in snap.pat]
    if len(keep) != len(snap.heap):
        heapq.heapify(keep)
        snap.heap = keep
    return snap


def run_from_snapshot(plan: Plan, snapshot: EDSim, patients: Sequence, t_end: float) -> RunMetrics:
    """Rolling loop: copy the live snapshot, add forecast patients, apply plan now, run to t_end."""
    sim = EDSim.restore(snapshot)
    sim.add_arrivals(patients)
    with boarding_cut_shim(0.0 if plan.is_nothing else plan.boarding_cut):
        if not plan.is_nothing:
            sim.apply(plan.action())
        sim.run_until(t_end)
    return compute_metrics(sim, snapshot.now, t_end)


def _batch_job(job) -> list:
    """Pool worker: one set of patients, several plans (patients are sent once per seed)."""
    patients, plans = job
    return [run_plan(p, patients) for p in plans]


def _runner_job(job) -> list:
    """Pool worker for a picklable runner: one set of patients, several plans."""
    runner, patients, plans = job
    return [runner(p, patients) for p in plans]


def run_batches(jobs: Sequence, workers: int = 1, runner=None) -> list:
    """jobs = [(patients, [plans])]. Returns, per job, the list of RunMetrics (same order as plans).

    ``runner(plan, patients)`` replaces run_plan (used by tests); it forces a serial run, unless the runner
    says it can be sent to worker processes (attribute ``parallel_safe`` = True, e.g. hospital.HospitalRunner).
    """
    if runner is not None:
        if workers > 1 and len(jobs) > 1 and getattr(runner, "parallel_safe", False):   # ADDED 2026-10-18
            with ProcessPoolExecutor(workers) as ex:
                return list(ex.map(_runner_job, [(runner, pats, plans) for pats, plans in jobs]))
        return [[runner(p, pats) for p in plans] for pats, plans in jobs]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(_batch_job, jobs))
    return [_batch_job(j) for j in jobs]


# ---- the scenario API (real or stub) ----------------------------------------
def load_scenario_api() -> SimpleNamespace:
    """Import the real scenario/forecaster modules (written by another teammate) only when needed."""
    try:   # CHANGED 2026-10-18: the scenario registry is the one definition of every scenario (same days for the old names)
        from src.simulation import registry
        return registry.api()
    except ImportError:
        pass
    try:
        from src.simulation import forecaster, scenarios
        return SimpleNamespace(
            generate=scenarios.generate, forecast=forecaster.forecast,
            truth_seed=scenarios.truth_seed, lookahead_seed=scenarios.lookahead_seed,
            patients_from_rates=scenarios.patients_from_rates,
            list_scenarios=scenarios.list_scenarios,
            scenario_info=getattr(scenarios, "scenario_info", None))   # CHANGED 2026-10-17: see get_info
    except (ImportError, AttributeError) as err:
        raise RuntimeError(
            "The scenario API is missing (needs scenarios.generate / truth_seed / lookahead_seed / "
            "patients_from_rates / list_scenarios and forecaster.forecast): " + str(err)) from err


def check_seed_streams(api, n_seeds: int) -> None:
    """The look-ahead seeds and the truth seeds must never overlap."""
    truth = {api.truth_seed(i) for i in range(n_seeds)}
    look = {api.lookahead_seed(i) for i in range(n_seeds)}
    if truth & look:
        raise ValueError("truth seeds and look-ahead seeds overlap")


def get_info(api, scenario_name: str) -> dict:
    """What a hospital knows in advance. ``info`` depends on the scenario NAME only
    (interface note). CHANGED 2026-10-17: use scenarios.scenario_info(name) when the api offers it, so no
    true day is built at all. The old way (ask generate() once and throw the true day away) is kept only as
    a fallback for stub apis; it was harmless today but would leak if info ever depended on the seed."""
    scenario_info = getattr(api, "scenario_info", None)
    if callable(scenario_info):
        return dict(scenario_info(scenario_name))
    _unused_truth, info = api.generate(scenario_name, api.truth_seed(0))
    return dict(info)


# ---- choosing a plan (forecast only) -----------------------------------------
@dataclass
class PlanStats:
    plan: Plan
    score: float
    on_time_pct: float
    nonurgent_median: float
    nonurgent_change_pct: float
    lwbs: float
    gain: float                    # mean paired score gain over do nothing
    gain_lo: Optional[float]
    gain_hi: Optional[float]
    feasible: bool


@dataclass
class Selection:
    stats: list
    best: PlanStats
    base: PlanStats
    n_seeds: int


def select_plan(info: dict, n_seeds: int, plans: Sequence, api, workers: int = 1,
                settings: Settings = DEFAULT_SETTINGS, runner=None) -> Selection:
    """Pick the best plan using ONLY the forecast. There is deliberately no ``truth``
    argument, and this function never calls ``api.generate``.

    Steps: forecast -> n_seeds look-ahead days -> every plan on the same days -> mean score,
    fairness check -> best feasible plan (ties: cheaper, then earlier).
    """
    plans = list(plans)
    if DO_NOTHING not in plans:
        plans.insert(0, DO_NOTHING)
    rates = api.forecast(info, 0, 24)
    jobs = [(api.patients_from_rates(rates, api.lookahead_seed(i), start_hour=0, base_pid=FORECAST_PID),
             plans) for i in range(n_seeds)]
    runs = run_batches(jobs, workers, runner)                       # runs[seed][plan]
    scores = [[score_run(runs[i][j], p.cost, settings) for j, p in enumerate(plans)]
              for i in range(n_seeds)]
    k0 = plans.index(DO_NOTHING)
    mean = lambda xs: sum(xs) / len(xs)
    base_wait = mean([runs[i][k0].nonurgent_median_wait for i in range(n_seeds)])
    stats = []
    for j, p in enumerate(plans):
        wait = mean([runs[i][j].nonurgent_median_wait for i in range(n_seeds)])
        gain, lo, hi = confidence_range([scores[i][j] - scores[i][k0] for i in range(n_seeds)])
        stats.append(PlanStats(
            p, mean([scores[i][j] for i in range(n_seeds)]),
            100.0 * mean([runs[i][j].urgent_share for i in range(n_seeds)]),
            wait, wait_change_pct(wait, base_wait), mean([runs[i][j].lwbs for i in range(n_seeds)]),
            gain, lo, hi, fairness_ok(wait, base_wait, settings)))
    best = min((s for s in stats if s.feasible),
               key=lambda s: (-s.score, s.plan.cost, s.plan.start, s.plan.label))
    return Selection(stats, best, stats[k0], n_seeds)


# ---- checking the choice on the true days -------------------------------------
def evaluate_on_truth(plan: Plan, scenario_name: str, n_seeds: int, api, burst: bool = False,
                      workers: int = 1, settings: Settings = DEFAULT_SETTINGS, runner=None) -> dict:
    """Run ``plan`` and "do nothing" on the SAME true days (truth seeds). Paired gains."""
    jobs = []
    for i in range(n_seeds):
        truth, _info = api.generate(scenario_name, api.truth_seed(i), burst=burst)
        jobs.append((truth, [DO_NOTHING, plan]))
    runs = run_batches(jobs, workers, runner)                       # runs[seed] = [base, plan]
    mean = lambda xs: sum(xs) / len(xs)
    diffs = [score_run(r[1], plan.cost, settings) - score_run(r[0], 0.0, settings) for r in runs]
    gain, lo, hi = confidence_range(diffs)
    base_wait = mean([r[0].nonurgent_median_wait for r in runs])
    plan_wait = mean([r[1].nonurgent_median_wait for r in runs])
    return {
        "n_seeds": n_seeds, "gain": gain, "gain_lo": lo, "gain_hi": hi,
        "base_on_time_pct": 100.0 * mean([r[0].urgent_share for r in runs]),
        "plan_on_time_pct": 100.0 * mean([r[1].urgent_share for r in runs]),
        "base_nonurgent_median": base_wait, "plan_nonurgent_median": plan_wait,
        "nonurgent_change_pct": wait_change_pct(plan_wait, base_wait),
        "fairness_ok": fairness_ok(plan_wait, base_wait, settings),
        "base_lwbs": mean([r[0].lwbs for r in runs]), "plan_lwbs": mean([r[1].lwbs for r in runs]),
    }


# ---- the full search ----------------------------------------------------------
@dataclass
class PlanSearchResult:
    scenario: str
    n_seeds: int
    quick: bool
    info: dict
    rows: list                     # one dict per plan
    selected: Plan                 # best plan by the look-ahead
    recommended: Plan              # selected plan if it passed the truth check, else do nothing
    reason: str
    truth: dict
    caution: str
    settings: dict

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "n_seeds": self.n_seeds, "quick": self.quick,
                "info": self.info, "selected": plan_dict(self.selected),
                "recommended": plan_dict(self.recommended), "reason": self.reason,
                "truth_check": self.truth, "caution": self.caution, "settings": self.settings,
                "rows": self.rows}

    @property
    def recommended_row(self) -> dict:
        return next(r for r in self.rows if r["recommended"])


def _decide(chosen: Plan, truth: dict) -> tuple:
    """(recommended plan, reason). Needs a gain whose 95% range is above zero and fairness."""
    lo = truth["gain_lo"]
    rng = "n/a" if lo is None else f"{lo:+.2f} to {truth['gain_hi']:+.2f}"
    if chosen.is_nothing:
        return DO_NOTHING, "Do nothing scored best on the look-ahead runs: no plan was worth its cost."
    if not truth["fairness_ok"]:
        return DO_NOTHING, (f"Do nothing: '{chosen.label}' raised the non-urgent median wait by "
                            f"{truth['nonurgent_change_pct']:.0f}% on the truth days (fairness limit broken).")
    if lo is None or lo <= MIN_BENEFIT_PP:      # CHANGED 2026-10-17: was "lo <= 0.0" (see MIN_BENEFIT_PP)
        return DO_NOTHING, (f"Do nothing: the 95% range of the gain of '{chosen.label}' on the truth "
                            f"days ({rng}) does not stay above {MIN_BENEFIT_PP:g} points.")
    return chosen, (f"Recommended: paired gain {truth['gain']:+.2f} points (95% range {rng}) on "
                    f"{truth['n_seeds']} truth days, fairness limit holds.")


def plan_search(scenario_name: str, n_seeds: int = DEFAULT_SEEDS, quick: bool = False, api=None,
                workers: int = 1, burst: bool = False, settings: Optional[Settings] = None,
                runner=None, replications: Optional[int] = None, rule_c: Optional[bool] = None) -> PlanSearchResult:
    """Offline plan search for one scenario.

    quick=True: 5 seeds and the coarse grid. ``api`` offers generate, forecast, truth_seed,
    lookahead_seed, patients_from_rates (None = the real modules, imported now).
    ``burst`` is passed to the TRUE days only (stress test); the forecast never contains a burst.
    """
    api = api or load_scenario_api()
    settings = settings or DEFAULT_SETTINGS
    if quick:
        n_seeds = QUICK_SEEDS
    if replications is not None:      # ADDED 2026-10-18: explicit replications win, also in quick mode
        n_seeds = int(replications)
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    check_seed_streams(api, n_seeds)
    info = get_info(api, scenario_name)
    sel = select_plan(info, n_seeds, plan_grid(quick), api, workers, settings, runner)
    truth = evaluate_on_truth(sel.best.plan, scenario_name, n_seeds, api, burst, workers, settings, runner)
    recommended, reason = _decide(sel.best.plan, truth)
    if rule_c is None:
        rule_c = not _legacy_params()
    if rule_c and recommended.is_nothing and scenario_name != "normal":
        recommended, reason, truth = _apply_rule_c(scenario_name, sel, truth, reason, n_seeds, api, burst, workers, settings, runner)
    caution = ""
    if info.get("burst_possible"):
        caution = ("A sudden burst of urgent patients is possible on this day. The forecast cannot "
                   "include it (its hour is unknown), so the plan was not tested against one"
                   + (" - except on the true days, which include a burst (burst=True)." if burst else "."))
    rows = [_row(s, sel.best.plan, recommended, truth) for s in sel.stats]
    return PlanSearchResult(scenario_name, n_seeds, quick, info, rows, sel.best.plan, recommended,
                            reason, truth, caution, asdict(settings))


# ---- rule (c) -----------------------------------------------------------------
# ADDED 2026-10-18 (team decision, option c): when urgent breaches are ABOVE a normal day, never return "do nothing"
# unless every tested plan's 95% range includes 0; then say "No tested plan helps; constraint is <...>".
# "Breaches rise" = the do-nothing share of urgent (triage 1-3) patients seen late on the truth days is higher than on
# normal-day truth days with the same day numbers. RULE_C_MAX_CHECKS (UNSOURCED compute budget): how many candidate plans
# are re-checked on the truth days before giving up.
RULE_C_MAX_CHECKS = 3


def _legacy_params() -> bool:
    try:
        from src.simulation.scenarios import USE_LEGACY_PARAMS
        return bool(USE_LEGACY_PARAMS)
    except ImportError:
        return False


def normal_reference_late_pct(api, n_seeds: int, workers: int = 1, runner=None) -> float:
    """Do-nothing % of urgent patients seen late on NORMAL truth days (same day numbers)."""
    jobs = [(api.generate("normal", api.truth_seed(i))[0], [DO_NOTHING]) for i in range(n_seeds)]
    runs = run_batches(jobs, workers, runner)
    return 100.0 * (1.0 - sum(r[0].urgent_share for r in runs) / len(runs))


def _apply_rule_c(scenario_name, sel, truth, reason, n_seeds, api, burst, workers, settings, runner):
    info = {"applied": False}
    try:
        ref = normal_reference_late_pct(api, n_seeds, workers, runner)
    except Exception as err:                      # e.g. a stub api without a normal day
        return DO_NOTHING, reason, {**truth, "rule_c": {"applied": False, "note": f"normal reference unavailable: {err!r}"}}
    base = 100.0 - truth["base_on_time_pct"]
    info.update(base_late_pct=base, normal_late_pct=ref, breaches_rise=base > ref + 1e-9)
    if not info["breaches_rise"]:
        return DO_NOTHING, reason, {**truth, "rule_c": info}
    cands = sorted([s for s in sel.stats if not s.plan.is_nothing and s.feasible and s.gain_lo is not None and s.gain_lo > 0],
                   key=lambda s: (-s.score, s.plan.cost, s.plan.start))
    for s in cands[:RULE_C_MAX_CHECKS]:
        t2 = evaluate_on_truth(s.plan, scenario_name, n_seeds, api, burst, workers, settings, runner)
        if t2["gain_lo"] is not None and t2["gain_lo"] > 0 and t2["fairness_ok"]:
            info["applied"] = "plan"
            why = (f"Rule (c): urgent patients are seen late more often than on a normal day ({base:.1f}% vs {ref:.1f}%), so "
                   f"'do nothing' is not allowed while a plan helps. '{s.plan.label}' has a paired gain of {t2['gain']:+.2f} points "
                   f"(95% range {t2['gain_lo']:+.2f} to {t2['gain_hi']:+.2f}) on {t2['n_seeds']} truth days: above 0 but not above "
                   f"the {MIN_BENEFIT_PP:g}-point bar. Fairness limit holds.")
            return s.plan, why, {**t2, "rule_c": info}
    diag = getattr(runner, "diagnose", None)
    constraint = (diag(api.generate(scenario_name, api.truth_seed(0), burst=burst)[0]) if callable(diag)
                  else "not known (no hospital model is attached to this search)")
    info.update(applied="no_plan_helps", constraint=constraint)
    why = (f"No tested plan helps; constraint is {constraint}. Urgent patients are seen late more often than on a normal day "
           f"({base:.1f}% vs {ref:.1f}%), but every tested plan's 95% range includes 0 or fails the truth / fairness check.")
    return DO_NOTHING, why, {**truth, "rule_c": info}


def _row(s: PlanStats, selected: Plan, recommended: Plan, truth: dict) -> dict:
    p = s.plan
    row = {
        "plan": p.label, "start_hour": None if p.is_nothing else p.start / 60.0,
        "clinician": p.clinician, "fast_track": p.fast_track, "boarding_cut_pct": p.boarding_cut * 100.0,
        "staff_hours": p.staff_hours, "cost_usd": p.cost,
        "urgent_on_time_pct": s.on_time_pct, "nonurgent_median_wait_min": s.nonurgent_median,
        "nonurgent_wait_change_pct": s.nonurgent_change_pct, "lwbs": s.lwbs, "score": s.score,
        "lookahead_gain": s.gain, "lookahead_gain_lo": s.gain_lo, "lookahead_gain_hi": s.gain_hi,
        "feasible": s.feasible, "selected": p == selected,
        "truth_gain": None, "truth_gain_lo": None, "truth_gain_hi": None,
        "truth_urgent_on_time_pct": None, "truth_nonurgent_median_wait_min": None, "truth_lwbs": None,
        "recommended": p == recommended,
    }
    if p == selected and not p.is_nothing:      # only the chosen plan and do nothing run on truth days
        row.update(truth_gain=truth["gain"], truth_gain_lo=truth["gain_lo"], truth_gain_hi=truth["gain_hi"],
                   truth_urgent_on_time_pct=truth["plan_on_time_pct"],
                   truth_nonurgent_median_wait_min=truth["plan_nonurgent_median"],
                   truth_lwbs=truth["plan_lwbs"])
    if p.is_nothing:
        row.update(truth_gain=0.0, truth_urgent_on_time_pct=truth["base_on_time_pct"],
                   truth_nonurgent_median_wait_min=truth["base_nonurgent_median"],
                   truth_lwbs=truth["base_lwbs"])
    return row


# ---- printing and saving ------------------------------------------------------
def _rng(lo, hi) -> str:
    return "n/a" if lo is None else f"{lo:+.1f}..{hi:+.1f}"


def format_table(result: PlanSearchResult, top: int = 10) -> str:
    """Plain-text table, best score first. top=0 shows every plan. The recommended plan,
    the look-ahead pick and "do nothing" are always shown."""
    rows = sorted(result.rows, key=lambda r: -r["score"])
    shown = rows if not top or top >= len(rows) else rows[:top]
    for r in rows:
        if (r["recommended"] or r["selected"] or r["plan"] == DO_NOTHING.label) and r not in shown:
            shown.append(r)
    i = result.info
    out = [f"Scenario: {result.scenario}   (surge known in advance: {i.get('surge_known')}, "
           f"from {i.get('known_from')}, window {i.get('window')}, burst possible: {i.get('burst_possible')})",
           f"Look-ahead seeds: {result.n_seeds}   Truth seeds: {result.n_seeds}   "
           f"Plans tested: {len(rows)}   Mode: {'quick' if result.quick else 'full'}",
           f"Look-ahead pick : {result.selected.label}",
           f"RECOMMENDED     : {result.recommended.label}",
           f"Why             : {result.reason}"]
    if not result.recommended.is_nothing:
        out.append(f"Cost            : ${result.recommended.cost:,.0f} = {result.recommended.staff_hours:g} staff-hours "
                   "(costs are UNSOURCED, illustrative)")
    t = result.truth
    if not result.selected.is_nothing:
        out.append(f"Truth check     : urgent on time {t['base_on_time_pct']:.1f}% -> {t['plan_on_time_pct']:.1f}%, "
                   f"non-urgent median wait {t['base_nonurgent_median']:.1f} -> {t['plan_nonurgent_median']:.1f} min "
                   f"({t['nonurgent_change_pct']:+.0f}%), LWBS {t['base_lwbs']:.1f} -> {t['plan_lwbs']:.1f}")
    if result.caution:
        out.append("Caution         : " + result.caution)
    out += ["", f"{'Plan':<52}{'OnTime%':>8}{'NU wait':>9}{'NU chg%':>8}{'LWBS':>6}{'StaffH':>7}"
                f"{'Cost$':>7}{'Score':>8}  {'Gain vs nothing (95%)':<20}{'Fair':>5}  Flag"]
    for r in shown:
        flag = "RECOMMENDED" if r["recommended"] else ("look-ahead pick" if r["selected"] else "")
        out.append(f"{r['plan']:<52}{r['urgent_on_time_pct']:>8.1f}{r['nonurgent_median_wait_min']:>9.1f}"
                   f"{r['nonurgent_wait_change_pct']:>+8.0f}{r['lwbs']:>6.1f}{r['staff_hours']:>7.0f}"
                   f"{r['cost_usd']:>7.0f}{r['score']:>8.1f}  "
                   f"{r['lookahead_gain']:>+6.1f} {_rng(r['lookahead_gain_lo'], r['lookahead_gain_hi']):<12}"
                   f"{'yes' if r['feasible'] else 'NO':>5}  {flag}")
    if len(shown) < len(rows):
        out.append(f"({len(shown)} of {len(rows)} plans shown; use --top 0 to show all)")
    out.append("Gain = score points over do nothing (1 point = 1 percentage point of urgent patients "
               "seen on time, after cost and LWBS penalties). Costs are UNSOURCED, illustrative; StaffH = staff-hours.")
    return "\n".join(out)


def write_results(result: PlanSearchResult, out_dir) -> list:
    """Write plan_search_<scenario>.csv and .json into out_dir. Returns the two paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"plan_search_{result.scenario}.csv"
    json_path = out / f"plan_search_{result.scenario}.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result.rows[0]))
        writer.writeheader()
        writer.writerows(result.rows)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, allow_nan=False)
    return [csv_path, json_path]


def main(argv=None, api=None, out_dir=None) -> int:
    """Command line: ``py -3.12 -B -m src.optimization.optimizer --quick``.

    --quick : scenarios normal and heatwave, 5 seeds, coarse grid, prints a table, writes NOTHING.
    full    : every scenario, writes results/plan_search_<scenario>.csv and .json (repo root).
    """
    ap = argparse.ArgumentParser(prog="python -m src.optimization.optimizer",
                                 description="Offline ED plan search (digital twin).")
    ap.add_argument("--quick", action="store_true", help="2 scenarios, 5 seeds, print only, write nothing")
    ap.add_argument("--scenario", nargs="+", help="scenario names (default: normal+heatwave quick, all full)")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="seeds per plan (full mode)")
    ap.add_argument("--workers", type=int, default=1, help="processes (1 = serial, the default)")
    ap.add_argument("--top", type=int, default=10, help="rows to print per scenario (0 = all)")
    ap.add_argument("--burst", action="store_true", help="add the burst stress test to the TRUE days only")
    ap.add_argument("--out", default=None, help="output folder for full mode (default: results/ in the repo root)")
    args = ap.parse_args(argv)
    try:
        api = api or load_scenario_api()
    except RuntimeError as err:
        print("Cannot run: " + str(err), file=sys.stderr)
        return 2
    if args.scenario:
        names = list(args.scenario)
    elif args.quick:
        names = ["normal", "heatwave"]
    else:
        names = list(api.list_scenarios()) if hasattr(api, "list_scenarios") else ["normal", "heatwave", "hurricane"]
    target = Path(args.out or out_dir or Path(__file__).resolve().parents[2] / "results")
    for name in names:
        try:
            result = plan_search(name, args.seeds, args.quick, api, args.workers, args.burst)
        except (ValueError, KeyError) as err:
            print(f"Cannot run scenario '{name}': {err!r}", file=sys.stderr)
            return 2
        print(format_table(result, args.top))
        print()
        if not args.quick:
            for path in write_results(result, target):
                print("wrote", path)
    return 0


# ---- rolling loop (version 2) --------------------------------------------------
class TruthLive:
    """A live ED built from a true day (list of Arrival). Same interface as LiveED:
    ``.sim``, ``.now``, ``advance_to(t)``, ``snapshot()`` (future hidden)."""

    def __init__(self, truth: Sequence):
        self.sim = EDSim()
        self.sim.add_arrivals(truth)

    @property
    def now(self) -> float:
        return self.sim.now

    def advance_to(self, t: float) -> None:
        self.sim.run_until(t)

    def snapshot(self) -> EDSim:
        return self.sim.snapshot(strip_future=True, prune=True)


def rolling_candidates(start: float, window: float = ROLLING_WINDOW_MIN) -> list:
    """Do nothing + 11 lever mixes starting now (12 candidates, plan asks for 10-15)."""
    plans = [DO_NOTHING]
    for doc in CLINICIAN_OPTIONS:
        for fast in (False, True):
            for cut in BOARDING_CUTS:
                plan = Plan(start, doc, fast, cut, window)
                if not plan.is_nothing:
                    plans.append(plan)
    return plans


def rolling_plan(live, info: dict, api=None, start: float = 0.0, until: float = DAY_MIN,
                 step: float = ROLLING_STEP_MIN, n_seeds: int = ROLLING_SEEDS,
                 horizon: float = ROLLING_HORIZON_MIN, window: float = ROLLING_WINDOW_MIN,
                 settings: Settings = DEFAULT_SETTINGS, apply: bool = True) -> list:
    """Re-decide every ``step`` minutes. Returns one dict per tick.

    ``live`` offers .sim, .now, .advance_to(t) and .snapshot() (LiveED or TruthLive). The
    optimiser only ever sees snapshot() (no future) and the forecast. At each tick:
      * if the previous action's window is still active -> hold (NO STACKING);
      * else run the 12 candidates ``n_seeds`` times from the snapshot on forecast patients,
        subtract a stability penalty from every candidate that differs from the last decision,
        pick the best fair candidate, and apply it only if the 95% range of its paired gain
        over "do nothing" is above zero (no false alarms).
    Assumption: api.patients_from_rates(rates, seed, start_hour) returns absolute minutes of the day.
    """
    api = api or load_scenario_api()
    rows, t, tick = [], start, 0
    active_until, active_cut, previous = -1.0, 0.0, DO_NOTHING.levers
    while t <= until + 1e-9:
        # advance the live ED; the shim must stay on while an active window can still cut boarding
        with boarding_cut_shim(active_cut if live.now < active_until else 0.0):
            live.advance_to(t)
        t_end = min(t + horizon, DAY_MIN)
        if t_end <= t + 1e-9:
            break
        sim = live.sim
        if t < active_until - 1e-9 or sim.ft_until > t + 1e-9 or sim.ex_until > t + 1e-9:
            rows.append({"t": t, "evaluated": False, "applied": False, "action": "hold",
                         "reason": f"previous action still active until minute {active_until:.0f}",
                         "gain": None, "gain_lo": None, "gain_hi": None})
            t, tick = t + step, tick + 1
            continue
        snapshot = clean_snapshot(live.snapshot())
        hour = int(t // 60)
        rates = api.forecast(info, hour, int(math.ceil(horizon / 60.0)) + 1)
        cands = rolling_candidates(t, window)
        patients = []
        for i in range(n_seeds):
            seed = api.lookahead_seed((tick * n_seeds + i) % 100_000)
            raw = api.patients_from_rates(rates, seed, start_hour=hour, base_pid=FORECAST_PID)
            patients.append([a for a in raw if t <= a.t < t_end])
        runs = [[run_from_snapshot(c, snapshot, patients[i], t_end) for c in cands] for i in range(n_seeds)]
        scores = [[score_run(runs[i][j], c.cost, settings) - stability_penalty(c.levers, previous, settings)
                   for j, c in enumerate(cands)] for i in range(n_seeds)]
        mean = lambda xs: sum(xs) / len(xs)
        base_wait = mean([runs[i][0].nonurgent_median_wait for i in range(n_seeds)])
        ok = [j for j in range(len(cands))
              if fairness_ok(mean([runs[i][j].nonurgent_median_wait for i in range(n_seeds)]), base_wait, settings)]
        best = min(ok, key=lambda j: (-mean([scores[i][j] for i in range(n_seeds)]), cands[j].cost, j))
        gain, lo, hi = confidence_range([scores[i][best] - scores[i][0] for i in range(n_seeds)])
        chosen = cands[best]
        go = (not chosen.is_nothing) and lo is not None and lo > 0.0
        if go and apply:
            with boarding_cut_shim(chosen.boarding_cut):
                live.sim.apply(chosen.action())
            active_until, active_cut = t + chosen.duration, chosen.boarding_cut
        previous = chosen.levers if go else DO_NOTHING.levers
        rows.append({"t": t, "evaluated": True, "applied": bool(go and apply),
                     "action": chosen.label if go else DO_NOTHING.label,
                     "reason": ("gain range above zero" if go else
                                "best candidate is do nothing" if chosen.is_nothing else
                                "gain range includes zero (no false alarm)"),
                     "gain": gain, "gain_lo": lo, "gain_hi": hi})
        t, tick = t + step, tick + 1
    return rows


if __name__ == "__main__":
    sys.exit(main())
