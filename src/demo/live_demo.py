"""Live demo for run.py (demo / rolling). Standard library only.

Replaces the old route through src/dashboard/app.py (which is left untouched):
  * capacity from configs/capacity.json (the calibrated reference hospital);
  * the TRUE day from the scenario registry (registry.generate, seed = truth_seed(--seed)); forecast patients from the
    advance information only (registry.forecast) with look-ahead seeds, which never overlap truth seeds;
  * real-data admission and care times (models/patient_flow/admission_lookup.json);
  * the SAME rulebook as the offline plan search, applied to the next 4 hours:
      score = % of urgent (triage 1-3) patients seen on time - cost penalty - left-unseen penalty (optimizer.score_run),
      fairness check on the non-urgent median wait, a plan only if the low end of its 95% gain range is above
      MIN_BENEFIT_PP, and rule (c): when urgent breaches are above a normal day, never say "do nothing" unless every
      tested plan's 95% range includes 0, and then say which resource is the constraint.
  * it always prints the whole-day plan from the offline plan search next to the 4-hour view.
With ED_USE_LEGACY_PARAMS=1, run.py keeps using the old route unchanged.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from src.optimization.optimizer import (BOARDING_CUTS, CLINICIAN_OPTIONS, DEFAULT_SETTINGS, DO_NOTHING, FORECAST_PID,
                                        MIN_BENEFIT_PP, Plan, boarding_cut_shim, compute_metrics, confidence_range,
                                        fairness_ok, score_run, wait_change_pct)
from src.simulation import registry
from src.simulation.ed_simulator import EDSim
from src.simulation.hospital import HospitalConfig, PRESETS, binding_constraint, build_sim, prepare
from src.simulation.scenarios import lookahead_seed, truth_seed

DEFAULT_REPLICATIONS = 30    # forecast runs per plan (run.py --replications); 30 so that 95% ranges are meaningful
HORIZON_MIN = 240.0          # the demo looks 4 hours ahead; its plans last exactly that window
RESULTS = Path(__file__).resolve().parents[2] / "results" / "plan_search"
COST_NOTE = "costs are UNSOURCED, illustrative"


def reference_config() -> HospitalConfig:
    """The calibrated reference hospital from configs/capacity.json."""
    from src.simulation import capacity
    return PRESETS[reference_name()]


def reference_name() -> str:
    from src.simulation import capacity
    return capacity.load()["reference_hospital"].get("preset_name", "Large")


class LiveDay:
    """A true day advanced in real time, for the configured hospital."""

    def __init__(self, key: str, seed: int = 42, cfg: Optional[HospitalConfig] = None):
        self.key, self.seed = key, seed
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


def forecast_patients(live: LiveDay, replications: int, hours: float = HORIZON_MIN / 60.0) -> list:
    """[forecast patients for the next ``hours``] per look-ahead seed, from the advance information only."""
    t0, t1 = live.now, live.now + hours * 60.0
    return [prepare(live.cfg, registry.forecast_patients(live.key, lookahead_seed(i), t0, t1, FORECAST_PID))
            for i in range(replications)]


def demo_plans(now: float) -> list:
    """The plan search's levers, as one window starting NOW and lasting the 4-hour horizon."""
    out = [DO_NOTHING]
    for doc in CLINICIAN_OPTIONS:
        for fast in (False, True):
            for cut in BOARDING_CUTS:
                p = Plan(now, doc, fast, cut, HORIZON_MIN)
                if not p.is_nothing:
                    out.append(p)
    return out


def _run(plan: Plan, snap: EDSim, patients: list, t_end: float):
    """(RunMetrics, finished sim) for one plan from the snapshot. The snapshot is never changed."""
    sim = EDSim.restore(snap)
    sim.add_arrivals(patients)
    sim.reset_stats()
    with boarding_cut_shim(0.0 if plan.is_nothing else plan.boarding_cut):
        if not plan.is_nothing:
            sim.apply(plan.action())
        sim.run_until(t_end)
    return compute_metrics(sim, snap.now, t_end), sim


def _clock(m: float) -> str:
    m = int(round(m)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


@dataclass
class Decision:
    t: float
    key: str
    plan: Plan
    reason: str
    rule: str                    # "bar", "rule_c", "nothing_needed", "no_plan_helps"
    gain: Optional[float]
    gain_lo: Optional[float]
    gain_hi: Optional[float]
    base: dict                   # do-nothing numbers for the next 4 h
    after: dict                  # numbers under the chosen plan (= base for do nothing)
    now_state: dict
    rows: list = field(default_factory=list)
    normal_late_pct: Optional[float] = None
    whole_day: Optional[dict] = None
    differ_note: str = ""

    @property
    def label(self) -> str:
        return "no action" if self.plan.is_nothing else self.plan.label

    def lines(self) -> list:
        s, b = self.now_state, self.base
        out = [f"Now ({_clock(self.t)}): {s['beds_busy']}/{s['beds_cap']} beds in use; {s['admitted_boarders']} admitted patients waiting "
               f"for a ward bed; {s['waiting']} waiting to be seen ({s['waiting_urgent']} urgent).",
               f"If nothing changes, next 4 h: {b['urgent_on_time_pct']:.1f}% of urgent patients (levels 1-3) seen on time, "
               f"non-urgent median wait {b['nonurgent_median_wait']:.0f} min, {b['lwbs']:.1f} leave unseen, beds {b['bed_util_pct']:.0f}% used."]
        if self.plan.is_nothing:
            out.append(f"Next 4h: no action. {self.reason}")
        else:
            a = self.after
            out.append(f"Next 4h: {self.plan.label}. Urgent seen on time {b['urgent_on_time_pct']:.1f}% -> {a['urgent_on_time_pct']:.1f}%, "
                       f"left unseen {b['lwbs']:.1f} -> {a['lwbs']:.1f}; score gain {self.gain:+.1f} points (95% range {self.gain_lo:+.1f} to "
                       f"{self.gain_hi:+.1f}); cost ${self.plan.cost:,.0f} = {self.plan.staff_hours:g} staff-hours ({COST_NOTE}). {self.reason}")
        w = self.whole_day
        out.append("Whole-day plan for this scenario: " + (f"{w['label']} ({w['why']})" if w else
                   "not computed yet (run: python run_all.py, or python scripts/run_scenarios.py)."))
        if self.differ_note:
            out.append(self.differ_note)
        return out


def _stats(plans, runs, base_j):
    """Per plan: mean numbers, paired score gain vs do nothing with 95% range, fairness (the plan search's rules)."""
    n = len(runs)
    mean = lambda xs: sum(xs) / len(xs)
    base_wait = mean([runs[i][base_j][0].nonurgent_median_wait for i in range(n)])
    rows = []
    for j, p in enumerate(plans):
        ms = [runs[i][j][0] for i in range(n)]
        sc = [score_run(m, p.cost, DEFAULT_SETTINGS) for m in ms]
        gains = [sc[i] - score_run(runs[i][base_j][0], 0.0, DEFAULT_SETTINGS) for i in range(n)]
        g, lo, hi = confidence_range(gains)
        wait = mean([m.nonurgent_median_wait for m in ms])
        rows.append({"plan": p, "score": mean(sc), "gain": g, "gain_lo": lo, "gain_hi": hi,
                     "urgent_on_time_pct": 100.0 * mean([m.urgent_share for m in ms]), "nonurgent_median_wait": wait,
                     "nonurgent_change_pct": wait_change_pct(wait, base_wait), "lwbs": mean([m.lwbs for m in ms]),
                     "bed_util_pct": 100.0 * mean([s.area["b"] / s.carea["b"] if s.carea["b"] else 0.0
                                                   for s in (runs[i][j][1] for i in range(n))]),
                     "feasible": fairness_ok(wait, base_wait, DEFAULT_SETTINGS)})
    return rows


def _now_state(snap: EDSim) -> dict:
    waiting = [p for p in snap.pat.values() if p.st in ("tri_q", "bed_q", "doc_q")]
    return {"beds_busy": snap.busy["b"], "beds_cap": snap.cap["b"], "admitted_boarders": snap.board_admitted,
            "waiting": len(waiting), "waiting_urgent": sum(p.ats <= 3 for p in waiting)}


def _late_pct_do_nothing(live: LiveDay, replications: int) -> float:
    snap, t_end = live.snapshot(), live.now + HORIZON_MIN
    ms = [_run(DO_NOTHING, snap, pats, t_end)[0] for pats in forecast_patients(live, replications)]
    return 100.0 * (1.0 - sum(m.urgent_share for m in ms) / len(ms))


def whole_day_verdict(key: str, cfg_name: str) -> Optional[dict]:
    """The offline plan search's whole-day answer for this scenario and hospital, if it has been run."""
    p = RESULTS / f"{key}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if d.get("hospital") not in (None, cfg_name):
        return None
    t = d.get("truth_check", {})
    rng = "" if t.get("gain_lo") is None else f"gain {t['gain']:+.1f} points, 95% range {t['gain_lo']:+.1f} to {t['gain_hi']:+.1f}, {d.get('n_seeds')} simulated days"
    return {"label": d["recommended"]["label"], "is_nothing": d["recommended"]["label"] == "Do nothing",
            "levers": (d["recommended"].get("clinician"), d["recommended"].get("fast_track"), d["recommended"].get("boarding_cut")),
            "start_hour": d["recommended"].get("start_hour"), "why": rng or d.get("reason", "")}


def _differ_note(dec: Decision, live: LiveDay) -> str:
    w = dec.whole_day
    if not w:
        return ""
    mine = (dec.plan.clinician, dec.plan.fast_track, dec.plan.boarding_cut)
    if (w["is_nothing"] and dec.plan.is_nothing) or (not w["is_nothing"] and not dec.plan.is_nothing and tuple(w["levers"]) == mine):
        return ""
    rates = registry.forecast(registry.info(live.key), 0, 24)
    h0 = int(dec.t // 60)
    window = sum(rates[(h0 + i) % 24] for i in range(4)) / 4.0
    day = sum(rates) / 24.0
    busier = "busier" if window > day else "quieter"
    return (f"Why they differ: the next-4-hours view looks only at {_clock(dec.t)}-{_clock(dec.t + HORIZON_MIN)}, which is {busier} than "
            f"the day as a whole (about {window:.1f} vs {day:.1f} expected arrivals per hour), while the whole-day plan must pay off across all 24 hours.")


def decide(live: LiveDay, replications: int = DEFAULT_REPLICATIONS) -> Decision:
    """The plan search's rulebook on the next 4 hours, from the live snapshot (never the true future)."""
    snap, t_end = live.snapshot(), live.now + HORIZON_MIN
    plans = demo_plans(live.now)
    fc = forecast_patients(live, replications)
    runs = [[_run(p, snap, pats, t_end) for p in plans] for pats in fc]           # runs[rep][plan] = (metrics, sim)
    rows = _stats(plans, runs, 0)
    base = rows[0]
    feas = [r for r in rows if r["feasible"]]
    best = min(feas, key=lambda r: (-r["score"], r["plan"].cost, r["plan"].label))
    chosen, rule, reason = DO_NOTHING, "nothing_needed", ""
    if not best["plan"].is_nothing and best["gain_lo"] is not None and best["gain_lo"] > MIN_BENEFIT_PP:
        chosen, rule = best["plan"], "bar"
        reason = f"It clears the plan search's bar (low end of the 95% range above {MIN_BENEFIT_PP:g} points) and the fairness check."
    normal_late = None
    if chosen.is_nothing:
        base_late = 100.0 - base["urgent_on_time_pct"]
        if live.key != "normal":
            ref = LiveDay("normal", live.seed, live.cfg)
            ref.advance_to(live.now)
            normal_late = _late_pct_do_nothing(ref, replications)
        if normal_late is not None and base_late > normal_late + 1e-9:          # rule (c): breaches rise
            helps = sorted([r for r in feas if not r["plan"].is_nothing and r["gain_lo"] is not None and r["gain_lo"] > 0],
                           key=lambda r: (-r["score"], r["plan"].cost))
            if helps:
                best, chosen, rule = helps[0], helps[0]["plan"], "rule_c"
                reason = (f"Rule (c): urgent patients are seen late more often than on a normal day ({base_late:.1f}% vs {normal_late:.1f}%), "
                          f"and this plan's 95% range stays above 0, so 'do nothing' is not allowed (it is below the {MIN_BENEFIT_PP:g}-point bar).")
            else:
                rule = "no_plan_helps"
                reason = (f"No tested plan helps; constraint is {binding_constraint(runs[0][0][1])}. Urgent patients are seen late more "
                          f"often than on a normal day ({base_late:.1f}% vs {normal_late:.1f}%), but every tested plan's 95% range includes 0.")
        else:
            reason = (f"Urgent waits are not above a normal day{'' if normal_late is None else f' ({base_late:.1f}% vs {normal_late:.1f}% late)'}"
                      f" and no plan clears the {MIN_BENEFIT_PP:g}-point bar.")
    after = best if not chosen.is_nothing else base
    dec = Decision(live.now, live.key, chosen, reason, rule, after["gain"] if not chosen.is_nothing else None,
                   after["gain_lo"] if not chosen.is_nothing else None, after["gain_hi"] if not chosen.is_nothing else None,
                   base, after, _now_state(snap), rows, normal_late, whole_day_verdict(live.key, reference_name()))
    dec.differ_note = _differ_note(dec, live)
    return dec


def compute_view(key: str, minute: float, seed: int = 42, workers: int = 1, replications: int = DEFAULT_REPLICATIONS) -> dict:
    """What run.py demo prints. ``workers`` is accepted for CLI compatibility; the 4-hour runs are serial (they are tiny)."""
    live = LiveDay(key, seed)
    live.advance_to(minute)
    dec = decide(live, replications)
    return {"config": live.cfg, "decision": dec, "description": registry.description(key), "label": registry.label(key)}


def rolling_view(key: str, seed: int = 42, start: float = 0.0, until: float = 1440.0, step: float = 30.0,
                 workers: int = 1, replications: int = DEFAULT_REPLICATIONS) -> Iterator[Decision]:
    """Every ``step`` minutes: advance the true day and decide for the next 4 hours (actions are NOT applied)."""
    live = LiveDay(key, seed)
    t = start
    while t <= until + 1e-9:
        live.advance_to(t)
        yield decide(live, replications)
        t += step
