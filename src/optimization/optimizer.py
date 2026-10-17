"""Rolling look-ahead optimizer: CRN what-if simulation of candidate actions."""
from __future__ import annotations

import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from src.simulation.ed_simulator import Action, EDSim, LiveED, SimResult, simulate
from src.simulation.scenarios import ATS_WEIGHT, SCENARIOS, Scenario, generate_arrivals

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
    h = _T95.get(len(xs) - 1, 1.96) * statistics.stdev(xs) / math.sqrt(len(xs))
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
              workers: int = 1) -> Recommendation:
    """Evaluate every candidate on identical forecast streams (CRN) and pick the min-loss action.

    The snapshot should come from ``LiveED.snapshot()`` (no future arrivals); the
    forward stream is *forecast* from the scenario rates, never read from the live truth.
    """
    cands = list(candidates or candidate_actions())
    if not any(a.name == "DO_NOTHING" for a in cands):
        cands.insert(0, Action())
    t0, t1 = snapshot.now, snapshot.now + hours * 60.0
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
         "boarders": snapshot.cnt["board"], "beds_busy": snapshot.busy["b"], "beds_cap": snapshot.cap["b"],
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
