"""Rule-based charge-nurse handover briefing (3 paragraphs, no external services)."""
from __future__ import annotations

from src.optimization.optimizer import Recommendation, pretty
from src.simulation.scenarios import BASE_ACUITY, BASE_VOLUME, Scenario

TIER = {1: "ATS 1 (resuscitation)", 2: "ATS 2 (emergency)", 3: "ATS 3 (urgent)",
        4: "ATS 4 (semi-urgent)", 5: "ATS 5 (non-urgent)"}


def _level(u: float) -> str:
    return "saturated" if u >= 0.9 else "strained" if u >= 0.75 else "manageable"


def build_briefing(sc: Scenario, rec: Recommendation) -> str:
    hh, mm = divmod(int(rec.t), 60)
    daily = sc.expected_daily()
    dev = 100.0 * (daily / BASE_VOLUME - 1.0)
    mid = sc.acuity[1] + sc.acuity[2]
    p1 = (f"SITUATION ({hh % 24:02d}:{mm:02d}): Active scenario is {sc.name}. Expected volume is about "
          f"{daily:.0f} patients per 24h, {dev:+.0f}% against the {BASE_VOLUME}/day baseline"
          + (f", with a {int(round((sc.burst[2] - 1) * 100))}% burst from {sc.burst[0]:02d}:00 to {sc.burst[1]:02d}:00" if sc.burst else "")
          + (f" and a {sc.elderly_surge_pct:.1f}% rise in patients aged 65+" if sc.elderly_surge_pct else "")
          + f". Incoming acuity is {100 * mid:.0f}% ATS 2-3 (baseline {100 * (BASE_ACUITY[1] + BASE_ACUITY[2]):.0f}%) "
          f"and {100 * (sc.acuity[0] + sc.acuity[1]):.0f}% ATS 1-2.")
    b, u, s = rec.base, rec.base["util"], rec.state
    rate = {i: b["breach_by_ats"][i] / b["total_by_ats"][i] for i in range(1, 6) if b["total_by_ats"][i] >= 1}
    worst = max(rate, key=rate.get) if rate else None
    board = ("boarding gridlock: admitted patients are holding beds and blocking flow" if s["boarders"] >= 0.25 * s["beds_cap"]
             else "boarding is elevated" if s["boarders"] >= 0.1 * s["beds_cap"] else "boarding is not a constraint")
    p2 = (f"BOTTLENECKS: Acute beds are {s['beds_busy']}/{s['beds_cap']} occupied now and project to "
          f"{100 * u['bed']:.0f}% utilisation over the next 4h ({_level(u['bed'])}); {board} "
          f"({s['boarders']} boarders). Physicians run at {100 * u['physician']:.0f}% and triage nurses at "
          f"{100 * u['nurse']:.0f}%. Queues now: {s['triage_q']} in triage, {s['provider_q']} awaiting bed/provider. "
          + (f"Without action, {TIER[worst]} is most exposed ({100 * rate[worst]:.0f}% of patients breach target) "
             f"with {b['lwbs']:.1f} expected walkouts (LWBS)." if worst else "No target breaches are projected."))
    if rec.action == "DO_NOTHING":
        p3 = ("INTERVENTIONS: No candidate action reduces the weighted loss enough to justify its cost; "
              "maintain current staffing and re-evaluate at the next 30-minute cycle.")
    else:
        _, lo, hi = rec.breach_reduction_ci
        _, wl, wh = rec.wait_reduction_ci
        p3 = (f"INTERVENTIONS: Recommended: {pretty(rec.action)} at a cost of ${rec.cost:,.0f}. Projected breach "
              f"reduction is {rec.breach_reduction_pct:.0f}% (95% CI {lo:.0f} to {hi:.0f}%), median wait falls "
              f"{rec.wait_reduction:.0f} min (95% CI {wl:.0f} to {wh:.0f}), and expected LWBS drops from "
              f"{b['lwbs']:.1f} to {rec.best['lwbs']:.1f}. Estimates come from 5 paired common-random-number "
              f"forecasts of the next 4 hours.")
    return "\n\n".join((p1, p2, p3))
