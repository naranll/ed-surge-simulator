"""ED Surge Simulator CLI.  python run.py demo|rolling|dashboard

demo / rolling use the calibrated hospital from configs/capacity.json and real-data patient times
(src/demo/live_demo.py). With ED_USE_LEGACY_PARAMS=1 they use the original route (src/dashboard/app.py), unchanged.
"""
import argparse
import subprocess
import sys

from src.simulation import registry
from src.simulation.scenarios import SCENARIOS, USE_LEGACY_PARAMS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["demo", "rolling", "dashboard"], nargs="?", default="demo")
    ap.add_argument("--scenario", choices=registry.names(), default="compound")
    ap.add_argument("--hour", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1, help="worker processes for the forecast runs (1 = serial)")
    ap.add_argument("--replications", type=int, default=30,
                    help="forecast runs per action (default 30; ignored with ED_USE_LEGACY_PARAMS=1, which keeps the old 5)")
    a = ap.parse_args()
    if a.cmd == "dashboard":
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"]))
    if USE_LEGACY_PARAMS:                       # the original behaviour, exactly as before
        if a.scenario not in SCENARIOS:
            sys.exit(f"scenario {a.scenario!r} is not available with ED_USE_LEGACY_PARAMS=1 (old scenarios: {list(SCENARIOS)})")
        from src.dashboard.app import compute_view
        from src.optimization.optimizer import rolling
        from src.simulation.ed_simulator import LiveED
        if a.cmd == "demo":
            v = compute_view(a.scenario, a.hour * 60, a.seed, a.workers)
            print(v["rec"].headline(), "\n\n" + v["briefing"])
        else:
            live, sc = LiveED(SCENARIOS[a.scenario], a.seed), SCENARIOS[a.scenario]
            for r in rolling(live, sc, 0, 24 * 60, 30, workers=a.workers):
                print(f"{int(r.t // 60):02d}:{int(r.t % 60):02d}  {r.action:<45} breach -{r.breach_reduction_pct:3.0f}%  "
                      f"wait -{r.wait_reduction:4.0f}m  cost ${r.cost:,.0f}")
        return
    from src.demo.live_demo import compute_view, rolling_view
    if a.cmd == "demo":
        v = compute_view(a.scenario, a.hour * 60, a.seed, a.workers, a.replications)
        c = v["config"]
        print(f"Scenario: {v['label']}. {v['description']}")
        print(f"Hospital: {c.effective_beds} beds, {c.doctors} doctors, {c.triage_capacity} triage nurses (configs/capacity.json). "
              f"Forecast runs per plan: {a.replications}\n")
        print("\n".join(v["decision"].lines()))
    else:
        for d in rolling_view(a.scenario, a.seed, 0, 24 * 60, 30, workers=a.workers, replications=a.replications):
            rng = "" if d.gain_lo is None else f"gain {d.gain:+5.1f} (95% {d.gain_lo:+.1f} to {d.gain_hi:+.1f})"
            cost = "" if d.plan.is_nothing else f"${d.plan.cost:,.0f} / {d.plan.staff_hours:g} staff-h"
            print(f"{int(d.t // 60):02d}:{int(d.t % 60):02d}  {d.label:<46} {rng:<30} {cost:<18} [{d.rule}]")


if __name__ == "__main__":
    main()
