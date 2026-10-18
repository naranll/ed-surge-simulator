#!/usr/bin/env python
"""Run every scenario on N simulated days for the reference hospital, then the offline plan search.

Writes results/scenario_summary.csv (one row per scenario) and results/plan_search/<scenario>.json (+ .txt table).
Standard library + pandas only. Safe to rerun (files are overwritten, never deleted).

    python scripts/run_scenarios.py --seeds 50           (every registry scenario, full plan grid)
    python scripts/run_scenarios.py --quick              (2 scenarios, 5 seeds, coarse grid)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import time

from _common import RESULTS, ROOT  # noqa: F401  (puts the repo on sys.path)

from src.optimization.optimizer import format_table, plan_search
from src.simulation import patient_flow, registry
from src.simulation.hospital import PRESETS, day_metrics, generate_day, run_day, runner_for
from src.simulation.scenarios import truth_seed

QUICK_REPLICATIONS = 30    # --quick keeps 5 days for the scenario table, but the plan search needs ~30 for meaningful 95% ranges


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--preset", default="Large", choices=list(PRESETS))
    ap.add_argument("--replications", type=int, default=None,
                    help="look-ahead and truth days per plan in the plan search (default: 30 in --quick, --seeds otherwise)")
    ap.add_argument("--workers", type=int, default=1, help="worker processes for the plan search")
    a = ap.parse_args(argv)
    names = ["normal", "heatwave"] if a.quick else registry.names()
    seeds = 5 if a.quick else a.seeds
    cfg = PRESETS[a.preset]
    if cfg.patient_flow and not patient_flow.available():
        import dataclasses
        print("note: models/patient_flow/admission_lookup.json not found -> using the old placeholder patient behaviour")
        cfg = dataclasses.replace(cfg, patient_flow=False)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "plan_search").mkdir(exist_ok=True)
    rows = []
    for name in names:
        t0 = time.time()
        ms = [day_metrics(run_day(cfg, generate_day(name, truth_seed(n))[0]), cfg) for n in range(seeds)]
        t_sim = time.time() - t0
        rows.append({"scenario": name, "hospital": a.preset, "days": seeds,
                     "patients_per_day": round(st.mean(m["patients"] for m in ms), 1),
                     "urgent_late_pct": round(st.mean(m["urgent_late_pct"] for m in ms), 2),
                     "lwbs_pct": round(st.mean(m["lwbs_pct"] for m in ms), 2),
                     "peak_bed_occupancy_pct": round(st.mean(m["peak_bed_occupancy_pct"] for m in ms), 1),
                     "longest_queue": round(st.mean(m["longest_queue"] for m in ms), 1),
                     "ms_per_simulated_day": round(1000 * t_sim / seeds, 2)})
        t0 = time.time()
        reps = a.replications or (QUICK_REPLICATIONS if a.quick else seeds)
        res = plan_search(name, n_seeds=seeds, quick=a.quick, api=registry.api(), runner=runner_for(cfg),
                          replications=reps, workers=a.workers)
        secs = time.time() - t0
        (RESULTS / "plan_search" / f"{name}.json").write_text(json.dumps({**res.to_dict(), "hospital": a.preset, "seconds": round(secs, 1)}, indent=1, default=str), encoding="utf-8")
        (RESULTS / "plan_search" / f"{name}.txt").write_text(format_table(res), encoding="utf-8")
        rows[-1]["recommended_plan"] = res.recommended.label
        rows[-1]["plan_search_seconds"] = round(secs, 1)
        rows[-1]["plan_search_replications"] = reps
        print(f"{name:<10} urgent late {rows[-1]['urgent_late_pct']:5.1f}%  left unseen {rows[-1]['lwbs_pct']:4.1f}%  peak beds {rows[-1]['peak_bed_occupancy_pct']:5.1f}%  "
              f"| recommended: {res.recommended.label}  ({secs:.0f}s search)")
    with open(RESULTS / "scenario_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
