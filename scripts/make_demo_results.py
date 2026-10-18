#!/usr/bin/env python
"""Collect the headline results into ONE file for the demo: results/demo_results.json. Missing pieces are marked, never invented.

    python scripts/make_demo_results.py
"""
from __future__ import annotations

import csv
import json
import time

from _common import MODELS, RESULTS, ROOT  # noqa: F401  (puts the repo on sys.path)


def jload(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def csv_rows(p):
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if any(r.values())]


def main() -> int:
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "caveat": ("Simulation built on national survey data (NHAMCS 2022) and stated assumptions. Not a real hospital, not medical advice. "
                      "The early-warning model learns the simulator and is not validated on a real hospital."),
           "scenarios": csv_rows(RESULTS / "scenario_summary.csv") or "not run yet",
           "calibration_real_vs_simulated": csv_rows(RESULTS / "calibration.csv") or "not run yet",
           "recommendations": {}}
    for name in ("normal", "heatwave", "hurricane"):
        d = jload(RESULTS / "plan_search" / f"{name}.json")
        if d:
            out["recommendations"][name] = {"recommended": d["recommended"]["label"], "reason": d["reason"], "truth_check": d["truth_check"],
                                            "days_tested": d["n_seeds"], "hospital": d.get("hospital")}
    m = jload(MODELS / "metrics.json") or {}
    pf, ew = m.get("patient_flow"), m.get("early_warning")
    out["models"] = {
        "patient_flow": ({"admission_model": pf["admission"]["chosen"]["family"], "admission_test": {k: pf["admission"]["test"][k] for k in ("chosen_model", "baseline_triage_only")},
                          "care_time_test": pf["care_time"]["test"]["chosen_model"], "saved_files": pf["saved_files"]} if pf else "not trained yet"),
        "early_warning": ({"headline_stage3_all_scenarios": ew["headline_stage3_all_scenarios"], "caveat": ew["caveat"]} if ew else "not trained yet")}
    try:
        from models.predict_risk import predict_risk
        from src.simulation.hospital import PRESETS, generate_day, run_day
        from src.simulation.scenarios import truth_seed
        cfg, got = PRESETS["Large"], {}
        truth, info = generate_day("heatwave", truth_seed(0))

        def grab(sim):
            if sim.now == 14 * 60:
                got["r"] = predict_risk(sim, info, cfg.volume_scale)

        run_day(cfg, truth, stop_every=30.0, on_stop=grab)
        out["early_warning_example"] = {"day": "heatwave, Large hospital, 14:00", **got.get("r", {})}
    except Exception as err:
        out["early_warning_example"] = f"not available: {err}"
    lx, local = jload(RESULTS / "linuxone_run.json"), jload(RESULTS / "local_run.json")
    if lx and lx.get("uname_m") != "s390x":     # only a run that really happened on s390x counts as LinuxONE
        out["linuxone_file_rejected"] = f"results/linuxone_run.json says machine {lx.get('uname_m')!r}, not s390x"
        lx = None
    out["linuxone"] = ({k: lx.get(k) for k in ("uname_m", "nproc", "python", "total_seconds", "seconds_per_step", "parallel_timing")} if lx
                       else "No LinuxONE run recorded yet")
    if local and not lx:
        out["local_run_not_linuxone"] = {k: local.get(k) for k in ("uname_m", "nproc", "python", "total_seconds", "parallel_timing")}
    out["limits"] = ["Triage missing in 25.9% of 2022 visits; only 139 level-1 visits; national sample, not one hospital.",
                     "Care time LOV-WAITTIME is a proxy, not service time. ATS time targets applied to NHAMCS triage levels are an assumption.",
                     "Simulated urgent median wait is lower than the survey's (see calibration).",
                     "Hospital sizes, consult times, patience, fast-track effect and plan costs are assumptions."]
    (RESULTS / "demo_results.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote", RESULTS / "demo_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
