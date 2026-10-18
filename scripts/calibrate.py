#!/usr/bin/env python
"""Calibrate beds / doctors / triage nurses so a NORMAL Monday is busy but not saturated, then save the
real-vs-simulated table (results/calibration.csv) and the grid (results/calibration_grid.csv).

Targets are computed from NHAMCS 2022 (not assumed): urgent (levels 1-3) median WAITTIME and the LWBS share.
Volume is held at the simulator's 200 patients/day Monday baseline (an ASSUMPTION); only capacity is searched.

    python scripts/calibrate.py            (30 days per setting)
    python scripts/calibrate.py --quick    (8 days per setting)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import itertools
import statistics as st

import numpy as np
import pandas as pd

from _common import NHAMCS_CLEAN, RESULTS, ROOT  # noqa: F401  (also puts the repo on sys.path)

from src.simulation.hospital import HospitalConfig, day_metrics, run_day, waits_by_level
from src.simulation.scenarios import HOURLY, BASE_ACUITY, generate, truth_seed

PEAK_MAX = 95.0       # ASSUMPTION: "not saturated" = average peak bed occupancy under 95%


def wmedian(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def simulate(cfg, days, scenario="normal"):
    rows = []
    for n in range(days):
        truth, _ = generate(scenario, truth_seed(n))
        sim = run_day(cfg, truth)
        m = day_metrics(sim, cfg)
        w = waits_by_level(sim)
        urgent = w[1] + w[2] + w[3]
        m["urgent_median_wait"] = float(np.median(urgent)) if urgent else 0.0
        m["admitted_share"] = float(np.mean([p.admit for p in sim.pat.values()]))
        m["care_by_level"] = {l: [p.consult + p.hold for p in sim.pat.values() if p.ats == l] for l in range(1, 6)}
        m["arrival_hours"] = [int(p.t // 60) for p in sim.pat.values()]
        rows.append(m)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="8 days per setting; never writes configs/capacity.json")
    ap.add_argument("--no-write-capacity", action="store_true", help="full mode, but leave configs/capacity.json as it is")
    a = ap.parse_args(argv)
    days = 8 if a.quick else 30
    RESULTS.mkdir(exist_ok=True)

    raw = pd.read_csv(NHAMCS_CLEAN, low_memory=False,
                      usecols=["IMMEDR_clean", "WAITTIME_minutes", "LOV_minutes", "LWBS", "ADMITHOS", "PATWT", "arrival_hour"])
    valid = raw[raw["IMMEDR_clean"].isin([1, 2, 3, 4, 5])]
    urg = valid[valid["IMMEDR_clean"].isin([1, 2, 3]) & valid["WAITTIME_minutes"].notna()]
    real = {"urgent_median_wait": wmedian(urg["WAITTIME_minutes"], urg["PATWT"]),
            "lwbs_pct": 100 * float(np.average(raw["LWBS"] == 1, weights=raw["PATWT"])),
            "admitted_pct": 100 * float(np.average(valid["ADMITHOS"] == 1, weights=valid["PATWT"]))}
    care = valid[valid["WAITTIME_minutes"].notna() & valid["LOV_minutes"].notna() & (valid["LWBS"] != 1)].copy()
    care["c"] = care["LOV_minutes"] - care["WAITTIME_minutes"]
    care = care[care["c"] > 0]
    real_care = {l: wmedian(care[care["IMMEDR_clean"] == l]["c"], care[care["IMMEDR_clean"] == l]["PATWT"]) for l in range(1, 6)}
    hrs = raw[raw["arrival_hour"].notna()]
    real_hour = {int(h): float(g["PATWT"].sum() / hrs["PATWT"].sum()) for h, g in hrs.groupby(hrs["arrival_hour"].astype(int))}

    grid = []
    grid_spec = {"beds": [35, 40, 45, 50, 55, 60], "doctors": [3, 4, 5, 6, 7], "triage_nurses": [3]}
    for beds, docs, nurses in itertools.product(grid_spec["beds"], grid_spec["doctors"], grid_spec["triage_nurses"]):
        cfg = HospitalConfig(beds=beds, doctors=docs, triage_nurses=nurses, patient_flow=True)
        r = simulate(cfg, days)
        g = {"beds": beds, "doctors": docs, "triage_nurses": nurses,
             "urgent_median_wait_min": st.mean(x["urgent_median_wait"] for x in r),
             "lwbs_pct": st.mean(x["lwbs_pct"] for x in r),
             "peak_bed_occupancy_pct": st.mean(x["peak_bed_occupancy_pct"] for x in r),
             "urgent_late_pct": st.mean(x["urgent_late_pct"] for x in r)}
        g["loss"] = (abs(g["urgent_median_wait_min"] - real["urgent_median_wait"]) / real["urgent_median_wait"]
                     + abs(g["lwbs_pct"] - real["lwbs_pct"]) / real["lwbs_pct"]
                     + (10.0 if g["peak_bed_occupancy_pct"] > PEAK_MAX else 0.0))
        grid.append(g)
    with open(RESULTS / "calibration_grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0]))
        w.writeheader()
        w.writerows(grid)
    best = min(grid, key=lambda g: (g["loss"], g["beds"] + g["doctors"]))
    cfg = HospitalConfig(beds=best["beds"], doctors=best["doctors"], triage_nurses=best["triage_nurses"], patient_flow=True)
    r = simulate(cfg, days)

    sim_care = {l: float(np.median([c for x in r for c in x["care_by_level"][l]] or [0])) for l in range(1, 6)}
    hours = [h for x in r for h in x["arrival_hours"]]
    sim_hour = {h: hours.count(h) / len(hours) for h in range(24)}
    rows = [
        ("urgent (levels 1-3) median wait, min", real["urgent_median_wait"], st.mean(x["urgent_median_wait"] for x in r), "NHAMCS WAITTIME vs simulated door-to-doctor"),
        ("left without being seen, %", real["lwbs_pct"], st.mean(x["lwbs_pct"] for x in r), "NHAMCS LWBS flag vs simulated (patience is a fixed 180 min ASSUMPTION)"),
        ("admitted to hospital, % of triaged visits", real["admitted_pct"], 100 * st.mean(x["admitted_share"] for x in r), "ADMITHOS vs simulated admission draw"),
        ("peak bed occupancy, % (design target, not data)", "", st.mean(x["peak_bed_occupancy_pct"] for x in r), f"ASSUMPTION: keep under {PEAK_MAX:.0f}% on a normal day"),
        ("peak arrival hour share, % (10:00)", 100 * real_hour[10], 100 * sim_hour[10], "NHAMCS ARRTIME vs simulated arrivals"),
        ("quietest arrival hour share, % (05:00)", 100 * real_hour[5], 100 * sim_hour[5], "NHAMCS ARRTIME vs simulated arrivals"),
    ] + [(f"median care time level {l}, min", real_care[l], sim_care[l], "LOV-WAITTIME (proxy) vs simulated consult + bed stay") for l in range(1, 6)]
    with open(RESULTS / "calibration.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "real_NHAMCS_2022", "simulated_normal_day", "note"])
        for q, a_, b_, n_ in rows:
            w.writerow([q, "" if a_ == "" else round(a_, 2), round(b_, 2), n_])
        w.writerow([])
        w.writerow(["chosen capacity (ASSUMPTION set by calibration)", f"beds={best['beds']}", f"doctors={best['doctors']}", f"triage nurses={best['triage_nurses']}; {days} simulated days"])
    print(f"chosen: beds {best['beds']}, doctors {best['doctors']}, triage nurses {best['triage_nurses']}")
    if not a.quick and not a.no_write_capacity:
        write_capacity(best, rows, days, grid_spec)
    for q, a_, b_, n_ in rows:
        print(f"  {q:<48} real {a_ if a_ == '' else round(a_, 2)!s:>8}  sim {round(b_, 2)!s:>8}")
    return 0


def write_capacity(best: dict, rows: list, days: int, grid_spec: dict) -> None:
    """The single capacity source of truth. Keeps the preset volume shares already in the file."""
    from src.simulation import capacity
    path = capacity.path()
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    shares = old.get("preset_volume_shares") or {
        "values": {"Small": 0.3, "Medium": 0.6, "Large": 1.0},
        "source": "UNSOURCED: illustrative volume shares (30% / 60% / 100% of the reference). Small and Medium are NOT calibrated."}
    data = {
        "schema_version": 1,
        "reference_hospital": {"beds": int(best["beds"]), "doctors": int(best["doctors"]), "triage_nurses": int(best["triage_nurses"]),
                               "preset_name": "Large"},
        "patients_per_day": {"value": 200, "source": "UNSOURCED: scenarios.BASE_VOLUME, a Monday assumption (NHAMCS has no per-ED volume)"},
        "source": ("Calibrated by scripts/calibrate.py (full mode) against NHAMCS 2022: urgent (triage 1-3) median WAITTIME and the LWBS "
                   "share, PATWT-weighted, with real-data admission and care times (models/patient_flow/admission_lookup.json). "
                   "NHAMCS has no bed or staff counts, so capacity is FITTED, not measured."),
        "method": {"simulated_days_per_setting": days, "search_grid": grid_spec,
                   "search_grid_source": "UNSOURCED: candidate range chosen by the team",
                   "not_saturated_rule": f"average peak bed occupancy under {PEAK_MAX:.0f}% (UNSOURCED design rule)",
                   "loss": "|sim urgent median wait - real| / real + |sim LWBS - real| / real (+10 if saturated)"},
        "fit": [{"quantity": q, "real_NHAMCS_2022": (None if a_ == "" else round(a_, 2)), "simulated": round(b_, 2)} for q, a_, b_, _ in rows[:3]],
        "preset_volume_shares": shares,
        "written": time.strftime("%Y-%m-%d"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
