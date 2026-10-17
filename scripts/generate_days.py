#!/usr/bin/env python
"""A2 data: simulate whole days and take a snapshot every 30 simulated minutes.

For each snapshot at time t the row holds
  * FEATURES: only what a hospital can see at t (see models/predict_risk.py FEATURES), and
  * LABELS from what happens in the NEXT 4 HOURS of the SAME run under "do nothing":
      y_urgent_late : at least 2 urgent (triage 1-3) patients arriving in the window are seen late
                      (wait over the ATS target, level 1 given a 10-minute grace, or they leave unseen)
                      AND at least 25% of the urgent patients in the window are late      [thresholds: ASSUMPTION]
      y_beds_full   : bed occupancy goes above 95% at any 15-minute sample in the window  [ASSUMPTION]
      y_lwbs        : at least 1 patient leaves without being seen in the window          [ASSUMPTION]
  * first_breach_*: minute of the first real breach of the day (for warning lead time).

Four scenario types: normal, heatwave, hurricane, hurricane_burst. Day number n -> truth seed truth_seed(n); the
look-ahead seeds the optimiser uses (lookahead_seed) are a disjoint range. The hospital size cycles
Small / Medium / Large with n. Output: results/early_warning_snapshots.csv.gz (not committed).

    python scripts/generate_days.py --days 750          (4 x 750 = 3000 days)
    python scripts/generate_days.py --quick             (4 x 60 days)
    python scripts/generate_days.py --benchmark         (timing for 1/2/4/8 workers)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from _common import RESULTS, ROOT  # noqa: F401  (puts the repo on sys.path)

from models.predict_risk import FEATURES, build_features
from src.simulation.ed_simulator import ATS_TARGET
from src.simulation.hospital import PRESETS, prepare, run_day
from src.simulation.scenarios import generate, lookahead_seed, truth_seed

SCENARIOS = [("normal", "normal", False), ("heatwave", "heatwave", False),
             ("hurricane", "hurricane", False), ("hurricane_burst", "hurricane", True)]
PRESET_NAMES = list(PRESETS)
STEP, HORIZON, DAY = 30.0, 240.0, 1440.0
L1_GRACE = 10.0                  # ASSUMPTION: a level-1 patient is "late" after 10 min (the ATS target of 0 is unreachable)
URGENT_MIN_LATE, URGENT_LATE_SHARE = 2, 0.25
BED_FULL = 0.95
OUT = RESULTS / "early_warning_snapshots.csv.gz"


def _target(ats: int) -> float:
    return L1_GRACE if ats == 1 else ATS_TARGET[ats]


def one_day(job: tuple) -> list:
    key, name, burst, n = job
    truth, info = generate(name, truth_seed(n), burst=burst)
    cfg = PRESETS[PRESET_NAMES[n % len(PRESET_NAMES)]]
    pats = prepare(cfg, truth)
    snaps = []

    def on_stop(sim):
        if STEP - 1e-9 <= sim.now <= DAY - HORIZON + 1e-9:
            f = build_features(sim, info, cfg.volume_scale)
            f["t"] = sim.now
            snaps.append(f)

    sim = run_day(cfg, pats, prepared=True, stop_every=STEP, on_stop=on_stop)
    beds = max(sim.cap["b"], 1)
    occ = [(s["t"], s["beds"] / beds) for s in sim.samples if s["t"] > 0]
    # event times for the whole day
    urgent, late_when, lwbs_times = [], [], []
    for p in sim.pat.values():
        if p.st == "future":
            continue
        wait = (p.t_doc - p.t) if p.t_doc is not None else (p.t_lwbs - p.t) if p.t_lwbs is not None else DAY - p.t
        is_late = p.st == "lwbs" or wait > _target(p.ats) + 1e-9
        if p.ats <= 3:
            urgent.append((p.t, is_late))
            if is_late:
                late_when.append(p.t + min(_target(p.ats), wait) if p.st != "lwbs" else p.t_lwbs)
        if p.st == "lwbs":
            lwbs_times.append(p.t_lwbs)
    late_when.sort()
    fb = {"first_breach_urgent_late": late_when[URGENT_MIN_LATE - 1] if len(late_when) >= URGENT_MIN_LATE else float("nan"),
          "first_breach_beds_full": next((t for t, o in occ if o > BED_FULL), float("nan")),
          "first_breach_lwbs": min(lwbs_times) if lwbs_times else float("nan")}
    rows = []
    for f in snaps:
        t = f["t"]
        win = [late for (ta, late) in urgent if t <= ta < t + HORIZON]
        row = {"scenario": key, "day": n, "hospital": PRESET_NAMES[n % len(PRESET_NAMES)], "t": t}
        row.update({k: f[k] for k in FEATURES})
        row["y_urgent_late"] = int(sum(win) >= URGENT_MIN_LATE and win and sum(win) / len(win) >= URGENT_LATE_SHARE)
        row["y_beds_full"] = int(any(t < s_t <= t + HORIZON and o > BED_FULL for s_t, o in occ))
        row["y_lwbs"] = int(any(t < x <= t + HORIZON for x in lwbs_times))
        row.update(fb)
        rows.append(row)
    return rows


def generate_days(days: int, workers: int = 1, out=OUT, quiet: bool = False) -> dict:
    jobs = [(key, name, burst, n) for n in range(days) for key, name, burst in SCENARIOS]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            chunks = list(ex.map(one_day, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
    else:
        chunks = [one_day(j) for j in jobs]
    secs = time.time() - t0
    df = pd.DataFrame([r for c in chunks for r in c])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")
    info = {"days_per_scenario": days, "total_days": len(jobs), "snapshots": len(df), "workers": workers,
            "seconds": round(secs, 2), "seconds_per_day": round(secs / len(jobs), 4), "file": str(out)}
    if not quiet:
        base = {k: round(float(df[df.scenario == k][c].mean()), 3) for k in ("normal", "heatwave", "hurricane", "hurricane_burst")
                for c in ("y_urgent_late",)} if len(df) else {}
        print(f"{len(jobs)} days, {len(df)} snapshots in {secs:.1f}s ({workers} worker(s)); share of snapshots with urgent_late=1 by scenario: {base}")
    return info


def benchmark(days: int = 25, worker_list=(1, 2, 4, 8)) -> list:
    """Same work with 1/2/4/8 workers. Reports honest timing; more workers may not help for small jobs."""
    cpu = os.cpu_count() or 1
    out = []
    for w in worker_list:
        if w > cpu * 2:
            continue
        info = generate_days(days, w, out=RESULTS / "_bench_tmp.csv.gz", quiet=True)
        out.append({"workers": w, "seconds": info["seconds"], "days": info["total_days"]})
    try:
        (RESULTS / "_bench_tmp.csv.gz").unlink()
    except OSError:
        pass
    base = out[0]["seconds"] if out else 1.0
    for r in out:
        r["speedup_vs_1_worker"] = round(base / r["seconds"], 2) if r["seconds"] else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=750, help="days per scenario (4 scenarios)")
    ap.add_argument("--quick", action="store_true", help="60 days per scenario")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    from pathlib import Path
    if a.benchmark:
        print(json.dumps(benchmark(), indent=1))
        return 0
    assert truth_seed(0) != lookahead_seed(0)
    info = generate_days(60 if a.quick else a.days, a.workers, Path(a.out))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
