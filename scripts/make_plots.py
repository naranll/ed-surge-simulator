#!/usr/bin/env python
"""Slide assets from whatever result files exist: PNG charts + results/for_slides/key_numbers.md.

Missing inputs are skipped with a message (never an error). Charts use colour-blind-safe colours.

    python scripts/make_plots.py
"""
from __future__ import annotations

import json
import shutil

from _common import MODELS, RESULTS, ROOT  # noqa: F401  (puts the repo on sys.path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = RESULTS / "for_slides"
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "grey": "#777777"}


def jload(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)
    print("wrote", OUT / name)


def arrivals_chart():
    from src.simulation.forecaster import forecast
    from src.simulation.scenarios import scenario_info
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, col in (("normal", C["blue"]), ("heatwave", C["red"]), ("hurricane", C["orange"])):
        ax.plot(range(24), forecast(scenario_info(name), 0, 24), label=name, color=col, lw=2)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("expected patients per hour (200/day reference)")
    ax.set_title("Expected arrivals per hour. Hour shape: NHAMCS 2022; surge sizes: published studies")
    ax.legend()
    save(fig, "hourly_arrivals_per_scenario.png")


def calibration_chart():
    p = RESULTS / "calibration.csv"
    if not p.exists():
        return print("skip calibration chart (results/calibration.csv missing)")
    d = pd.read_csv(p).dropna(subset=["real_NHAMCS_2022"])
    d = d[pd.to_numeric(d["real_NHAMCS_2022"], errors="coerce").notna()].copy()
    d["real"] = pd.to_numeric(d["real_NHAMCS_2022"])
    d["sim"] = pd.to_numeric(d["simulated_normal_day"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(d))
    ax.bar([i - 0.2 for i in x], d["real"], 0.4, label="real (NHAMCS 2022)", color=C["blue"])
    ax.bar([i + 0.2 for i in x], d["sim"], 0.4, label="simulated normal day", color=C["orange"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([q.replace(", ", "\n")[:34] for q in d["quantity"]], rotation=60, ha="right", fontsize=7)
    ax.set_yscale("symlog")
    ax.set_title("Calibration: real vs simulated (log scale)")
    ax.legend()
    save(fig, "calibration_real_vs_simulated.png")


def waits_chart():
    rows = []
    for name in ("normal", "heatwave", "hurricane"):
        d = jload(RESULTS / "plan_search" / f"{name}.json")
        if d:
            t = d["truth_check"]
            rows.append((name, t["base_on_time_pct"], t["plan_on_time_pct"] if d["recommended"]["label"] != "Do nothing" else t["base_on_time_pct"],
                         d["recommended"]["label"]))
    if not rows:
        return print("skip before/after chart (results/plan_search missing)")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], [r[1] for r in rows], 0.4, label="do nothing", color=C["red"])
    ax.bar([i + 0.2 for i in x], [r[2] for r in rows], 0.4, label="recommended plan (or do nothing if none cleared the gate)", color=C["green"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r[0]}\n{r[3][:30]}" for r in rows], fontsize=8)
    ax.set_ylabel("urgent patients seen on time (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Before vs after on the same simulated days")
    ax.legend(fontsize=8)
    save(fig, "waits_before_after.png")


def timing_chart():
    rec = jload(RESULTS / "linuxone_run.json")
    if rec and rec.get("uname_m") != "s390x":     # a file with this name from another machine is not a LinuxONE result
        rec = None
    label = "LinuxONE"
    if not rec:
        rec, label = jload(RESULTS / "local_run.json"), "LOCAL (not LinuxONE)"
    if not rec or not rec.get("parallel_timing"):
        return print("skip timing chart (no run json yet)")
    t = rec["parallel_timing"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(r["workers"]) for r in t], [r["seconds"] for r in t], color=C["purple"])
    for i, r in enumerate(t):
        ax.text(i, r["seconds"], f"{r['speedup_vs_1_worker']}x", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("workers")
    ax.set_ylabel("seconds for the same data generation")
    ax.set_title(f"Parallel timing on {label}: {rec.get('uname_m')}, {rec.get('nproc')} cores")
    save(fig, "timing_workers.png")


def key_numbers():
    m = jload(MODELS / "metrics.json") or {}
    L = ["# Key numbers (auto-written by scripts/make_plots.py)", "",
         "Every number below comes from a file in this repo. Sources: NHAMCS 2022 (CDC public-use file) for patient behaviour; published studies for surge sizes "
         "(heatwave +6% to +13%, hurricane +9% to +13%, as cited by the team, not independently checked). Weather, FEMA and hurricane records are context only.", ""]
    pf = m.get("patient_flow")
    if pf:
        t, c = pf["admission"]["test"], pf["care_time"]["test"]
        L += ["## Model 1 - patient flow (real data)",
              f"- Visits used: {pf['data']['rows_used']:,} of {pf['data']['rows_all']:,} (valid triage 1-5), {pf['data']['hospitals']} hospitals; held-out hospitals for testing.",
              f"- Admission: {pf['admission']['chosen']['family']}; AUC {t['chosen_model']['auc_roc']:.3f} (95% {t['chosen_model']['ci95']['auc_roc'][0]:.3f}-{t['chosen_model']['ci95']['auc_roc'][1]:.3f}) vs triage-only baseline {t['baseline_triage_only']['auc_roc']:.3f}.",
              f"- Care time (log, proxy LOV-WAITTIME): R-squared {c['chosen_model']['r2']:.2f}; error {c['chosen_model']['mae_log']:.3f} vs baseline {c['baseline_triage_median_only']['mae_log']:.3f}.", ""]
    ew = m.get("early_warning")
    if ew:
        L += ["## Model 2 - early warning (simulated days; learns the simulator)",
              f"- {ew['data']['days_per_scenario']} day numbers x 4 scenario types, {ew['data']['snapshots']:,} snapshots; split by whole day ({ew['data']['train_days']}/{ew['data']['val_days']}/{ew['data']['test_days']})."]
        for tg, v in ew.get("headline_stage3_all_scenarios", {}).items():
            L.append(f"- {tg}: AUC {v.get('auc_roc', float('nan')):.3f}; catches {100 * v.get('recall_at_5pct_fpr', float('nan')):.0f}% of risky moments at 5% false alarms on normal days "
                     f"(simple rule: {100 * v.get('rule_recall_at_5pct_fpr', float('nan')):.0f}%).")
        L.append("")
    cal = RESULTS / "calibration.csv"
    if cal.exists():
        L += ["## Calibration (real vs simulated normal day)", "```", cal.read_text(encoding="utf-8").strip(), "```", ""]
    ss = RESULTS / "scenario_summary.csv"
    if ss.exists():
        L += ["## Scenarios (reference hospital)", "```", ss.read_text(encoding="utf-8").strip(), "```", ""]
    L += ["## Limits and honesty",
          "- Triage level is missing in 25.9% of 2022 visits; wait time in 17.2%. Only 139 level-1 visits. National sample (about 40% response rate), not one hospital.",
          "- Care time LOV-WAITTIME is a proxy (includes tests and boarding), not pure service time. ATS time targets are applied to NHAMCS triage levels as an assumption.",
          "- Hospital sizes, doctor consult times, patience (180 min), fast-track effect, breach thresholds and plan costs are ASSUMPTIONS (see the parameter table in the dashboard).",
          "- The early-warning model learns the simulator, which rests on stated assumptions; it is not validated on a real hospital.",
          "- The 'urgent median wait' of the simulator is lower than the survey's (see calibration table); we report this gap rather than hide it."]
    lx = jload(RESULTS / "linuxone_run.json")
    if lx and lx.get("uname_m") != "s390x":
        lx = None
    L += ["", "## LinuxONE", (f"- Recorded run: {lx.get('uname_m')}, {lx.get('nproc')} cores, {lx.get('total_seconds')} s total." if lx else "- No LinuxONE run recorded yet.")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "key_numbers.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT / "key_numbers.md")


def main() -> int:
    for fn in (arrivals_chart, calibration_chart, waits_chart, timing_chart, key_numbers):
        try:
            fn()
        except Exception as err:      # one missing input must not stop the others
            print(f"skip {fn.__name__}: {err}")
    sc = RESULTS / "stage_chart.png"
    if sc.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy(sc, OUT / "stage_results.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
