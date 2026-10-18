#!/usr/bin/env python
"""A2 - early-warning risk model trained on SIMULATED days (staged).

Learns, from the state a hospital can see at time t, whether in the NEXT 4 HOURS (a) urgent patients will be
seen late, (b) beds will go above 95% full, (c) patients will leave unseen. Honest scope: this model learns the
SIMULATOR, which rests on stated assumptions. It is a fast early-warning surrogate, not validated on a real hospital.

Stages (results in models/stage_results.csv and results/stage_chart.png):
  1  train on NORMAL days only, test on held-out normal days
  2  test that same model on UNSEEN surge days (expect worse; the table shows by how much)
  3  retrain on normal + surge days, test on held-out days of every scenario, and leave-one-scenario-out
  4  models/predict_risk.py: predict_risk(snapshot, info) -> three risks + "breaks first" + a plain line

Splits are by whole simulated DAY (a day number is never in two splits, in any scenario) and stratified by
which scenarios had a breach. Alarm line = the score that gives 5% false alarms on NORMAL validation days.

    python scripts/train_early_warning.py          (about 3-6 minutes on 3,000 days)
    python scripts/train_early_warning.py --quick  (a few hundred days, under 2 minutes)
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from _common import MODELS, RESULTS, ROOT, update_json, update_md_section  # noqa: F401
import train_patient_flow as a1                      # reuse the same candidates, metrics and selection rule

from models.predict_risk import FEATURES, TARGETS
from src.simulation.scenarios import lookahead_seed, truth_seed

warnings.filterwarnings("ignore")
SEED = 20261018
from src.simulation import registry  # noqa: E402
SURGE = [k for k in registry.names() if k != "normal"]   # compound is only ever a TEST scenario in leave-one-out
ALL = ["normal"] + SURGE
FPR = 0.05
SENTENCE = ("This model learns the SIMULATOR, which rests on stated assumptions. It is a fast early-warning surrogate, "
            "not validated on a real hospital.")
PORTABLE = ("logistic", "xgb")        # HistGradientBoosting is tried but cannot be saved in a portable file


# ------------------------------------------------------------------ splits
def make_split(df: pd.DataFrame) -> dict:
    """Day numbers -> train / val / test (60/20/20), stratified by hospital size and which scenarios had a breach."""
    from sklearn.model_selection import StratifiedKFold
    any_breach = (df[["y_urgent_late", "y_beds_full", "y_lwbs"]].max(axis=1) > 0)
    day = df.assign(b=any_breach).groupby(["day", "scenario"])["b"].max().unstack("scenario").reindex(columns=ALL).fillna(False)
    code = sum(day[s].astype(int) * (2 ** i) for i, s in enumerate(ALL)) + 16 * (day.index.to_numpy() % 3)
    code = pd.Series(code, index=day.index)
    counts = code.value_counts()
    strata = code.map(lambda c: c if counts[c] >= 5 else -1).to_numpy()
    days = day.index.to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(days, strata))
    test, val = days[folds[0][1]], days[folds[1][1]]
    train = np.setdiff1d(days, np.concatenate([test, val]))
    return {"train": sorted(map(int, train)), "val": sorted(map(int, val)), "test": sorted(map(int, test))}


def check_split(split: dict) -> None:
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    assert not (tr & va or tr & te or va & te), "a day number is in two splits"
    look = {lookahead_seed(n) for n in (tr | va | te)}
    truth = {truth_seed(n) for n in (tr | va | te)}
    assert not (look & truth), "truth seeds overlap look-ahead seeds"


# ------------------------------------------------------------------ evaluation
def recall_at(p, y, thr):
    pos = y == 1
    return float((p[pos] >= thr).mean()) if pos.any() else float("nan")


def norm_threshold(score_val_normal, y_val_normal) -> float:
    """Alarm line: the score that only 5% of NEGATIVE normal-day validation snapshots reach."""
    neg = score_val_normal[y_val_normal == 0]
    return float(np.quantile(neg, 1 - FPR)) if len(neg) else 0.5


def lead_stats(sub: pd.DataFrame, p: np.ndarray, thr: float, target: str) -> tuple:
    """(median lead minutes, share of breach days warned ahead) for days that really had this breach."""
    col = f"first_breach_{target.replace('y_', '')}"
    leads, warned, n_b = [], 0, 0
    for (_, _), g in sub.assign(_p=p).groupby(["day", "scenario"]):
        b = g[col].iloc[0]
        if math.isnan(b):
            continue
        n_b += 1
        al = g.loc[g["_p"] >= thr, "t"]
        if len(al) and al.min() <= b:
            warned += 1
            leads.append(b - al.min())
    return (float(np.median(leads)) if leads else float("nan"), warned / n_b if n_b else float("nan"), n_b)


def evaluate(stage, train_on, test_on, target, sub, p, rule, thr, rule_thr, n_boot, normal_sub=None, normal_p=None, rule_normal=None):
    y = sub[f"y_{target}"].to_numpy()
    groups = sub["day"].to_numpy()
    row = {"stage": stage, "train_on": train_on, "test_on": test_on, "target": target,
           "snapshots": int(len(sub)), "positives": int(y.sum()), "alarm_line": thr}
    if y.sum() == 0 or y.sum() == len(y):
        row["note"] = "only one class in this test set"
        return row

    def metrics(idx):
        yy, pp = y[idx], p[idx]
        m = a1.clf_metrics(yy, pp, np.ones(len(yy)))
        m["recall_at_5pct_fpr"] = recall_at(pp, yy, thr)
        return m

    m = metrics(np.arange(len(y)))
    ci = a1.cluster_bootstrap(metrics, groups, n_boot, SEED)
    for k, v in m.items():
        row[k] = v
        row[f"{k}_lo"], row[f"{k}_hi"] = ci[k]
    row["rule_recall_at_5pct_fpr"] = recall_at(rule, y, rule_thr)
    from sklearn.metrics import roc_auc_score
    row["rule_auc_roc"] = float(roc_auc_score(y, rule))
    lead, warned, n_b = lead_stats(sub, p, thr, target)
    row["median_lead_min"], row["share_breach_days_warned"], row["breach_days"] = lead, warned, n_b
    rlead, rwarned, _ = lead_stats(sub, rule, rule_thr, target)
    row["rule_median_lead_min"], row["rule_share_breach_days_warned"] = rlead, rwarned
    if normal_sub is not None:       # false alarms per normal test day (snapshots; 40 per day)
        neg = normal_sub[f"y_{target}"].to_numpy() == 0
        fa = ((normal_p >= thr) & neg)
        row["false_alarm_snapshots_per_normal_day"] = float(fa.sum() / normal_sub["day"].nunique())
        row["rule_false_alarm_snapshots_per_normal_day"] = float(((rule_normal >= rule_thr) & neg).sum() / normal_sub["day"].nunique())
    return row


def rule_score(df: pd.DataFrame, target: str) -> np.ndarray:
    """Simple baseline score. urgent: urgent queue length (levels 1-3). beds: bed occupancy. lwbs: longest wait now."""
    if target == "urgent_late":
        return (df["queue_l1"] + df["queue_l2"] + df["queue_l3"]).to_numpy(float)
    if target == "beds_full":
        return df["bed_occupancy"].to_numpy(float)
    return df[[f"longest_wait_l{l}" for l in range(1, 6)]].max(axis=1).to_numpy(float)


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=str(RESULTS / "early_warning_snapshots.csv.gz"))
    ap.add_argument("--out", default=str(MODELS))
    ap.add_argument("--days", type=int, default=750, help="days per scenario, if the data must be generated")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    t_start = time.time()
    try:
        import sklearn, xgboost  # noqa: F401
    except ImportError as err:
        print(f"SKIPPED A2: a machine-learning library is missing ({err}).")
        return 2
    out = Path(a.out)
    ew = out / "early_warning"
    ew.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    if not data.exists():
        import generate_days
        generate_days.generate_days(60 if a.quick else a.days, a.workers, data)
    df = pd.read_csv(data).reset_index(drop=True)
    n_cfg, n_boot, folds, budget = (6, 30, 3, 25.0) if a.quick else (12, 100, 3, 70.0)
    split = make_split(df)
    check_split(split)
    (ew / "splits.json").write_text(json.dumps({**split, "truth_seed": "100000 + day", "lookahead_seed": "200000 + day (disjoint)"}), encoding="utf-8")
    part = {k: df["day"].isin(v).to_numpy() for k, v in split.items()}
    scen = df["scenario"].to_numpy()
    X = df[list(FEATURES)]
    print(f"days {df['day'].nunique()} x {df['scenario'].nunique()} scenarios | snapshots {len(df)} | "
          f"train/val/test days {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")

    def mask(split_name, scens):
        return part[split_name] & np.isin(scen, scens)

    # ---- choose settings once per target on the stage-3 training days (every 2nd snapshot, for speed)
    chosen = {}
    sel_rows = mask("train", ALL) & (df["t"].to_numpy() % 60 == 0)
    sid = pd.Series(scen).map({s: i for i, s in enumerate(ALL)}).to_numpy()
    for tg in TARGETS:
        y = df[f"y_{tg}"].to_numpy()
        ch, best, res, secs = a1.cv_select(X[sel_rows], y[sel_rows], np.ones(sel_rows.sum()), df["day"].to_numpy()[sel_rows],
                                           (sid * 2 + y)[sel_rows], "clf", n_cfg, budget, folds)
        port = [r for r in res if r["family"] in PORTABLE]
        pbest = min(port, key=lambda r: r["cv_mean"])
        pick = min([r for r in port if r["cv_mean"] <= pbest["cv_mean"] + pbest["cv_se"]], key=lambda r: (a1.COMPLEXITY[r["family"]], r["cv_mean"]))
        chosen[tg] = {"picked": pick, "best_any_family": best, "configs_tried": len(res), "search_seconds": secs,
                      "cv_table": res}
        print(f"[{tg}] tried {len(res)} configs in {secs}s; best any family {best['family']}; deployable pick {pick['family']} {pick['params']}")

    def fit_target(tg, rows):
        sp = chosen[tg]["picked"]
        return a1.fit(a1.make_model(sp["family"], sp["params"], "clf"), X[rows], df[f"y_{tg}"].to_numpy()[rows], np.ones(rows.sum()))

    def run_stage(stage, train_scens, tests, tag, models=None):
        rows_out = []
        models = models or {tg: fit_target(tg, mask("train", train_scens)) for tg in TARGETS}
        for tg in TARGETS:
            m = models[tg]
            vm = mask("val", ["normal"])
            pv = a1.predict(m, X[vm], "clf")
            thr = norm_threshold(pv, df[f"y_{tg}"].to_numpy()[vm])
            rthr = norm_threshold(rule_score(df[vm], tg), df[f"y_{tg}"].to_numpy()[vm])
            nm = mask("test", ["normal"])
            pn = a1.predict(m, X[nm], "clf")
            rn = rule_score(df[nm], tg)
            for label, scens in tests:
                tm = mask("test", scens)
                sub = df[tm]
                rows_out.append(evaluate(stage, train_scens if isinstance(train_scens, str) else "+".join(train_scens),
                                         label, tg, sub, a1.predict(m, X[tm], "clf"), rule_score(sub, tg), thr, rthr, n_boot,
                                         df[nm], pn, rn))
        return rows_out, models

    results = []
    r1, m_s1 = run_stage("1", ["normal"], [("normal", ["normal"])], "s1")
    results += r1
    # stage 2 = the SAME stage-1 models on unseen surge days
    r2, _ = run_stage("2", ["normal"], [(s, [s]) for s in SURGE], "s2", models=m_s1)
    results += r2
    r3, m_s3 = run_stage("3", ALL, [(s, [s]) for s in ALL] + [("all scenarios", ALL)], "s3")
    results += r3
    rl1, _ = run_stage("3-leave-out", ["normal", "heatwave"], [("hurricane", ["hurricane"]), ("hurricane_burst", ["hurricane_burst"])], "loso1")
    rl2, _ = run_stage("3-leave-out", ["normal", "hurricane", "hurricane_burst"], [("heatwave", ["heatwave"])], "loso2")
    results += rl1 + rl2
    res = pd.DataFrame(results)
    s1auc = res[(res.stage == "1")].set_index("target")["auc_roc"].to_dict()
    res["auc_change_vs_stage1"] = res.apply(lambda r: r["auc_roc"] - s1auc.get(r["target"], float("nan")) if "auc_roc" in r and pd.notna(r.get("auc_roc")) else float("nan"), axis=1)
    res.to_csv(out / "stage_results.csv", index=False)
    print("stage table written;", len(res), "rows")

    # ---- stage 4: save the deployable (stage-3) models in portable files
    meta = {"features": list(FEATURES), "models": {}, "alarm_thresholds": {}, "false_alarm_rate_on_normal_days": FPR,
            "horizon_minutes": 240, "caveat": SENTENCE}
    vm = mask("val", ["normal"])
    for tg in TARGETS:
        m, sp = m_s3[tg], chosen[tg]["picked"]
        meta["alarm_thresholds"][tg] = norm_threshold(a1.predict(m, X[vm], "clf"), df[f"y_{tg}"].to_numpy()[vm])
        if sp["family"] == "logistic":
            sc, est = m.steps[0][1], m.steps[1][1]
            fn = f"{tg}_logistic.json"
            (ew / fn).write_text(json.dumps({"kind": "linear", "features": list(FEATURES), "mean": sc.mean_.tolist(),
                                             "scale": sc.scale_.tolist(), "coef": np.ravel(est.coef_).tolist(),
                                             "intercept": float(np.ravel(est.intercept_)[0])}), encoding="utf-8")
            meta["models"][tg] = {"kind": "linear", "file": fn, "params": sp["params"]}
        else:
            fn = f"{tg}_xgboost.json"
            m.get_booster().save_model(str(ew / fn))
            meta["models"][tg] = {"kind": "xgboost", "file": fn, "params": sp["params"]}
    (ew / "early_warning_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    # ---- headline numbers + card + chart
    head = res[(res.stage == "3") & (res.test_on == "all scenarios")].set_index("target")
    summary = {"data": {"days_per_scenario": int(df["day"].nunique()), "snapshots": int(len(df)),
                        "train_days": len(split["train"]), "val_days": len(split["val"]), "test_days": len(split["test"])},
               "chosen": {tg: {k: v for k, v in chosen[tg].items() if k != "cv_table"} for tg in TARGETS},
               "headline_stage3_all_scenarios": {tg: {k: (None if pd.isna(head.loc[tg].get(k)) else float(head.loc[tg][k]))
                                                      for k in ("auc_roc", "auc_roc_lo", "auc_roc_hi", "pr_auc", "brier", "ece",
                                                                "recall_at_5pct_fpr", "rule_recall_at_5pct_fpr", "median_lead_min",
                                                                "false_alarm_snapshots_per_normal_day") if k in head.columns}
                                                 for tg in TARGETS if tg in head.index},
               "alarm_thresholds": meta["alarm_thresholds"], "caveat": SENTENCE,
               "runtime_seconds": round(time.time() - t_start, 1)}
    update_json(out / "metrics.json", "early_warning", summary)
    write_card(out / "model_card.md", res, summary, chosen)
    make_chart(res, RESULTS / "stage_chart.png")
    print(f"done in {summary['runtime_seconds']}s")
    return 0


def make_chart(res: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    col = {"1": "#0072B2", "2": "#D55E00", "3": "#009E73", "3-leave-out": "#CC79A7"}   # Okabe-Ito, colour-blind safe
    lab = {"1": "Stage 1: train normal, test normal", "2": "Stage 2: same model on unseen surge",
           "3": "Stage 3: trained on normal + surge", "3-leave-out": "Stage 3: scenario type never seen"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, tg in zip(axes, TARGETS):
        d = res[(res.target == tg) & res.get("auc_roc").notna() & (res.test_on != "all scenarios")].reset_index(drop=True)
        for i, r in d.iterrows():
            ax.bar(i, r["auc_roc"], color=col[r["stage"]], yerr=[[r["auc_roc"] - r["auc_roc_lo"]], [r["auc_roc_hi"] - r["auc_roc"]]],
                   capsize=2, label=lab[r["stage"]] if lab[r["stage"]] not in ax.get_legend_handles_labels()[1] else None)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([f"{r.test_on}" for r in d.itertuples()], rotation=60, ha="right", fontsize=8)
        ax.set_title(tg.replace("_", " "))
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    axes[0].set_ylabel("AUC-ROC (higher is better; 0.5 = coin flip)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Early-warning model on simulated days (95% ranges from resampling days). Learns the simulator, not a real hospital.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_card(path: Path, res: pd.DataFrame, s: dict, chosen: dict) -> None:
    def g(stage, test, tg, k):
        r = res[(res.stage == stage) & (res.test_on == test) & (res.target == tg)]
        return float(r.iloc[0][k]) if len(r) and k in r.columns and pd.notna(r.iloc[0][k]) else float("nan")

    lines = []
    for tg in TARGETS:
        lines.append(f"- **{tg.replace('_', ' ')}**: stage 1 AUC {g('1', 'normal', tg, 'auc_roc'):.3f} (normal days) -> stage 2 on heatwave "
                     f"{g('2', 'heatwave', tg, 'auc_roc'):.3f}, hurricane {g('2', 'hurricane', tg, 'auc_roc'):.3f}, hurricane+burst "
                     f"{g('2', 'hurricane_burst', tg, 'auc_roc'):.3f} -> stage 3 (all scenarios) {g('3', 'all scenarios', tg, 'auc_roc'):.3f}. "
                     f"Pooled over all four scenario types (mostly surge days), the alarm line set at 5% false alarms on NORMAL days catches "
                     f"{100 * g('3', 'all scenarios', tg, 'recall_at_5pct_fpr'):.0f}% of risky moments (the simple rule: "
                     f"{100 * g('3', 'all scenarios', tg, 'rule_recall_at_5pct_fpr'):.0f}%). "
                     f"Leave-one-scenario-out AUC: hurricane {g('3-leave-out', 'hurricane', tg, 'auc_roc'):.3f}, heatwave {g('3-leave-out', 'heatwave', tg, 'auc_roc'):.3f}.")
    honest = []
    for tg in TARGETS:
        mean = lambda stage, k: float(np.nanmean([g(stage, sc, tg, k) for sc in SURGE]))
        a1_, a2_ = g("1", "normal", tg, "auc_roc"), mean("2", "auc_roc")
        honest.append(
            f"- {tg.replace('_', ' ')}: ranking quality (AUC) {a1_:.3f} on normal days vs {a2_:.3f} on unseen surge days "
            f"({'worse' if a2_ < a1_ - 0.01 else 'NOT worse'}). Catch rate at the alarm line: {100 * g('1', 'normal', tg, 'recall_at_5pct_fpr'):.0f}% (normal days) "
            f"-> {100 * mean('2', 'recall_at_5pct_fpr'):.0f}% (surge days, stage-1 model) -> {100 * mean('3', 'recall_at_5pct_fpr'):.0f}% (surge days, retrained). "
            f"On normal days the model catches {100 * g('3', 'normal', tg, 'recall_at_5pct_fpr'):.0f}% vs {100 * g('3', 'normal', tg, 'rule_recall_at_5pct_fpr'):.0f}% for the simple rule.")
    d = s["data"]
    text = f"""## Model 2 - early warning (simulated days)

**What it does, in plain words.** Every 30 simulated minutes it looks at what a charge nurse could see (queues, longest waits, bed use, who is on shift, the hour,
and how many patients the forecast expects) and says how likely it is that in the next 4 hours (1) urgent patients will be seen late, (2) beds will be over 95% full,
(3) patients will walk out unseen. "Breaks first" is the risk that is highest compared with its own alarm line.

**{SENTENCE}**

**Data.** {d['days_per_scenario']} day numbers x 4 scenario types (normal, heatwave, hurricane, hurricane with a burst) = {d['days_per_scenario'] * 4} simulated days,
{d['snapshots']:,} snapshots; hospital size cycles Small / Medium / Large. Labels come from what happens next in the same run under "do nothing".
Thresholds that define a breach are ASSUMPTIONS (urgent late: at least 2 late urgent patients and at least 25% of urgent patients in the window; beds: occupancy over 95%; leaving: at least 1).

**Honest splits.** By whole day number: {d['train_days']} train / {d['val_days']} validation / {d['test_days']} test days. A day number is never in two splits (in any scenario),
and the seed range for the optimiser's look-ahead runs (200000+) never overlaps the truth seeds (100000+). Features never use the true day or the future (a test checks the feature list).
Settings were picked once on the stage-3 training days with day-grouped cross-validation and the "simplest within one standard error" rule, among models that can be saved in a portable file (logistic or XGBoost; HistGradientBoosting was tried too).

**Results (95% ranges are in models/stage_results.csv).**
{chr(10).join(lines)}

**What the stages really showed (computed from the table, not hand-written).**
{chr(10).join(honest)}
Note: "patients leaving unseen" is very predictable because patience is a fixed 180-minute ASSUMPTION in the simulator (anyone already waiting about 150 minutes will leave), so a high score there says little about a real hospital.
Alarm lines come from normal-day validation snapshots only; surge days contain many more risky moments, so a higher catch rate there is expected.

**How to read the stages.** Stage 1 is the easy case (normal days only). Stage 2 shows what happens when surge days were never seen in training: the model is worse.
Stage 3 retrains with surge days. The leave-one-scenario-out rows test a surge type the model never saw in training.
"""
    update_md_section(path, "early_warning", text)


if __name__ == "__main__":
    raise SystemExit(main())
