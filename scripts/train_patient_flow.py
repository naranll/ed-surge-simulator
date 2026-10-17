#!/usr/bin/env python
"""A1 - patient-flow models on REAL data (NHAMCS 2022, weighted by PATWT).

Two targets, using ONLY facts known at arrival:
  (a) admitted to this hospital (ADMITHOS == 1; official label "Admit to this hospital", 0 = no, 1 = yes)
  (b) care time = LOV - WAITTIME on the log scale (a PROXY: the whole stay after first provider contact,
      including tests and boarding; it is NOT pure service time)

Candidates: simple baseline, logistic / ridge, sklearn HistGradientBoosting, XGBoost.
Chosen by hospital-grouped, stratified cross-validation on the TRAINING rows only; the simplest model
within one standard error of the best wins. The test hospitals are never used to choose anything.

Writes models/patient_flow/*, models/metrics.json (key "patient_flow"), models/model_card.md (A1 section)
and the lookup the standard-library simulator reads (admission_lookup.json).

    python scripts/train_patient_flow.py            (full, about 2-4 minutes)
    python scripts/train_patient_flow.py --quick    (fewer configs and bootstrap draws)
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

from _common import MODELS, NHAMCS_CLEAN, ROOT, update_json, update_md_section

warnings.filterwarnings("ignore")

SEED = 20261017
AGE_BANDS = [("0-17", 0, 17), ("18-44", 18, 44), ("45-64", 45, 64), ("65+", 65, 200)]
SHRINK_K = 10.0          # ASSUMPTION: a cell is blended with its triage-level rate as if it had 10 extra patients
AUC_LEAK_LIMIT = 0.95    # above this, stop: probable leakage

INCLUDED = {
    "age_years": "AGE_years (cleaned AGE; 94 means 94 or older)",
    "triage_level": "IMMEDR_clean (valid 1-5 only)",
    "arrival_hour": "arrival_hour from ARRTIME (HHMM)",
    "weekday": "VDAYR (official labels: 1 Sunday ... 7 Saturday)",
    "payment_type": "PAYTYPER (recoded primary expected source of payment)",
    "sex": "SEX",
    "ambulance_arrival": "ARREMS (1 yes, 2 no, negative = unknown)",
    "injury_flag": "INJURY (0 no, 1 yes, 2 questionable, negative = unknown)",
}
EXCLUDED = ("WAITTIME, LOV (they define the care-time target); ADISP / ADMITHOS / LOS / BOARDED / disposition columns "
            "(outcomes); diagnoses, tests, imaging, procedures, medicines (recorded after arrival); vital signs, "
            "pain scale and reason-for-visit (recorded at triage, not at the door); LWBS; hospital-level "
            "survey columns. HOSPCODE is used only to GROUP the split, never as a feature.")


# ------------------------------------------------------------------ data
def load_table(path=NHAMCS_CLEAN) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, low_memory=False)
    n_all = len(raw)
    d = pd.DataFrame({
        "age_years": raw["AGE_years"], "triage_level": raw["IMMEDR_clean"], "arrival_hour": raw["arrival_hour"],
        "weekday": raw["VDAYR"], "payment_type": raw["PAYTYPER"], "sex": raw["SEX"],
        "ambulance_arrival": raw["ARREMS"], "injury_flag": raw["INJURY"],
        "admitted": (raw["ADMITHOS"] == 1).astype(int), "lwbs": raw["LWBS"],
        "wait_min": raw["WAITTIME_minutes"], "lov_min": raw["LOV_minutes"],
        "hospital": raw["HOSPCODE"], "w": raw["PATWT"]})
    d["care_min"] = d["lov_min"] - d["wait_min"]
    keep = d["triage_level"].isin([1, 2, 3, 4, 5]) & d["age_years"].notna() & d["arrival_hour"].notna()
    notes = {"rows_all": n_all, "rows_valid_triage_1_5": int(d["triage_level"].isin([1, 2, 3, 4, 5]).sum()),
             "rows_used": int(keep.sum()), "rows_dropped_missing_age_or_hour": int(
                 (d["triage_level"].isin([1, 2, 3, 4, 5]) & ~(d["age_years"].notna() & d["arrival_hour"].notna())).sum())}
    d = d[keep].reset_index(drop=True)
    d["triage_level"] = d["triage_level"].astype(int)
    d["arrival_hour"] = d["arrival_hour"].astype(int)
    # care time is only defined for patients who were actually seen, with both times valid and LOV > WAITTIME
    d["care_ok"] = d["care_min"].notna() & (d["care_min"] > 0) & (d["lwbs"] != 1)
    notes["rows_with_valid_care_time"] = int(d["care_ok"].sum())
    notes["hospitals"] = int(d["hospital"].nunique())
    return d, notes


def age_band(age: float) -> str:
    for name, lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return name
    return AGE_BANDS[-1][0]


def build_X(d: pd.DataFrame) -> pd.DataFrame:
    """Numeric design matrix with FIXED columns, so train and test always agree."""
    X = pd.DataFrame(index=d.index)
    X["age_years"] = d["age_years"].astype(float)
    X["triage_level"] = d["triage_level"].astype(float)
    ang = 2 * math.pi * d["arrival_hour"].astype(float) / 24.0
    X["hour_sin"], X["hour_cos"] = np.sin(ang), np.cos(ang)
    X["sex_female"] = (d["sex"] == 2).astype(float)
    for k in range(1, 8):
        X[f"weekday_{k}"] = (d["weekday"] == k).astype(float)
    for k in (1, 2, 3, 4, 5, 6, 7):
        X[f"pay_{k}"] = (d["payment_type"] == k).astype(float)
    X["pay_unknown"] = (d["payment_type"] < 0).astype(float)
    X["amb_yes"] = (d["ambulance_arrival"] == 1).astype(float)
    X["amb_unknown"] = (d["ambulance_arrival"] < 0).astype(float)
    X["inj_yes"] = (d["injury_flag"] == 1).astype(float)
    X["inj_questionable"] = (d["injury_flag"] == 2).astype(float)
    X["inj_unknown"] = (d["injury_flag"] < 0).astype(float)
    return X


# ------------------------------------------------------------------ metrics
def wmedian(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def ece_table(y, p, w, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece, tot = [], 0.0, float(np.sum(w))
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        wb = float(np.sum(w[m]))
        pm, om = float(np.average(p[m], weights=w[m])), float(np.average(y[m], weights=w[m]))
        ece += wb / tot * abs(pm - om)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_predicted": round(pm, 4), "observed_rate": round(om, 4)})
    return ece, rows


def clf_metrics(y, p, w) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    return {"auc_roc": float(roc_auc_score(y, p, sample_weight=w)),
            "pr_auc": float(average_precision_score(y, p, sample_weight=w)),
            "brier": float(brier_score_loss(y, p, sample_weight=w)),
            "ece": float(ece_table(y, p, w)[0])}


def reg_metrics(y, p, w) -> dict:
    from sklearn.metrics import r2_score
    err = np.abs(y - p)
    return {"mae_log": float(np.average(err, weights=w)), "median_abs_error_log": wmedian(err, w),
            "r2": float(r2_score(y, p, sample_weight=w))}


def cluster_bootstrap(fn, groups, n_boot, seed):
    """95% interval from resampling whole hospitals (rows of one hospital stay together)."""
    rng = np.random.RandomState(seed)
    ids = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ids}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([by[g] for g in pick])
        try:
            out.append(fn(idx))
        except ValueError:      # a draw with one class only
            continue
    keys = out[0].keys()
    return {k: [float(np.percentile([o[k] for o in out], 2.5)), float(np.percentile([o[k] for o in out], 97.5))]
            for k in keys}


# ------------------------------------------------------------------ models
def make_model(family: str, params: dict, task: str):
    if family == "logistic":
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        est = LogisticRegression(C=params["C"], max_iter=3000) if task == "clf" else Ridge(alpha=params["alpha"])
        return make_pipeline(StandardScaler(), est)
    if family == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        cls = HistGradientBoostingClassifier if task == "clf" else HistGradientBoostingRegressor
        return cls(random_state=SEED, **params)
    if family == "xgb":
        import xgboost as xgb
        cls = xgb.XGBClassifier if task == "clf" else xgb.XGBRegressor
        return cls(random_state=SEED, n_jobs=4, verbosity=0, tree_method="hist", **params)
    raise ValueError(family)


def fit(model, X, y, w):
    if hasattr(model, "steps"):
        model.fit(X, y, **{model.steps[-1][0] + "__sample_weight": w})
    else:
        model.fit(X, y, sample_weight=w)
    return model


def predict(model, X, task):
    return model.predict_proba(X)[:, 1] if task == "clf" else model.predict(X)


def candidate_configs(n_total: int, task: str) -> list:
    """Random search, at most n_total configs in all (the logistic ones are few and cheap)."""
    rs = np.random.RandomState(SEED)
    cfg = []
    if task == "clf":
        cfg += [("logistic", {"C": c}) for c in (0.1, 1.0, 10.0)]
    else:
        cfg += [("logistic", {"alpha": a}) for a in (1.0, 10.0, 100.0)]
    rest = max(0, n_total - len(cfg))
    for i in range(rest):
        if i % 2 == 0:
            cfg.append(("hgb", {"max_iter": int(rs.choice([60, 120, 200])), "learning_rate": float(rs.choice([0.03, 0.06, 0.1])),
                                "max_depth": int(rs.choice([2, 3, 4])), "min_samples_leaf": int(rs.choice([20, 50, 100])),
                                "l2_regularization": float(rs.choice([0.0, 1.0, 5.0]))}))
        else:
            cfg.append(("xgb", {"n_estimators": int(rs.choice([60, 120, 200])), "learning_rate": float(rs.choice([0.03, 0.06, 0.1])),
                                "max_depth": int(rs.choice([2, 3, 4])), "min_child_weight": float(rs.choice([1, 5, 10])),
                                "subsample": float(rs.choice([0.7, 0.9, 1.0])), "reg_lambda": float(rs.choice([1.0, 5.0, 10.0]))}))
    return cfg[:n_total]


COMPLEXITY = {"baseline": 0, "logistic": 1, "hgb": 2, "xgb": 3}


def cv_select(X, y, w, groups, strata, task, n_total, budget_sec, folds=5):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import log_loss
    t0 = time.time()
    splits = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED).split(X, strata, groups))
    results = []
    for family, params in candidate_configs(n_total, task):
        if time.time() - t0 > budget_sec:
            break
        losses = []
        for tr, va in splits:
            m = fit(make_model(family, params, task), X.iloc[tr], y[tr], w[tr])
            p = predict(m, X.iloc[va], task)
            if task == "clf":
                losses.append(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6), sample_weight=w[va]))
            else:
                losses.append(float(np.average(np.abs(y[va] - p), weights=w[va])))
        results.append({"family": family, "params": params, "cv_mean": float(np.mean(losses)),
                        "cv_se": float(np.std(losses, ddof=1) / math.sqrt(len(losses)))})
    best = min(results, key=lambda r: r["cv_mean"])
    ok = [r for r in results if r["cv_mean"] <= best["cv_mean"] + best["cv_se"]]
    chosen = min(ok, key=lambda r: (COMPLEXITY[r["family"]], r["cv_mean"]))
    return chosen, best, results, round(time.time() - t0, 1)


# ------------------------------------------------------------------ lookup for the stdlib simulator
def build_lookup(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["band"] = d["age_years"].map(age_band)
    out = {"tag": "DATA", "source": "NHAMCS 2022 official public-use file, PATWT-weighted, valid triage 1-5 rows",
           "age_bands": [b[0] for b in AGE_BANDS], "shrinkage_k": SHRINK_K,
           "note": ("Admission chance by triage level x age band. Cells are blended toward the triage-level rate "
                    f"(shrinkage k={SHRINK_K:g}, an ASSUMPTION) because some cells are small. 'n' is the unweighted count. "
                    "Care time is LOV - WAITTIME, a proxy that includes tests and boarding, NOT pure service time."),
           "admission": {}, "age_share_given_triage": {}, "care_time_log": {}}
    for lvl in range(1, 6):
        s = d[d["triage_level"] == lvl]
        lvl_rate = float(np.average(s["admitted"], weights=s["w"]))
        out["admission"][str(lvl)] = {"triage_rate": round(lvl_rate, 5), "n": int(len(s)), "bands": {}}
        shares = s.groupby("band")["w"].sum()
        tot = float(shares.sum())
        out["age_share_given_triage"][str(lvl)] = {b[0]: round(float(shares.get(b[0], 0.0)) / tot, 5) for b in AGE_BANDS}
        for b, _, _ in AGE_BANDS:
            c = s[s["band"] == b]
            n = int(len(c))
            raw = float(np.average(c["admitted"], weights=c["w"])) if n else lvl_rate
            # effective sample size of the weighted cell for shrinkage = unweighted n (simple, stated)
            shr = (n * raw + SHRINK_K * lvl_rate) / (n + SHRINK_K)
            out["admission"][str(lvl)]["bands"][b] = {"rate_raw": round(raw, 5), "rate": round(shr, 5), "n": n}
        for adm in (0, 1):
            c = s[(s["care_ok"]) & (s["admitted"] == adm)]
            if len(c) >= 5:
                lg = np.log(c["care_min"].to_numpy())
                mu = float(np.average(lg, weights=c["w"]))
                sd = float(math.sqrt(np.average((lg - mu) ** 2, weights=c["w"])))
                out["care_time_log"][f"{lvl}_{'admitted' if adm else 'discharged'}"] = {
                    "mu": round(mu, 4), "sigma": round(sd, 4), "median_min": round(wmedian(c["care_min"], c["w"]), 1),
                    "n": int(len(c))}
    adm_all = float(np.average(d["admitted"], weights=d["w"]))
    out["overall_admission_rate_weighted"] = round(adm_all, 5)
    return out


def real_targets(raw_path) -> dict:
    """NHAMCS equivalents of the old, unsourced calibration targets (computed, not assumed)."""
    r = pd.read_csv(raw_path, low_memory=False, usecols=["IMMEDR_clean", "WAITTIME_minutes", "LWBS", "PATWT"])
    urgent = r[r["IMMEDR_clean"].isin([1, 2, 3]) & r["WAITTIME_minutes"].notna()]
    return {
        "tag": "DATA", "source": "NHAMCS 2022, PATWT-weighted",
        "urgent_levels_1_3_median_wait_min": round(wmedian(urgent["WAITTIME_minutes"], urgent["PATWT"]), 1),
        "urgent_n": int(len(urgent)),
        "lwbs_share_weighted": round(float(np.average(r["LWBS"] == 1, weights=r["PATWT"])), 5),
        "lwbs_share_unweighted": round(float((r["LWBS"] == 1).mean()), 5),
        "note": ("WAITTIME is the time from arrival to first provider contact as recorded by the survey. The simulator's "
                 "door-to-doctor wait is the closest equivalent. 17.2% of visits have no WAITTIME; LWBS here is the "
                 "survey's LWBS flag over all 16,025 visits."),
    }


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="9 configs, 100 bootstrap draws, 3 folds")
    ap.add_argument("--data", default=str(NHAMCS_CLEAN))
    ap.add_argument("--out", default=str(MODELS))
    ap.add_argument("--budget-sec", type=float, default=600.0, help="stop the config search after this many seconds")
    a = ap.parse_args(argv)
    t_start = time.time()
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        import xgboost  # noqa: F401
    except ImportError as err:
        print(f"SKIPPED A1: a machine-learning library is missing ({err}). Install scikit-learn and xgboost.")
        return 2
    if not Path(a.data).exists():
        print(f"STOPPED: real data not found at {a.data}")
        return 1

    out = Path(a.out)
    pf = out / "patient_flow"
    pf.mkdir(parents=True, exist_ok=True)
    n_cfg, n_boot, folds = (9, 100, 3) if a.quick else (20, 300, 5)

    d, notes = load_table(a.data)
    X = build_X(d)
    strata = (d["triage_level"] * 2 + d["admitted"]).to_numpy()
    groups = d["hospital"].to_numpy()
    w_all = d["w"].to_numpy(float)
    w_all = w_all / w_all.mean()

    # one hospital-grouped, stratified 80/20 split, fixed seed; test hospitals are never used for choosing
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, te = next(sgkf.split(X, strata, groups))
    assert not (set(groups[tr]) & set(groups[te])), "a hospital is in both train and test"
    print(f"rows used {len(d)} | train {len(tr)} / test {len(te)} | hospitals {len(set(groups[tr]))} / {len(set(groups[te]))}")

    report = {"data": notes, "split": {"seed": SEED, "grouped_by": "HOSPCODE (hospital number)", "stratified_by": "triage level x admitted",
                                       "train_rows": int(len(tr)), "test_rows": int(len(te)),
                                       "train_hospitals": int(len(set(groups[tr]))), "test_hospitals": int(len(set(groups[te])))},
              "features_included": INCLUDED, "features_excluded": EXCLUDED}
    lookup_tr = build_lookup(d.iloc[tr])     # lookup built on TRAIN rows only, to be scored on test as a candidate

    # ---------------- (a) admission
    y = d["admitted"].to_numpy()
    chosen, best, cvres, secs = cv_select(X.iloc[tr], y[tr], w_all[tr], groups[tr], strata[tr], "clf", n_cfg, a.budget_sec, folds)
    print(f"[admission] searched {len(cvres)} configs in {secs}s -> chosen {chosen['family']} {chosen['params']}")
    mdl = fit(make_model(chosen["family"], chosen["params"], "clf"), X.iloc[tr], y[tr], w_all[tr])
    p_te = predict(mdl, X.iloc[te], "clf")
    # baseline: admission rate by triage level only (train weighted rates)
    rate = {l: float(np.average(y[tr][d["triage_level"].to_numpy()[tr] == l], weights=w_all[tr][d["triage_level"].to_numpy()[tr] == l]))
            for l in range(1, 6)}
    p_base = np.array([rate[l] for l in d["triage_level"].to_numpy()[te]])
    bands = d["age_years"].map(age_band).to_numpy()[te]
    p_look = np.array([lookup_tr["admission"][str(l)]["bands"][b]["rate"] for l, b in zip(d["triage_level"].to_numpy()[te], bands)])
    yt, wt, gt = y[te], w_all[te], groups[te]
    models_eval = {"chosen_model": p_te, "baseline_triage_only": p_base, "lookup_triage_x_age_band": p_look}
    adm = {"chosen": chosen, "best_by_cv": best, "all_configs": cvres, "search_seconds": secs, "test": {}}
    for name, p in models_eval.items():
        m = clf_metrics(yt, p, wt)
        ci = cluster_bootstrap(lambda idx, p=p: clf_metrics(yt[idx], p[idx], wt[idx]), gt, n_boot, SEED)
        adm["test"][name] = {**m, "ci95": ci}
    if adm["test"]["chosen_model"]["auc_roc"] > AUC_LEAK_LIMIT:
        print(f"STOP: admission AUC {adm['test']['chosen_model']['auc_roc']:.3f} > {AUC_LEAK_LIMIT}: probable leakage")
        return 3
    adm["reliability_table_chosen"] = ece_table(yt, p_te, wt)[1]
    report["admission"] = adm
    print("[admission] test AUC chosen {:.3f} vs triage-only baseline {:.3f}".format(
        adm["test"]["chosen_model"]["auc_roc"], adm["test"]["baseline_triage_only"]["auc_roc"]))

    # ---------------- (b) log care time
    ok = d["care_ok"].to_numpy()
    tr_c, te_c = tr[ok[tr]], te[ok[te]]
    lc = np.log(np.where(ok, d["care_min"].to_numpy(), 1.0))
    ch2, best2, cv2, secs2 = cv_select(X.iloc[tr_c], lc[tr_c], w_all[tr_c], groups[tr_c], strata[tr_c], "reg", n_cfg, a.budget_sec, folds)
    print(f"[care time] searched {len(cv2)} configs in {secs2}s -> chosen {ch2['family']} {ch2['params']}")
    m2 = fit(make_model(ch2["family"], ch2["params"], "reg"), X.iloc[tr_c], lc[tr_c], w_all[tr_c])
    q_te = predict(m2, X.iloc[te_c], "reg")
    med = {l: wmedian(lc[tr_c][d["triage_level"].to_numpy()[tr_c] == l], w_all[tr_c][d["triage_level"].to_numpy()[tr_c] == l]) for l in range(1, 6)}
    q_base = np.array([med[l] for l in d["triage_level"].to_numpy()[te_c]])
    yc, wc, gc = lc[te_c], w_all[te_c], groups[te_c]
    care = {"chosen": ch2, "best_by_cv": best2, "all_configs": cv2, "search_seconds": secs2, "test": {},
            "target_note": "log(LOV - WAITTIME) in minutes: a proxy for time in the ED after first provider contact (includes tests, boarding)."}
    for name, p in {"chosen_model": q_te, "baseline_triage_median_only": q_base}.items():
        m = reg_metrics(yc, p, wc)
        ci = cluster_bootstrap(lambda idx, p=p: reg_metrics(yc[idx], p[idx], wc[idx]), gc, n_boot, SEED)
        care["test"][name] = {**m, "ci95": ci}
    report["care_time"] = care
    print("[care time] test MAE(log) chosen {:.3f} vs baseline {:.3f}; R2 {:.3f}".format(
        care["test"]["chosen_model"]["mae_log"], care["test"]["baseline_triage_median_only"]["mae_log"],
        care["test"]["chosen_model"]["r2"]))

    # ---------------- save portable artefacts (final fit on ALL valid rows for the saved models)
    saved = {}
    for tag, ch_, cvr, task, rows, yy, tr_rows, te_rows in (
            ("admission", chosen, cvres, "clf", np.arange(len(d)), y, tr, te),
            ("care_time", ch2, cv2, "reg", np.where(ok)[0], lc, tr_c, te_c)):
        spec = ch_
        if spec["family"] == "hgb":
            # HistGradientBoosting has no portable file format. Save the best portable candidate (by CV) too, and report its test score.
            spec = min([r for r in cvr if r["family"] in ("logistic", "xgb")], key=lambda r: r["cv_mean"])
            saved[tag + "_note"] = (f"CV winner is HistGradientBoosting (cannot be saved portably); the saved file is the best portable "
                                    f"candidate instead: {spec['family']} {spec['params']}")
            twin = fit(make_model(spec["family"], spec["params"], task), X.iloc[tr_rows], yy[tr_rows], w_all[tr_rows])
            pt = predict(twin, X.iloc[te_rows], task)
            tw = (clf_metrics(yy[te_rows], pt, w_all[te_rows]) if task == "clf" else reg_metrics(yy[te_rows], pt, w_all[te_rows]))
            report["admission" if task == "clf" else "care_time"]["test"]["portable_saved_model"] = {"family": spec["family"], "params": spec["params"], **tw}
        family, params = spec["family"], spec["params"]
        full = fit(make_model(family, params, task), X.iloc[rows], yy[rows], w_all[rows])
        if family == "logistic":
            sc, est = full.steps[0][1], full.steps[1][1]
            path = pf / f"{tag}_logistic.json"
            path.write_text(json.dumps({"kind": "linear", "task": task, "features": list(X.columns),
                                        "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                                        "coef": np.ravel(est.coef_).tolist(), "intercept": float(np.ravel(est.intercept_)[0])},
                                       indent=1), encoding="utf-8")
            saved[tag] = path.name
        elif family == "xgb":
            path = pf / f"{tag}_xgboost.json"
            full.get_booster().save_model(str(path))
            saved[tag] = path.name
        else:
            saved[tag] = "none (HistGradientBoosting has no portable file; rerun this script on the target machine)"
    report["saved_files"] = saved

    lookup = build_lookup(d)
    lookup["real_targets_for_calibration"] = real_targets(a.data)
    (pf / "admission_lookup.json").write_text(json.dumps(lookup, indent=1), encoding="utf-8")
    update_json(out / "metrics.json", "patient_flow", report)
    report["runtime_seconds"] = round(time.time() - t_start, 1)
    update_json(out / "metrics.json", "patient_flow", report)
    write_card(out / "model_card.md", report, lookup)
    print(f"done in {report['runtime_seconds']}s. Wrote {pf}/ , metrics.json, model_card.md")
    return 0


def write_card(path, r, lookup) -> None:
    t, c = r["admission"]["test"], r["care_time"]["test"]
    ch, ch2 = r["admission"]["chosen"], r["care_time"]["chosen"]
    f = lambda m, k: f"{m[k]:.3f} (95% {m['ci95'][k][0]:.3f} to {m['ci95'][k][1]:.3f})"
    text = f"""## Model 1 - patient flow (real data, NHAMCS 2022)

**What it does, in plain words.** It learns, from 10,000+ real emergency visits, two things about a patient *when they walk in*:
how likely they are to be admitted to the hospital, and how long they will need care. The simulator uses the result
(the admission lookup) so that simulated patients behave like real ones.

**Data.** NHAMCS 2022 official public-use file (CDC). {r['data']['rows_used']:,} visits with a valid triage level 1-5 were used
(of {r['data']['rows_all']:,}); {r['data']['rows_with_valid_care_time']:,} have a valid care time. Everything is weighted by PATWT
(the survey weight). The sample is national, from {r['data']['hospitals']} hospitals, not one Houston hospital. Triage is missing for about 26% of visits.

**Only facts known at arrival were used:** {', '.join(INCLUDED)}.
**Never used:** {EXCLUDED}

**Split.** Whole hospitals are held out (80% train / 20% test, fixed seed {SEED}); no hospital is in both. Inside the training part,
5-fold hospital-grouped, stratified cross-validation chose the model. Rule: take the simplest model whose CV error is within one
standard error of the best.

**Admission model.** Chosen: **{ch['family']}** {ch['params']} (searched {len(r['admission']['all_configs'])} settings).
On the held-out hospitals: AUC {f(t['chosen_model'], 'auc_roc')}; "triage level only" baseline AUC {f(t['baseline_triage_only'], 'auc_roc')}.
Brier {f(t['chosen_model'], 'brier')} vs {f(t['baseline_triage_only'], 'brier')}. Calibration error (ECE) {f(t['chosen_model'], 'ece')}.
Lookup table (triage x age band) AUC {f(t['lookup_triage_x_age_band'], 'auc_roc')}.
Plain reading: age adds some information on top of triage level, but a lot is unknown at the door, so this is a modest model, not a crystal ball.

**Care-time model (log scale).** Chosen: **{ch2['family']}** {ch2['params']}. Median absolute error (log) {f(c['chosen_model'], 'median_abs_error_log')},
R-squared {f(c['chosen_model'], 'r2')}; baseline (median by triage) MAE {f(c['baseline_triage_median_only'], 'mae_log')} vs model {f(c['chosen_model'], 'mae_log')}.
**Care time = LOV - WAITTIME is a proxy**: it is the whole stay after first provider contact, including tests and, for admitted patients, boarding. It is not pure service time.

**Limits.** Survey data (about 40% response rate, possible non-response bias). No variance estimates. Only 139 level-1 visits.
Model accuracy is on held-out hospitals of the same survey, not on a real Houston hospital.
Saved files: {'; '.join(f'{k}: {v}' for k, v in r['saved_files'].items())}.
The admission lookup that the simulator reads is a descriptive table computed on ALL valid rows; its accuracy on held-out hospitals (shown above) was measured with a copy built from the training rows only.
"""
    update_md_section(path, "patient_flow", text)


if __name__ == "__main__":
    raise SystemExit(main())
