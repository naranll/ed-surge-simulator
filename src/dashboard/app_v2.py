"""ED Surge Planner (v2): a friendly what-if tool for a hospital planner.

Run:  streamlit run src/dashboard/app_v2.py
Everything here is a SIMULATION. It is not a real hospital, not medical advice, and it does not run on LinuxONE
unless the LinuxONE panel (bottom) shows a recorded run.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.optimization.optimizer import OPTIMISER_PARAMETERS, plan_search  # noqa: E402
from src.simulation import patient_flow  # noqa: E402
from src.simulation.hospital import (HOSPITAL_PARAMETERS, PRESETS, HospitalConfig, day_metrics, generate_day,  # noqa: E402
                                     run_day, runner_for)
from src.simulation.scenarios import ATS_TARGET, PARAMETER_TABLE, SURGE_RANGES, TRIAGE_LEVEL_NOTE, truth_seed  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE, RED = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"   # colour-blind-safe (Okabe-Ito)
SCENARIOS = {"Normal Monday": "normal", "Heatwave": "heatwave", "Hurricane (receiving hospital)": "hurricane"}
CAVEAT = ("Simulation only: the numbers come from a computer model built on national survey data (NHAMCS 2022) and stated assumptions. "
          "It is not a real hospital and not medical advice.")
MODEL_CAVEAT = ("The early-warning model learns the SIMULATOR, which rests on stated assumptions. It is a fast early-warning surrogate, "
                "not validated on a real hospital.")

st.set_page_config(page_title="ED Surge Planner", page_icon=None, layout="wide")


# ------------------------------------------------------------------ cached work
def _cfg(d: dict) -> HospitalConfig:
    return HospitalConfig(**d)


@st.cache_data(show_spinner="Simulating days...")
def simulate_many(cfg_d: dict, scenario: str, intensity, burst: bool, seed0: int, runs: int) -> list:
    cfg = _cfg(cfg_d)
    out = []
    for n in range(seed0, seed0 + runs):
        truth, _ = generate_day(scenario, truth_seed(n), intensity, burst)
        out.append(day_metrics(run_day(cfg, truth), cfg))
    return out


@st.cache_data(show_spinner="Reading risk every 30 minutes...")
def risk_timeline(cfg_d: dict, scenario: str, intensity, burst: bool, seed0: int) -> list:
    from models.predict_risk import predict_risk
    cfg = _cfg(cfg_d)
    truth, info = generate_day(scenario, truth_seed(seed0), intensity, burst)
    rows = []

    def on_stop(sim):
        if 30 <= sim.now <= 1200:
            r = predict_risk(sim, info, cfg.volume_scale)
            rows.append({"hour": sim.now / 60.0, **{k: r.get(k) for k in ("risk_urgent_late", "risk_beds_full", "risk_lwbs", "breaks_first", "message", "available")}})

    run_day(cfg, truth, stop_every=30.0, on_stop=on_stop)
    return rows


@st.cache_data(show_spinner="Testing plans on many simulated days (10-30 seconds)...")
def search_plan(cfg_d: dict, scenario: str, burst: bool, n_seeds: int) -> dict:
    cfg = _cfg(cfg_d)
    base = dataclasses.replace(cfg, boarding_cut=0.0, extra_staff_start=None)    # the search adds its own shift / transfer speed-up
    res = plan_search(scenario, n_seeds=n_seeds, quick=False, burst=burst, runner=runner_for(base))
    return res.to_dict()


def mean_metrics(rs: list) -> dict:
    n = len(rs)
    avg = lambda f: sum(f(r) for r in rs) / n
    tl = [r["timeline"] for r in rs]
    k = min(len(t["hour"]) for t in tl)
    return {
        "urgent_late_pct": avg(lambda r: r["urgent_late_pct"]), "lwbs_pct": avg(lambda r: r["lwbs_pct"]),
        "peak_bed": avg(lambda r: r["peak_bed_occupancy_pct"]), "longest_queue": avg(lambda r: r["longest_queue"]),
        "patients": avg(lambda r: r["patients"]),
        "wait": {l: avg(lambda r: r["median_wait_by_level"][l]) for l in range(1, 6)},
        "hour": tl[0]["hour"][:k],
        "queue": [sum(t["queue"][i] for t in tl) / n for i in range(k)],
        "bed_pct": [sum(t["bed_pct"][i] for t in tl) / n for i in range(k)],
    }


def jload(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ sidebar
st.title("ED Surge Planner")
st.caption(CAVEAT)
st.caption(TRIAGE_LEVEL_NOTE)

with st.expander("How to read this page", expanded=False):
    st.markdown(
        "- **Left side:** describe your emergency department and pick a day type. Every box has a (?) help tip.\n"
        "- **Urgent seen late** = share of urgent patients (triage levels 1-3) who waited longer than their target time.\n"
        "- **Left without being seen** = patients who gave up waiting. **Peak beds** = the fullest the treatment beds got.\n"
        "- Results are the **average of many simulated days** (the 'repeat runs' box). Same random days are used for every comparison, so differences are real, not luck.\n"
        "- **Early warning** shows the chance that something breaks in the next 4 hours. It is learned from the simulator, not from a real hospital.\n"
        "- **Recommended plan** is shown only when its benefit stays clearly above zero across many days; otherwise the advice is 'do nothing'.")

sb = st.sidebar
sb.header("Your hospital")
preset = sb.selectbox("Start from a preset", list(PRESETS), index=2,
                      help="Small, Medium and Large are example hospitals. Pick one, then change any box below. "
                           "Large is the one we calibrated against survey data (about 200 patients a day).")
b = PRESETS[preset]
k = preset
beds = sb.number_input("Treatment beds", 1, 300, b.beds, key=f"beds_{k}", help="Beds where patients are treated. Each patient seen by a doctor holds one bed until they leave.")
rooms = sb.number_input("Treatment rooms (0 = no limit)", 0, 300, 0, key=f"rooms_{k}", help="If set, usable beds = the smaller of beds and rooms (a bed needs a room).")
stations = sb.number_input("Triage stations (0 = no limit)", 0, 50, 0, key=f"stations_{k}", help="If set, triage capacity = the smaller of stations and triage nurses.")
nurses = sb.number_input("Triage nurses", 1, 30, b.triage_nurses, key=f"nurses_{k}", help="Nurses doing the first quick assessment of each patient.")
doctors = sb.number_input("Doctors on at once", 1, 40, b.doctors, key=f"doctors_{k}", help="Doctors seeing patients at the same time.")
bays = sb.number_input("Fast-track bays", 0, 10, 0, key=f"bays_{k}", help="0 = off. 1 or more = fast track on all day: low-urgency patients (levels 4-5) need 40% less doctor time. More than one bay does not add more effect in this model.")
cut_label = sb.selectbox("Inpatient bed availability (ward-transfer speed)", ["Normal", "25% faster transfers", "50% faster transfers"], key=f"cut_{k}",
                         help="When ward beds are free, admitted patients leave the ED sooner and free their bed. 'Faster' cuts their waiting time in the ED by that share, all day.")
extra = sb.checkbox("Add an extra doctor for 6 hours", key=f"extra_{k}", help="One additional doctor on shift for 6 hours.")
start_h = sb.slider("Extra doctor starts at (hour)", 0, 18, 12, key=f"start_{k}", disabled=not extra, help="Hour of the day the extra shift begins.")

sb.header("The day")
scen_label = sb.selectbox("Day type", list(SCENARIOS), help="Normal Monday, a heatwave (more visits in the afternoon), or a hurricane (more visits all day at a receiving hospital).")
scenario = SCENARIOS[scen_label]
if scenario in SURGE_RANGES:
    lo, hi = round((SURGE_RANGES[scenario]["low"] - 1) * 100), round((SURGE_RANGES[scenario]["high"] - 1) * 100)
    pct = sb.slider("Surge size (% extra patients)", lo, hi, (lo + hi) // 2, help=f"Limited to the published range (+{lo}% to +{hi}%). It cannot be set outside what studies reported.")
    intensity = pct / 100.0
else:
    intensity = None
burst = sb.checkbox("Add a sudden burst of very urgent patients", disabled=scenario == "normal",
                    help="Stress test: 10-20 level 1-2 patients arrive within one hour. This size is our ASSUMPTION, not from a study.")
seed0 = int(sb.number_input("Random seed (first day number)", 0, 9999, 0, help="Changes which random days are simulated. Same number = same days."))
runs = int(sb.slider("Repeat runs (simulated days)", 3, 40, 10, help="More runs = steadier averages but a slightly slower page."))

flow_ok = patient_flow.available()
if not flow_ok:
    sb.warning("Real-data patient tables not found (models/patient_flow/admission_lookup.json). Using the old placeholder patient behaviour.")
try:
    cfg = HospitalConfig(beds=int(beds), rooms=int(rooms) or None, triage_stations=int(stations) or None, triage_nurses=int(nurses),
                         doctors=int(doctors), fast_track_bays=int(bays), boarding_cut={"Normal": 0.0, "25% faster transfers": 0.25, "50% faster transfers": 0.5}[cut_label],
                         extra_staff_start=float(start_h * 60) if extra else None, patient_flow=flow_ok, volume_scale=b.volume_scale)
except ValueError as err:
    st.error(str(err))
    st.stop()
with sb.expander("What these boxes change"):
    for kk, v in cfg.what_changes().items():
        st.write(f"**{kk.replace('_', ' ')}:** {v}")
    st.write(f"**patients per day:** about {round(200 * cfg.volume_scale)} (preset volume; the model's reference is 200)")

cfg_d = dataclasses.asdict(cfg)
base_d = dataclasses.asdict(dataclasses.replace(b, patient_flow=flow_ok))

# ------------------------------------------------------------------ 1. key results
chosen = mean_metrics(simulate_many(cfg_d, scenario, intensity, burst, seed0, runs))
normal = mean_metrics(simulate_many(cfg_d, "normal", None, False, seed0, runs))
st.subheader(f"1. Key results: {scen_label}" + (f" (+{round(intensity * 100)}%)" if intensity else "") + f", average of {runs} simulated days")
c = st.columns(5)
dl = lambda a, b_, fmt="{:+.1f}": fmt.format(a - b_) + " vs normal day"
c[0].metric("Urgent seen late", f"{chosen['urgent_late_pct']:.1f}%", dl(chosen["urgent_late_pct"], normal["urgent_late_pct"]), delta_color="inverse",
            help="Share of urgent patients (levels 1-3) who waited longer than their time target.")
c[1].metric("Left without being seen", f"{chosen['lwbs_pct']:.1f}%", dl(chosen["lwbs_pct"], normal["lwbs_pct"]), delta_color="inverse")
c[2].metric("Peak bed occupancy", f"{chosen['peak_bed']:.0f}%", dl(chosen["peak_bed"], normal["peak_bed"], "{:+.0f}"), delta_color="inverse")
c[3].metric("Longest queue", f"{chosen['longest_queue']:.0f} patients", dl(chosen["longest_queue"], normal["longest_queue"], "{:+.0f}"), delta_color="inverse")
c[4].metric("Patients that day", f"{chosen['patients']:.0f}")
wt = pd.DataFrame({"Triage level": [f"{l} (target {ATS_TARGET[l]:.0f} min)" for l in range(1, 6)],
                   "Median wait, chosen day (min)": [round(chosen["wait"][l], 1) for l in range(1, 6)],
                   "Median wait, normal day (min)": [round(normal["wait"][l], 1) for l in range(1, 6)]})
st.dataframe(wt, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 2. charts
st.subheader("2. Through the day: chosen day vs a normal day (same hospital)")
two = scen_label != "Normal Monday"
hrs = chosen["hour"]


def series(a, b_):
    d = {"Hour of day": hrs, scen_label: a}
    if two:
        d["Normal Monday"] = b_[:len(hrs)]
    return pd.DataFrame(d).set_index("Hour of day")


ch, cb = series(chosen["queue"], normal["queue"]), series(chosen["bed_pct"], normal["bed_pct"])
colors = [RED, BLUE] if two else [BLUE]
a1c, a2c = st.columns(2)
a1c.markdown("**Patients waiting (all queues)**")
a1c.line_chart(ch, color=colors)
a2c.markdown("**Beds in use (%)**")
a2c.line_chart(cb, color=colors)
st.markdown("**Median wait by triage level (minutes)**")
bw = wt.set_index("Triage level")[["Median wait, chosen day (min)", "Median wait, normal day (min)"]]
st.bar_chart(bw, color=[RED, BLUE])

# ------------------------------------------------------------------ 3. early warning
st.subheader("3. Early warning: what could break in the next 4 hours")
st.caption(MODEL_CAVEAT)
if cfg.effective_beds != b.effective_beds or cfg.doctors != b.doctors or cfg.triage_capacity != b.triage_capacity:
    st.info("You changed the hospital size. The early-warning model was trained on the three preset sizes only, so read its numbers as rough.")
try:
    rt = risk_timeline(cfg_d, scenario, intensity, burst, seed0)
except Exception as err:    # a model or library problem must never break the page
    rt = []
    st.warning(f"Early warning is not available here: {err}")
if rt and rt[0].get("available"):
    rdf = pd.DataFrame(rt).set_index("hour")[["risk_urgent_late", "risk_beds_full", "risk_lwbs"]]
    rdf.columns = ["Urgent patients seen late", "Beds above 95% full", "Patients leaving unseen"]
    st.line_chart(rdf, color=[RED, BLUE, GREEN])
    at = st.slider("Look at the hour", 0.5, 20.0, 12.0, 0.5, help="Pick a time of day to read the warning at that moment (first simulated day).")
    row = min(rt, key=lambda r: abs(r["hour"] - at))
    st.write(f"**At {int(row['hour']):02d}:{int(row['hour'] % 1 * 60):02d}:** breaks first: **{row['breaks_first']}**. {row['message']}")
elif rt:
    st.info(rt[0]["message"])

# ------------------------------------------------------------------ 4. recommended plan
st.subheader("4. Recommended plan")
st.caption("Tests an extra doctor for 6 hours, fast track and faster ward transfers, at different start times, on many simulated days built from the advance forecast only. "
           "A plan is recommended only if the low end of its 95% benefit range stays above a small margin (2 points, an ASSUMPTION) and non-urgent patients do not wait unfairly longer (median wait up at most 15%). "
           "The search starts from your beds, staff and fast-track setting; it adds its own extra shift and transfer speed-up.")
n_seeds = st.slider("Days to test each plan on", 10, 50, 20, help="More days = a firmer answer but a longer wait (about 10-40 seconds).")
if st.button("Find the best plan for this day type"):
    try:
        rd = search_plan(cfg_d, scenario, burst, n_seeds)
        rec = rd["recommended"]
        t = rd["truth_check"]
        if rec["label"] == "Do nothing":
            st.success("Recommendation: **do nothing extra.** " + rd["reason"])
        else:
            st.success(f"Recommendation: **{rec['label']}**  (cost about ${rec['cost']:.0f}). " + rd["reason"])
            st.write(f"Urgent patients on time: {t['base_on_time_pct']:.1f}% -> {t['plan_on_time_pct']:.1f}%. "
                     f"Non-urgent median wait: {t['base_nonurgent_median']:.1f} -> {t['plan_nonurgent_median']:.1f} min ({t['nonurgent_change_pct']:+.0f}%). "
                     f"Left unseen per day: {t['base_lwbs']:.1f} -> {t['plan_lwbs']:.1f}.")
        if rd.get("caution"):
            st.caption(rd["caution"])
        tab = pd.DataFrame(rd["rows"]).sort_values("score", ascending=False).head(8)
        keep = [c_ for c_ in ("plan", "urgent_on_time_pct", "nonurgent_median_wait_min", "lwbs", "cost_usd", "score", "lookahead_gain", "feasible") if c_ in tab.columns]
        st.dataframe(tab[keep].round(2), hide_index=True, use_container_width=True)
    except Exception as err:
        st.warning(f"The plan search could not run: {err}")
else:
    st.caption("Press the button to run the search.")

# ------------------------------------------------------------------ 5. mine vs preset
st.subheader(f"5. Your hospital vs the '{preset}' preset (same days)")
pm = mean_metrics(simulate_many(base_d, scenario, intensity, burst, seed0, runs))
cmp_ = pd.DataFrame({"Measure": ["Urgent seen late (%)", "Left without being seen (%)", "Peak bed occupancy (%)", "Longest queue (patients)", "Patients that day"],
                     "Your hospital": [chosen["urgent_late_pct"], chosen["lwbs_pct"], chosen["peak_bed"], chosen["longest_queue"], chosen["patients"]],
                     f"{preset} preset": [pm["urgent_late_pct"], pm["lwbs_pct"], pm["peak_bed"], pm["longest_queue"], pm["patients"]]}).round(1)
st.dataframe(cmp_, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 6. parameters
st.subheader("6. Where every number comes from")
st.caption("DATA = measured in survey data we processed. EVIDENCE = a published study (cited by the team, not independently checked by us). ASSUMPTION = our own stated choice.")
rows = [{"name": p["name"], "value": json.dumps(p["value"], default=str)[:140], "tag": p["tag"], "status": p["status"], "note": p["note"]} for p in PARAMETER_TABLE + OPTIMISER_PARAMETERS + HOSPITAL_PARAMETERS]
sp = jload(ROOT / "src/simulation/params/scenario_parameters.json")
ptab = pd.DataFrame(rows)
tag_filter = st.multiselect("Show tags", ["DATA", "EVIDENCE", "ASSUMPTION"], default=["DATA", "EVIDENCE", "ASSUMPTION"])
st.dataframe(ptab[ptab["tag"].isin(tag_filter)], hide_index=True, use_container_width=True)
st.caption("Data limits: triage level missing in 25.9% of 2022 visits; only 139 level-1 visits; a national sample, not one hospital; response rate about 40%. "
           "Weather, FEMA and hurricane records are context only and never set a surge size.")

# ------------------------------------------------------------------ 7. model accuracy
st.subheader("7. How accurate are the models?")
metrics = jload(ROOT / "models/metrics.json")
if not metrics:
    st.info("No model results found yet (models/metrics.json). Run the training scripts.")
else:
    pf = metrics.get("patient_flow")
    if pf:
        t = pf["admission"]["test"]
        st.markdown(f"**Model 1: patient flow (real survey data).** Admission model chosen: **{pf['admission']['chosen']['family']}**. "
                    f"AUC on hospitals it never saw: **{t['chosen_model']['auc_roc']:.2f}** "
                    f"(95% {t['chosen_model']['ci95']['auc_roc'][0]:.2f} to {t['chosen_model']['ci95']['auc_roc'][1]:.2f}) vs {t['baseline_triage_only']['auc_roc']:.2f} for 'triage level only'. "
                    "AUC: 0.5 = coin flip, 1.0 = perfect.")
    ew = metrics.get("early_warning")
    if ew:
        st.markdown("**Model 2: early warning (simulated days).** " + MODEL_CAVEAT)
        h = ew.get("headline_stage3_all_scenarios", {})
        if h:
            st.dataframe(pd.DataFrame(h).T.round(3), use_container_width=True)
    stg = ROOT / "models/stage_results.csv"
    if stg.exists():
        sdf = pd.read_csv(stg)
        cols = [c_ for c_ in ("stage", "train_on", "test_on", "target", "auc_roc", "auc_roc_lo", "auc_roc_hi", "recall_at_5pct_fpr", "rule_recall_at_5pct_fpr", "median_lead_min") if c_ in sdf.columns]
        with st.expander("Stage-by-stage results (stage 1: normal only; 2: unseen surge; 3: retrained with surge)"):
            st.dataframe(sdf[cols].round(3), hide_index=True, use_container_width=True)
    img = ROOT / "results/stage_chart.png"
    if img.exists():
        st.image(str(img))

# ------------------------------------------------------------------ 8. LinuxONE
st.subheader("8. LinuxONE")
lx = jload(ROOT / "results/linuxone_run.json")
if lx and lx.get("uname_m") != "s390x":     # only a run that really happened on s390x counts as LinuxONE
    st.warning(f"results/linuxone_run.json says machine {lx.get('uname_m')!r}, not s390x, so it is NOT shown as a LinuxONE result.")
    lx = None
if lx:
    st.success(f"LinuxONE run recorded: machine {lx.get('uname_m')}, {lx.get('nproc')} cores, Python {lx.get('python')}.")
    steps = lx.get("seconds_per_step", {})
    if steps:
        st.dataframe(pd.DataFrame({"step": list(steps), "seconds": list(steps.values())}), hide_index=True)
    tm = lx.get("parallel_timing")
    if tm:
        st.markdown("**Parallel data generation (more workers is not always faster for small jobs):**")
        st.dataframe(pd.DataFrame(tm), hide_index=True)
else:
    st.info("No LinuxONE run recorded yet.")
    loc = jload(ROOT / "results/local_run.json")
    if loc:
        st.caption(f"A local (non-LinuxONE) test run exists: machine {loc.get('uname_m')}, {loc.get('nproc')} cores. It is not shown as a LinuxONE result.")
st.caption("This dashboard itself is a normal Streamlit page; it is not claimed to run on LinuxONE.")
