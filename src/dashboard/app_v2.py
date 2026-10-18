"""KnightAI - ED Surge Simulator (dashboard v2): a plain-words what-if tool for a hospital planner / charge nurse.

Run from the repo root (the theme comes from .streamlit/config.toml there):
    streamlit run src/dashboard/app_v2.py
Recorded results are read from results/ (another folder can be picked in the sidebar).
Open a day type directly with ?scenario=heatwave (normal, heatwave, hurricane, hurricane_burst, compound).
Model files are always read from models/.
Everything here is a SIMULATION. It is not a real hospital, not medical advice, and it is not shown as a LinuxONE result
unless the chosen results folder holds a linuxone_run.json recorded on an s390x machine.
"""
from __future__ import annotations

import dataclasses
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.optimization.optimizer import (DEFAULT_SETTINGS, MIN_BENEFIT_PP, OPTIMISER_PARAMETERS, plan_search)  # noqa: E402
from src.simulation import patient_flow, registry  # noqa: E402
from src.simulation.ed_simulator import FT_FACTOR  # noqa: E402
from src.simulation.hospital import (HOSPITAL_PARAMETERS, PRESETS, HospitalConfig, binding_constraint, day_metrics,  # noqa: E402
                                     generate_day, run_day, runner_for)
from src.simulation.scenarios import (ATS_TARGET, BURST_SIZE_RANGE, PARAMETER_TABLE, SURGE_RANGES, TRIAGE_LEVEL_NOTE,  # noqa: E402
                                      truth_seed)

PRODUCT = "KnightAI"                      # product name used in README.md
SUBTITLE = "ED Surge Simulator"
# Clinical palette: medical blue (primary), green (good / recommended), amber and red only for warnings.
BLUE, GREEN, AMBER, RED = "#0B5CAD", "#15803D", "#B45309", "#B91C1C"
GREY_LINE = "#6B7280"                     # 'normal day' reference line in charts
SCENARIOS = {registry.label(k): k for k in registry.names()}     # the one scenario registry (label -> key)
KEY2LABEL = {v: k for k, v in SCENARIOS.items()}
SHORT = {"normal": "Normal day", "heatwave": "Heatwave", "hurricane": "Hurricane",
         "hurricane_burst": "Hurricane + burst", "compound": "Heatwave + hurricane"}
CAVEAT_LINE = ("Simulation only: a computer model built on national survey data (NHAMCS 2022) and stated assumptions. "
               "Not a real hospital, not medical advice.")
CAVEAT = ("Simulation only: the numbers come from a computer model built on national survey data (NHAMCS 2022) and stated assumptions. "
          "It is not a real hospital and not medical advice.")
MODEL_CAVEAT = ("The early-warning model learns the SIMULATOR, which rests on stated assumptions. It is a fast early-warning surrogate, "
                "not validated on a real hospital.")
ATS_TEXT = ", ".join(("immediate" if ATS_TARGET[l] == 0 else f"{ATS_TARGET[l]:.0f} min") + f" (level {l})" for l in range(1, 6))
HELP_URGENT = ("Share of urgent patients (triage levels 1-3) who waited longer than their time target before seeing a doctor. "
               f"Targets: {ATS_TEXT}. These are Australasian Triage Scale (ATS) time targets applied to the US NHAMCS triage "
               "levels 1-5 as an ASSUMPTION.")
HELP_LWBS = ("LWBS = 'left without being seen': patients who gave up waiting and left before a doctor saw them, "
             "as a share of all patients that day.")
HELP_BEDS = "The fullest the treatment beds got during the day, as a share of all usable beds (100% = every bed taken)."
HELP_RANGE = ("95% range: we tested the plan on many simulated days and paired each with the same day without the plan. "
              "If the test were repeated, the true average gain would fall inside this range about 95 times in 100. "
              "Gain is in points: 1 point = 1 percentage point more urgent patients seen on time, after the cost and "
              "left-unseen penalties.")
COST_NOTE = "Costs are illustrative and UNSOURCED (our assumption), always shown with the staff-hours they stand for."

st.set_page_config(page_title=f"{PRODUCT} - {SUBTITLE}", page_icon=None, layout="wide")

# Light clinical look. Colours and fonts come from .streamlit/config.toml (repo root); this block only styles the page's
# own cards and fixes contrast where Streamlit fades text (captions, widget labels) below WCAG AA.
_CSS = """<style>
  .block-container { max-width: 1240px; padding-top: 1.6rem; padding-bottom: 3rem; }
  h1, h2, h3, h4 { color: #111827; }
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] * { color: #4B5563 !important; opacity: 1 !important; }
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: #1F2937; font-weight: 600; }
  [data-testid="stExpander"] summary p { color: #111827; font-weight: 600; }
  [data-testid="stSidebar"] h2 { font-size: 1.05rem; color: #111827; margin-top: .6rem; }
  [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
      background: #FFFFFF !important; border-color: #D1D5DB !important; color: #1F2937; }
  [data-baseweb="input"] input, [data-baseweb="select"] div { color: #1F2937; }
  [data-testid="stNumberInputContainer"] { background: #FFFFFF; border-color: #D1D5DB; }
  [data-testid="stNumberInput"] button { background: #FFFFFF; color: #1F2937; }
  [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px 16px; }
  [data-testid="stMetricLabel"] p { color: #1F2937; font-weight: 600; }
  [data-testid="stMetricValue"] { color: #111827; }
  [data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]), [data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) * { color: #B91C1C !important; }
  [data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]), [data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) * { color: #15803D !important; }
  .po-hdr { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px 16px;
            padding-bottom: 12px; border-bottom: 1px solid #E5E7EB; }
  .po-brand { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .po-word { font-size: 30px; font-weight: 800; color: #0B5CAD; letter-spacing: -.01em; line-height: 1.1; }
  .po-sub { font-size: 18px; font-weight: 600; color: #1F2937; }
  .po-pills { display: flex; gap: 8px; flex-wrap: wrap; }
  .po-pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 13px;
             font-weight: 600; border: 1px solid transparent; white-space: nowrap; }
  .po-lx { background: #F0FDF4; color: #15803D; border-color: #BBF7D0; }
  .po-dot { width: 8px; height: 8px; border-radius: 50%; background: #15803D; display: inline-block; }
  .po-caveat { color: #4B5563; font-size: 14px; margin: 8px 0 4px 0; }
  .po-h2 { font-size: 20px; font-weight: 700; color: #111827; margin: 24px 0 4px 0; }
  .po-h2-sub { color: #4B5563; font-size: 14px; margin: 0 0 12px 0; }
  .po-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; }
  .po-kpi { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px 16px; }
  .po-kpi-label { font-size: 14px; font-weight: 600; color: #1F2937; }
  .po-kpi-value { font-size: 32px; font-weight: 700; color: #111827; line-height: 1.2; margin: 4px 0 2px 0; }
  .po-kpi-cap { font-size: 13px; color: #4B5563; }
  .po-delta { display: inline-block; margin-top: 8px; padding: 2px 8px; border-radius: 6px; font-size: 13px; font-weight: 600; }
  .po-same { background: #F3F4F6; color: #4B5563; }
  .po-worse { background: #FEF2F2; color: #B91C1C; }
  .po-better { background: #F0FDF4; color: #15803D; }
  .po-rec { background: #FFFFFF; border: 1px solid #E5E7EB; border-left: 6px solid #15803D; border-radius: 8px; padding: 16px 20px; }
  .po-rec.po-none { border-left-color: #0B5CAD; }
  .po-rec-tag { font-size: 12px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #15803D; }
  .po-rec.po-none .po-rec-tag { color: #0B5CAD; }
  .po-rec-title { font-size: 22px; font-weight: 700; color: #111827; margin: 2px 0 12px 0; }
  .po-rec-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
  .po-rec-val { font-size: 22px; font-weight: 700; color: #111827; margin: 2px 0; }
  .po-rec-effect { margin-top: 12px; padding-top: 12px; border-top: 1px solid #E5E7EB; color: #1F2937; font-size: 15px; }
  .po-rec-note { margin-top: 8px; color: #4B5563; font-size: 13px; }
  .po-risk { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 4px 0 8px 0; }
  .po-risk > div { border: 1px solid #E5E7EB; border-radius: 8px; padding: 10px 14px; background: #FFFFFF; }
  .po-risk .v { font-size: 24px; font-weight: 700; color: #111827; }
  .po-risk .v.hi { color: #B91C1C; } .po-risk .v.mid { color: #B45309; }
  .po-risk .l { font-size: 13px; color: #4B5563; }
  [class*="st-key-scen_pick"] [role="radiogroup"] { gap: 0; flex-wrap: wrap; }
  [class*="st-key-scen_pick"] [role="radiogroup"] label { margin: 0 -1px 4px 0; padding: 7px 16px; border: 1px solid #D1D5DB; background: #FFFFFF; cursor: pointer; }
  [class*="st-key-scen_pick"] [role="radiogroup"] > label:first-of-type, [class*="st-key-scen_pick"] [role="radiogroup"] > div:first-of-type label { border-radius: 8px 0 0 8px; }
  [class*="st-key-scen_pick"] [role="radiogroup"] > label:last-of-type, [class*="st-key-scen_pick"] [role="radiogroup"] > div:last-of-type label { border-radius: 0 8px 8px 0; }
  [class*="st-key-scen_pick"] [role="radiogroup"] label > div:first-of-type:not(:has([data-testid="stMarkdownContainer"])), [class*="st-key-scen_pick"] [role="radiogroup"] label > div > div:first-child:not([data-testid="stMarkdownContainer"]) { display: none; }
  [class*="st-key-scen_pick"] [role="radiogroup"] label p { color: #1F2937; font-weight: 600; font-size: 15px; }
  [class*="st-key-scen_pick"] [role="radiogroup"] label:has(input:checked) { background: #0B5CAD; border-color: #0B5CAD; z-index: 1; }
  [class*="st-key-scen_pick"] [role="radiogroup"] label:has(input:checked) p { color: #FFFFFF; }
  [class*="st-key-scen_pick"] [role="radiogroup"] label:hover { background: #EFF6FF; }
  [class*="st-key-scen_pick"] [role="radiogroup"] label:has(input:checked):hover { background: #0B5CAD; }
  .po-foot { font-size: 12px; color: #4B5563; overflow-wrap: anywhere; }
  [class*="st-key-results_dir"] [data-baseweb="select"] div { white-space: normal; overflow-wrap: anywhere; text-overflow: clip; overflow: visible; }
  [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: #4B5563; }
</style>"""
# One block, no comments or blank lines: Streamlit Cloud (newer Markdown renderer) shows a <style> block as text
# if a blank line ends the HTML block. st.html is used when this Streamlit has it.
if hasattr(st, "html"):
    st.html(_CSS)
else:
    st.markdown(_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ results folder
def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def results_choices() -> list:
    """results/, every results/results_* subfolder, every results_* folder at the repo root,
    plus the ED_RESULTS_DIR folder if it is somewhere else."""
    base = ROOT / "results"
    opts = ["results"] + sorted(f"results/{p.name}" for p in base.glob("results_*") if p.is_dir())
    opts += sorted(p.name for p in ROOT.glob("results_*") if p.is_dir())
    env = os.environ.get("ED_RESULTS_DIR", "").strip()
    if env:
        envp = Path(env) if Path(env).is_absolute() else ROOT / env
        rel = _rel(envp)
        if rel not in opts:
            opts.insert(0, rel)
    return opts


def default_results() -> str:
    env = os.environ.get("ED_RESULTS_DIR", "").strip()
    if not env:
        return "results"
    return _rel(Path(env) if Path(env).is_absolute() else ROOT / env)


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


@st.cache_data(show_spinner=False)
def pressure_point(cfg_d: dict, scenario: str, intensity, burst: bool, seed0: int) -> str:
    """Which resource limits a 'do nothing' day (first simulated day) for this hospital."""
    cfg = _cfg(cfg_d)
    truth, _ = generate_day(scenario, truth_seed(seed0), intensity, burst)
    return binding_constraint(run_day(cfg, truth))


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
    res = plan_search(scenario, n_seeds=n_seeds, quick=False, api=registry.api(), burst=burst, runner=runner_for(base))
    return res.to_dict()


@st.cache_data(show_spinner="Checking the next 4 hours...")
def next_four_hours(scenario: str, minute: float, seed: int, results_dir: str) -> dict:
    """The live demo's rulebook (src/demo/live_demo.py) on the next 4 hours of the reference hospital.
    Its whole-day comparison is read from the chosen results folder."""
    from src.demo import live_demo
    live_demo.RESULTS = Path(results_dir) / "plan_search"      # read-only redirect; the module file is not changed
    v = live_demo.compute_view(scenario, minute, seed)
    d = v["decision"]
    return {"label": d.label, "is_nothing": d.plan.is_nothing, "rule": d.rule, "reason": d.reason,
            "gain": d.gain, "gain_lo": d.gain_lo, "gain_hi": d.gain_hi, "cost": d.plan.cost, "staff_hours": d.plan.staff_hours,
            "base_on_time": d.base["urgent_on_time_pct"], "after_on_time": d.after["urgent_on_time_pct"],
            "base_lwbs": d.base["lwbs"], "after_lwbs": d.after["lwbs"], "now": d.now_state,
            "whole_day": d.whole_day, "differ_note": d.differ_note, "hospital": live_demo.reference_name(),
            "horizon_h": live_demo.HORIZON_MIN / 60.0}


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


def csv_row(path: Path, scenario: str):
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    m = df[df["scenario"] == scenario] if "scenario" in df.columns else df.iloc[0:0]
    return None if m.empty else m.iloc[0].to_dict()


def clock(h: float) -> str:
    m = int(round(h * 60)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def money(cost, hours) -> str:
    return f"${cost:,.0f} = {hours:g} staff-hours"


def esc(x) -> str:
    return html.escape(str(x))


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="po-h2">{esc(title)}</div>' + (f'<div class="po-h2-sub">{esc(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)


def delta_pill(diff: float, unit: str, fmt: str = "{:+.1f}") -> str:
    """Neutral grey when unchanged; red when worse; green when better. For every KPI here, higher = worse."""
    if abs(diff) < 0.05:
        return '<span class="po-delta po-same">Same as normal day</span>'
    cls, arrow = ("po-worse", "&#9650;") if diff > 0 else ("po-better", "&#9660;")
    return f'<span class="po-delta {cls}">{arrow} {fmt.format(diff)}{unit} vs normal day</span>'


def split_pp(pp: str) -> tuple:
    """'beds (96% busy)' -> ('Beds', '96% busy')."""
    if pp.endswith(")") and " (" in pp:
        head, tail = pp.rsplit(" (", 1)
        return head.capitalize(), tail[:-1]
    return pp.capitalize(), ""


# ------------------------------------------------------------------ scenario state (sidebar select box + main picker, kept in sync)
if "scen_sel" not in st.session_state:
    q = st.query_params.get("scenario", "normal")
    st.session_state.scen_sel = KEY2LABEL[q if q in KEY2LABEL else "normal"]


def _from_seg(key: str):
    """Main-page day picker -> the sidebar select box (the one source of truth)."""
    v = st.session_state.get(key)
    if v in KEY2LABEL:
        st.session_state.scen_sel = KEY2LABEL[v]


# ------------------------------------------------------------------ sidebar (order of the select boxes is used by the tests)
sb = st.sidebar
sb.header("Your hospital")
sb.caption("These boxes drive the live simulation and the early warning. Recorded results always use the reference hospital.")
preset = sb.selectbox("Start from a preset", list(PRESETS), index=2,
                      help="Small, Medium and Large are example hospitals. Pick one, then change any box below. "
                           "Large is the one we calibrated against survey data.")
b = PRESETS[preset]
k = preset
beds = sb.number_input("Treatment beds", 1, 300, b.beds, key=f"beds_{k}", help="Beds where patients are treated. Each patient seen by a doctor holds one bed until they leave.")
rooms = sb.number_input("Treatment rooms (0 = no limit)", 0, 300, 0, key=f"rooms_{k}", help="If set, usable beds = the smaller of beds and rooms (a bed needs a room).")
stations = sb.number_input("Triage stations (0 = no limit)", 0, 50, 0, key=f"stations_{k}", help="If set, triage capacity = the smaller of stations and triage nurses.")
nurses = sb.number_input("Triage nurses", 1, 30, b.triage_nurses, key=f"nurses_{k}", help="Nurses doing the first quick assessment of each patient.")
doctors = sb.number_input("Doctors on at once", 1, 40, b.doctors, key=f"doctors_{k}", help="Doctors seeing patients at the same time.")
bays = sb.number_input("Fast-track bays", 0, 10, 0, key=f"bays_{k}",
                       help=f"0 = off. 1 or more = fast track on all day: low-urgency patients (levels 4-5) need {round((1 - FT_FACTOR) * 100)}% "
                            "less doctor time (an ASSUMPTION). More than one bay does not add more effect in this model.")
cut_label = sb.selectbox("Ward-transfer speed", ["Normal", "25% faster transfers", "50% faster transfers"], key=f"cut_{k}",
                         help="When ward beds are free, admitted patients leave the ED sooner and free their bed. 'Faster' cuts their waiting time in the ED by that share, all day.")
extra = sb.checkbox("Add an extra doctor for 6 hours", key=f"extra_{k}", help="One additional doctor on shift for 6 hours.")
start_h = sb.slider("Extra doctor starts at (hour)", 0, 18, 12, key=f"start_{k}", disabled=not extra, help="Hour of the day the extra shift begins.")

sb.header("The day")
scen_label = sb.selectbox("Day type", list(SCENARIOS), key="scen_sel",
                          help="Same choice as the buttons at the top of the page.")
scenario = SCENARIOS[scen_label]
intensity = {}
for comp in registry.info(scenario)["components"]:       # one slider per cited component, limited to its published range
    lo, hi = round((SURGE_RANGES[comp]["low"] - 1) * 100), round((SURGE_RANGES[comp]["high"] - 1) * 100)
    pct = sb.slider(f"{comp.capitalize()} surge size (% extra patients)", lo, hi, (lo + hi) // 2, key=f"surge_{comp}",
                    help=f"Limited to the published range (+{lo}% to +{hi}%). It cannot be set outside what studies reported.")
    intensity[comp] = pct / 100.0
intensity = intensity or None
built_in_burst = registry.REGISTRY[scenario]["burst"]
burst = sb.checkbox("Add a sudden burst of very urgent patients", value=built_in_burst, disabled=scenario == "normal" or built_in_burst,
                    help=f"Stress test: {BURST_SIZE_RANGE[0]}-{BURST_SIZE_RANGE[1]} level 1-2 patients arrive within one hour. "
                         "This size is UNSOURCED (our assumption), not from a study.")
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

sb.header("Recorded results")
choices = results_choices()
dflt = default_results()
res_choice = sb.selectbox("Results folder to read", choices, index=choices.index(dflt) if dflt in choices else 0,
                      help="Saved plan searches, scenario summary and run records are read from this folder. "
                           "Start the page with ED_RESULTS_DIR=<folder> to pick one by default. Model files always come from models/.",
                      key="results_dir")
RES = Path(res_choice) if Path(res_choice).is_absolute() else ROOT / res_choice
sb.markdown(f'<div class="po-foot">Reading recorded results from: <b>{esc(_rel(RES))}</b></div>', unsafe_allow_html=True)

cfg_d = dataclasses.asdict(cfg)
base_d = dataclasses.asdict(dataclasses.replace(b, patient_flow=flow_ok))

# ------------------------------------------------------------------ header
lx = jload(RES / "linuxone_run.json")
lx_bad = None
if lx and lx.get("uname_m") != "s390x":     # only a run that really happened on s390x counts as LinuxONE
    lx_bad, lx = lx.get("uname_m"), None
lx_pill = ""
if lx:
    tip = esc(f"From {_rel(RES)}/linuxone_run.json: machine {lx.get('uname_m')}, {lx.get('nproc')} cores, "
              f"total {lx.get('total_seconds')} s, tests: {lx.get('tests_summary')}")
    lx_pill = f'<span class="po-pill po-lx" title="{tip}"><span class="po-dot"></span>Ran on IBM LinuxONE (s390x)</span>'
st.markdown(f'<div class="po-hdr"><div class="po-brand"><span class="po-word">{PRODUCT}</span><span class="po-sub">{SUBTITLE}</span></div>'
            f'<div class="po-pills">{lx_pill}</div></div>', unsafe_allow_html=True)
st.caption(CAVEAT_LINE)
with st.expander("About these numbers"):
    st.caption(CAVEAT)
    st.caption(TRIAGE_LEVEL_NOTE)
    st.markdown(
        "- **Urgent seen late** = share of urgent patients (triage levels 1-3) who waited longer than their target time.\n"
        "- **Left without being seen (LWBS)** = patients who gave up waiting. **Peak bed use** = the fullest the treatment beds got.\n"
        "- **Recorded results** come from a full saved run (many simulated days). **Try it** below runs a short simulation on this computer.\n"
        "- The same random days are used for every comparison, so differences are not luck.\n"
        "- **Recommended plan** is shown only when its benefit stays clearly above zero across many days; otherwise the advice is 'do nothing'.\n"
        "- **Early warning** shows the chance that something breaks in the next 4 hours. It is learned from the simulator, not from a real hospital.")

# ------------------------------------------------------------------ day picker
st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
# A horizontal radio styled as a segmented control (st.segmented_control in single mode is not supported by AppTest).
seg_key = f"scen_pick_{scenario}"       # re-created whenever the sidebar box changes, so both always agree
st.radio("Day type", list(KEY2LABEL), index=list(KEY2LABEL).index(scenario), format_func=lambda x: SHORT.get(x, x),
         key=seg_key, on_change=_from_seg, args=(seg_key,), horizontal=True)
_desc = registry.description(scenario)
_first = _desc.split(". ")[0].rstrip(".") + "."
st.caption(_first)
if _first != _desc:
    with st.expander("Source and assumptions for this day type"):
        st.markdown(_desc)
st.query_params["scenario"] = scenario

# ------------------------------------------------------------------ 1. recorded KPI cards
rec_row = csv_row(RES / "scenario_summary.csv", scenario)
norm_row = csv_row(RES / "scenario_summary.csv", "normal")
if rec_row:
    section(f"{KEY2LABEL[scenario]}: recorded results",
            f"Full saved run: {rec_row.get('hospital')} reference hospital, {rec_row.get('days')} simulated days, "
            f"about {float(rec_row.get('patients_per_day', 0)):.0f} patients a day, default surge size.")
    kpis = [("Urgent patients seen late", "urgent_late_pct", "{:.1f}%", "of urgent patients (levels 1-3) waited past their target", " pts"),
            ("Left without being seen", "lwbs_pct", "{:.1f}%", "of all patients gave up and left", " pts"),
            ("Peak bed use", "peak_bed_occupancy_pct", "{:.0f}%", "of treatment beds taken at the fullest moment", " pts"),
            ("Longest queue", "longest_queue", "{:.0f}", "patients waiting at once, at the worst moment", "")]
    cards = []
    for label, key, fmt, cap, unit in kpis:
        v = float(rec_row[key])
        d = delta_pill(v - float(norm_row[key]), unit) if norm_row and scenario != "normal" else (
            '<span class="po-delta po-same">This is the normal day</span>' if scenario == "normal" else "")
        cards.append(f'<div class="po-kpi"><div class="po-kpi-label">{label}</div><div class="po-kpi-value">{fmt.format(v)}</div>'
                     f'<div class="po-kpi-cap">{cap}</div>{d}</div>')
    st.markdown(f'<div class="po-kpis">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.caption(f"Source: {_rel(RES)}/scenario_summary.csv. Changes compare with the normal-day row of the same file "
               "(pts = percentage points).")
else:
    section(f"{KEY2LABEL[scenario]}: recorded results")
    st.info(f"No recorded row for this day type in {_rel(RES)}/scenario_summary.csv. Pick another results folder in the sidebar.")

# ------------------------------------------------------------------ 2. recommended plan
section("Recommended plan")
ps = jload(RES / "plan_search" / f"{scenario}.json")
if ps:
    rec, tc, sel = ps.get("recommended", {}), ps.get("truth_check", {}), ps.get("selected", {})
    nothing = rec.get("label") == "Do nothing"
    ref = PRESETS.get(ps.get("hospital", ""), PRESETS["Large"])
    ref_pp = pressure_point(dataclasses.asdict(dataclasses.replace(ref, patient_flow=flow_ok)), scenario, None, built_in_burst, 0)
    pp_name, pp_detail = split_pp(ref_pp)
    n_days = tc.get("n_seeds", ps.get("n_seeds"))
    rng = (f"95% range {tc['gain_lo']:+.1f} to {tc['gain_hi']:+.1f}" if tc.get("gain_lo") is not None else "")
    pp_block = (f'<div><div class="po-kpi-label">Main pressure point</div><div class="po-rec-val">{esc(pp_name)}</div>'
                f'<div class="po-kpi-cap">{esc(pp_detail)}{"; " if pp_detail else ""}with no extra action, one simulated day</div></div>')
    if not nothing:
        grid = (f'<div><div class="po-kpi-label" title="{esc(HELP_RANGE)}">Expected gain &#9432;</div>'
                f'<div class="po-rec-val">{tc.get("gain", 0):+.1f} points</div>'
                f'<div class="po-kpi-cap">{rng}. 1 point = 1% more urgent patients seen on time</div></div>'
                f'<div><div class="po-kpi-label" title="{esc(COST_NOTE)}">Cost &#9432;</div><div class="po-rec-val">${rec.get("cost", 0):,.0f}</div>'
                f'<div class="po-kpi-cap">= {rec.get("staff_hours", 0):g} staff-hours. Illustrative, unsourced cost</div></div>' + pp_block)
        effect = (f'Urgent patients seen on time <b>{tc["base_on_time_pct"]:.1f}% &rarr; {tc["plan_on_time_pct"]:.1f}%</b> &middot; '
                  f'Left unseen per day <b>{tc["base_lwbs"]:.1f} &rarr; {tc["plan_lwbs"]:.1f}</b> &middot; '
                  f'Non-urgent median wait <b>{tc["base_nonurgent_median"]:.0f} &rarr; {tc["plan_nonurgent_median"]:.0f} min</b>'
                  + (" (fairness limit holds)" if tc.get("fairness_ok") else " (fairness limit NOT met)")) if tc else ""
        tag, title, cls = "Recommended plan &middot; whole day", rec.get("label", ""), ""
    else:
        grid = (f'<div><div class="po-kpi-label" title="{esc(HELP_RANGE)}">Best plan tested &#9432;</div>'
                f'<div class="po-rec-val">{esc(sel.get("label", "?"))}</div>'
                f'<div class="po-kpi-cap">gain {tc.get("gain", 0):+.1f} points, {rng}; a plan needs the whole range above {MIN_BENEFIT_PP:g} points</div></div>'
                f'<div><div class="po-kpi-label">Urgent seen on time, doing nothing</div>'
                f'<div class="po-rec-val">{tc.get("base_on_time_pct", 0):.1f}%</div><div class="po-kpi-cap">of urgent patients (levels 1-3)</div></div>' + pp_block)
        effect = "No tested plan helps clearly enough on this day type, so the advice is to keep the normal staffing."
        tag, title, cls = "No extra action needed", "Do nothing extra", " po-none"
    note = (f'&#9432; Whole-day plan for the {esc(ps.get("hospital", "?"))} reference hospital, tested on {n_days} simulated days '
            f'(file: {esc(_rel(RES))}/plan_search/{esc(scenario)}.json). The "next 4 hours" check below looks only 4 hours ahead '
            f'from a chosen time, so its advice can differ.')
    st.markdown(f'<div class="po-rec{cls}"><div class="po-rec-tag">{tag}</div><div class="po-rec-title">{esc(title)}</div>'
                f'<div class="po-rec-grid">{grid}</div><div class="po-rec-effect">{effect}</div>'
                f'<div class="po-rec-note">{note}</div></div>', unsafe_allow_html=True)
    with st.expander("Plan details (rule, caution, cost note)"):
        st.write(ps.get("reason", ""))
        if ps.get("caution"):
            st.caption(ps["caution"])
        st.caption(f"Main pressure point rule of thumb (an UNSOURCED assumption): boarding if admitted patients waiting for a ward "
                   f"hold 25% or more of beds; otherwise a resource busy 90% or more of the time; otherwise the busiest resource. "
                   f"Read live from the first simulated day of the {ps.get('hospital', '?')} hospital: {ref_pp}.")
        st.caption(COST_NOTE)
else:
    st.info(f"No saved whole-day plan search for '{KEY2LABEL[scenario]}' in {_rel(RES)}/plan_search/. "
            "Use the next-4-hours check below, or 'Search plans for your own hospital' further down.")

with st.expander("Next 4 hours vs the whole day"):
    st.caption("Applies the same rules to the next 4 hours only, from a moment in the reference hospital's day, using the advance "
               "forecast (never the true future). Default surge size.")
    t_now = st.slider("Time now", 0.0, 20.0, 10.0, 0.5, format="%.1f h", key="t_now",
                      help="Hour of the day to look ahead from (10.5 = 10:30).")
    try:
        nf = next_four_hours(scenario, t_now * 60.0, 42 + seed0, str(RES))
        hz = f"{nf['horizon_h']:g} h"
        if nf["is_nothing"]:
            st.markdown(f"**Next {hz} from {clock(t_now)}: no action.** {nf['reason']}")
        else:
            st.markdown(f"**Next {hz} from {clock(t_now)}: {nf['label']}.** Urgent seen on time {nf['base_on_time']:.1f}% -> "
                        f"{nf['after_on_time']:.1f}%; left unseen {nf['base_lwbs']:.1f} -> {nf['after_lwbs']:.1f}. "
                        f"Gain {nf['gain']:+.1f} points (95% range {nf['gain_lo']:+.1f} to {nf['gain_hi']:+.1f}); cost "
                        f"{money(nf['cost'], nf['staff_hours'])} (illustrative, UNSOURCED).", help=HELP_RANGE)
            st.caption(nf["reason"])
        s = nf["now"]
        st.caption(f"At {clock(t_now)}: {s['beds_busy']}/{s['beds_cap']} beds in use, {s['admitted_boarders']} admitted patients waiting "
                   f"for a ward bed, {s['waiting']} waiting to be seen ({s['waiting_urgent']} urgent). {nf['hospital']} hospital.")
        w = nf["whole_day"]
        if w:
            st.markdown(f"**Whole day:** {w['label']} ({w['why']}).")
            if nf["differ_note"]:
                st.warning(nf["differ_note"])
            else:
                st.caption("The next-4-hours view and the whole-day plan agree.")
        else:
            st.caption("No whole-day plan for this day type in the chosen results folder.")
    except Exception as err:     # never break the page
        st.warning(f"The next-4-hours check could not run: {err}")

# ------------------------------------------------------------------ 3. early warning
section("Early warning: what could break in the next 4 hours",
        "Chance of each problem in the next 4 hours, read every 30 minutes through one simulated day of the hospital in the left panel.")
if cfg.effective_beds != b.effective_beds or cfg.doctors != b.doctors or cfg.triage_capacity != b.triage_capacity:
    st.info("You changed the hospital size. The early-warning model was trained on the three preset sizes only, so read its numbers as rough.")
try:
    rt = risk_timeline(cfg_d, scenario, intensity, burst, seed0)
except Exception as err:    # a model or library problem must never break the page
    rt = []
    st.warning(f"Early warning is not available here: {err}")
if rt and rt[0].get("available"):
    at = st.slider("Look at the hour", 0.5, 20.0, 14.0, 0.5, format="%.1f h",
                   help="Pick a time of day to read the warning at that moment (first simulated day).")
    row = min(rt, key=lambda r: abs(r["hour"] - at))

    def _tile(v, label):
        lvl = " hi" if v >= 0.5 else (" mid" if v >= 0.2 else "")
        return f'<div><div class="v{lvl}">{100 * v:.0f}%</div><div class="l">{label}</div></div>'

    st.markdown(f'<div class="po-risk">{_tile(row["risk_urgent_late"], "Urgent patients seen late")}'
                f'{_tile(row["risk_beds_full"], "Beds above 95% full")}{_tile(row["risk_lwbs"], "Patients leaving unseen")}</div>',
                unsafe_allow_html=True)
    st.markdown(f"**At {clock(row['hour'])}, breaks first:** {row['breaks_first']}. {row['message']}")
    rdf = pd.DataFrame(rt).set_index("hour")[["risk_urgent_late", "risk_beds_full", "risk_lwbs"]]
    rdf.columns = ["Urgent seen late", "Beds above 95% full", "Leaving unseen"]
    rdf = rdf[sorted(rdf.columns)]          # Streamlit colours series in name order
    rdf.index.name = "Hour of day"
    st.line_chart(rdf, color=[BLUE, AMBER, RED], height=240, y_label="chance (0-1)")
    st.caption(MODEL_CAVEAT)
elif rt:
    st.info(rt[0]["message"])

# ------------------------------------------------------------------ 4. live simulation
chosen = mean_metrics(simulate_many(cfg_d, scenario, intensity, burst, seed0, runs))
normal = mean_metrics(simulate_many(cfg_d, "normal", None, False, seed0, runs))
section("Try it: live simulation (short run)",
        f"Runs now on this computer for the hospital in the left panel: average of {runs} simulated days"
        + (" (" + ", ".join(f"{c} +{round(v * 100)}%" for c, v in intensity.items()) + ")" if intensity else "")
        + ". Change any box on the left to see what happens. It can differ from the recorded results above.")


def live_metric(col, label, value, diff, unit, fmt, help_):
    if scenario == "normal" or abs(diff) < 0.05:
        col.metric(label, value, "same as normal day" if scenario != "normal" else None, delta_color="off", help=help_)
    else:
        col.metric(label, value, f"{fmt.format(diff)}{unit}", delta_color="inverse", help=help_)


c = st.columns(4)
live_metric(c[0], "Urgent seen late", f"{chosen['urgent_late_pct']:.1f}%", chosen["urgent_late_pct"] - normal["urgent_late_pct"], " pts", "{:+.1f}", HELP_URGENT)
live_metric(c[1], "Left without being seen", f"{chosen['lwbs_pct']:.1f}%", chosen["lwbs_pct"] - normal["lwbs_pct"], " pts", "{:+.1f}", HELP_LWBS)
live_metric(c[2], "Peak bed use", f"{chosen['peak_bed']:.0f}%", chosen["peak_bed"] - normal["peak_bed"], " pts", "{:+.0f}", HELP_BEDS)
live_metric(c[3], "Longest queue", f"{chosen['longest_queue']:.0f}", chosen["longest_queue"] - normal["longest_queue"], " patients", "{:+.0f}",
            "The most patients waiting at the same moment (all queues).")
pp = pressure_point(cfg_d, scenario, intensity, burst, seed0)
st.caption(("Arrows: change vs a normal day for the same hospital (pts = percentage points). " if scenario != "normal" else "")
           + f"About {chosen['patients']:.0f} patients a day. Main pressure point with no extra action (first simulated day): {pp}.")

two = scenario != "normal"
hrs = chosen["hour"]


def series(a, b_):
    d = {"Hour of day": hrs, "Chosen day": a}
    if two:
        d["Normal day"] = b_[:len(hrs)]
    return pd.DataFrame(d).set_index("Hour of day")


wt = pd.DataFrame({"Level": [f"Level {l}" for l in range(1, 6)],
                   "Target": ["immediate" if ATS_TARGET[l] == 0 else f"{ATS_TARGET[l]:.0f} min" for l in range(1, 6)],
                   "Chosen day (min)": [round(chosen["wait"][l], 1) for l in range(1, 6)],
                   "Normal day (min)": [round(normal["wait"][l], 1) for l in range(1, 6)]})
ch, cb = series(chosen["queue"], normal["queue"]), series(chosen["bed_pct"], normal["bed_pct"])
colors = [BLUE, GREY_LINE] if two else [BLUE]
a1c, a2c = st.columns(2)
a1c.markdown("**Patients waiting (all queues)**")
a1c.line_chart(ch, color=colors, height=240, y_label="patients")
a2c.markdown("**Beds in use (%)**")
a2c.line_chart(cb, color=colors, height=240, y_label="% of beds")
if not two:
    wt = wt.drop(columns=["Normal day (min)"])
b1, b2 = st.columns([1, 1])
b1.markdown("**Median wait by triage level (minutes)**", help=HELP_URGENT)
b1.bar_chart(wt.set_index("Level")[["Chosen day (min)", "Normal day (min)"]] if two else wt.set_index("Level")[["Chosen day (min)"]],
             color=colors, height=240, stack=False, horizontal=True, x_label="minutes", y_label="")
b2.markdown("**Median wait table**")
b2.dataframe(wt, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 5. details
section("Details")

with st.expander("Search plans for your own hospital"):
    st.caption(f"Tests an extra doctor for 6 hours, fast track and faster ward transfers, at different start times, on many simulated days built from the advance forecast only. "
               f"A plan is recommended only if the low end of its 95% benefit range stays above {MIN_BENEFIT_PP:g} points (an ASSUMPTION) and non-urgent patients do not wait "
               f"unfairly longer (median wait up at most {DEFAULT_SETTINGS.fairness_limit * 100:.0f}%). "
               "The search starts from your beds, staff and fast-track setting; it adds its own extra shift and transfer speed-up.")
    n_seeds = st.slider("Days to test each plan on", 10, 50, 30, help="More days = a firmer answer but a longer wait (about 10-40 seconds).")
    if st.button("Find the best plan for this day type", type="primary"):
        try:
            rd = search_plan(cfg_d, scenario, burst, n_seeds)
            rec = rd["recommended"]
            t = rd["truth_check"]
            if rec["label"] == "Do nothing":
                st.info("Recommendation: **do nothing extra.** " + rd["reason"])
            else:
                st.success(f"Recommendation: **{rec['label']}**  (cost {money(rec['cost'], rec.get('staff_hours', 0))}, illustrative). " + rd["reason"])
                st.write(f"Urgent patients on time: {t['base_on_time_pct']:.1f}% -> {t['plan_on_time_pct']:.1f}%. "
                         f"Non-urgent median wait: {t['base_nonurgent_median']:.1f} -> {t['plan_nonurgent_median']:.1f} min ({t['nonurgent_change_pct']:+.0f}%). "
                         f"Left unseen per day: {t['base_lwbs']:.1f} -> {t['plan_lwbs']:.1f}.")
            if rd.get("caution"):
                st.caption(rd["caution"])
            tab = pd.DataFrame(rd["rows"]).sort_values("score", ascending=False).head(8)
            keep = [c_ for c_ in ("plan", "urgent_on_time_pct", "nonurgent_median_wait_min", "lwbs", "staff_hours", "cost_usd", "score", "lookahead_gain", "feasible") if c_ in tab.columns]
            st.dataframe(tab[keep].round(2), hide_index=True, use_container_width=True)
            st.caption(COST_NOTE)
        except Exception as err:
            st.warning(f"The plan search could not run: {err}")
    else:
        st.caption("Press the button to run the search.")

with st.expander(f"Your hospital vs the '{preset}' preset (same days)"):
    pm = mean_metrics(simulate_many(base_d, scenario, intensity, burst, seed0, runs))
    cmp_ = pd.DataFrame({"Measure": ["Urgent seen late (%)", "Left without being seen (%)", "Peak bed use (%)", "Longest queue (patients)", "Patients that day"],
                         "Your hospital": [chosen["urgent_late_pct"], chosen["lwbs_pct"], chosen["peak_bed"], chosen["longest_queue"], chosen["patients"]],
                         f"{preset} preset": [pm["urgent_late_pct"], pm["lwbs_pct"], pm["peak_bed"], pm["longest_queue"], pm["patients"]]}).round(1)
    st.dataframe(cmp_, hide_index=True, use_container_width=True)

with st.expander("Where every number comes from"):
    st.caption("DATA = measured in survey data we processed. EVIDENCE = a published study (cited by the team, not independently checked by us). ASSUMPTION = our own stated choice.")
    rows = [{"name": p["name"], "value": json.dumps(p["value"], default=str)[:140], "tag": p["tag"], "status": p["status"], "note": p["note"]} for p in PARAMETER_TABLE + OPTIMISER_PARAMETERS + HOSPITAL_PARAMETERS]
    ptab = pd.DataFrame(rows)
    tag_filter = st.multiselect("Show tags", ["DATA", "EVIDENCE", "ASSUMPTION"], default=["DATA", "EVIDENCE", "ASSUMPTION"])
    st.dataframe(ptab[ptab["tag"].isin(tag_filter)], hide_index=True, use_container_width=True)
    demo = jload(RES / "demo_results.json") or {}
    limits = demo.get("limits")
    if limits:
        st.caption("Data limits (from " + _rel(RES) + "/demo_results.json): " + " ".join(limits))
    st.caption("Weather, FEMA and hurricane records are context only and never set a surge size.")

with st.expander("How accurate are the models?"):
    metrics = jload(ROOT / "models/metrics.json")
    if not metrics:
        st.info("No model results found yet (models/metrics.json). Run the training scripts.")
    else:
        pf = metrics.get("patient_flow")
        if pf:
            t = pf["admission"]["test"]
            st.markdown(f"**Model 1: patient flow (real survey data).** Admission model chosen: **{pf['admission']['chosen']['family']}**. "
                        f"AUC on hospitals it never saw: **{t['chosen_model']['auc_roc']:.3f}** "
                        f"(95% {t['chosen_model']['ci95']['auc_roc'][0]:.3f} to {t['chosen_model']['ci95']['auc_roc'][1]:.3f}) vs {t['baseline_triage_only']['auc_roc']:.3f} for 'triage level only'. "
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
            st.markdown("**Stage-by-stage results** (stage 1: normal only; 2: unseen surge; 3: retrained with surge)")
            st.dataframe(sdf[cols].round(3), hide_index=True, use_container_width=True)
        img = RES / "stage_chart.png"
        if img.exists():
            st.image(str(img), caption=f"{_rel(RES)}/stage_chart.png")

with st.expander("LinuxONE run record"):
    if lx_bad is not None:
        st.warning(f"{_rel(RES)}/linuxone_run.json says machine {lx_bad!r}, not s390x, so it is NOT shown as a LinuxONE result.")
    if lx:
        st.success(f"Ran on IBM LinuxONE (s390x): recorded in {_rel(RES)}/linuxone_run.json.")
        m = st.columns(3)
        m[0].metric("Cores (nproc)", f"{lx.get('nproc')}")
        m[1].metric("Total run time", f"{lx.get('total_seconds')} s", help="Wall-clock time of the whole recorded run on that machine.")
        m[2].metric("Python", f"{lx.get('python')}")
        if lx.get("tests_summary"):
            st.write(f"**Tests on that machine:** {lx['tests_summary']}")
        steps = lx.get("seconds_per_step", {})
        l1, l2 = st.columns(2)
        if steps:
            l1.markdown("**Seconds per step**")
            l1.dataframe(pd.DataFrame({"step": list(steps), "seconds": list(steps.values())}), hide_index=True, use_container_width=True)
        tm = lx.get("parallel_timing")
        if tm:
            l2.markdown("**Parallel data generation** (more workers is not always faster for small jobs)")
            l2.dataframe(pd.DataFrame(tm), hide_index=True, use_container_width=True)
    else:
        st.info(f"No LinuxONE run recorded yet in {_rel(RES)}.")
        loc = jload(RES / "local_run.json")
        if loc:
            st.caption(f"A local (non-LinuxONE) test run exists: machine {loc.get('uname_m')}, {loc.get('nproc')} cores. It is not shown as a LinuxONE result.")
    st.caption("This dashboard itself is a normal Streamlit page; it is not claimed to run on LinuxONE.")
