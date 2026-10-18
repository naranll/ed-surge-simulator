"""Emergency Department Real-Time Decision Support System & Digital Twin.
High-performance clinical operations platform for surge management and capacity optimization.
Run: streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explainer.briefing import build_briefing  # noqa: E402
from src.optimization.optimizer import recommend, pretty, OPTIMISER_PARAMETERS  # noqa: E402
from src.simulation.ed_simulator import Action, LiveED, simulate  # noqa: E402
from src.simulation.hospital import HOSPITAL_PARAMETERS, PRESETS, HospitalConfig  # noqa: E402
from src.simulation.scenarios import ATS_TARGET, PARAMETER_TABLE, SCENARIOS, scenario_info  # noqa: E402

# Clinical Triage Standards (Australasian Triage Scale / Emergency Severity Index)
ATS_CONFIG = {
    1: {"name": "ATS 1", "desc": "Resuscitation", "color": "#dc2626", "bg": "#fef2f2", "border": "#fecaca", "target": 0},
    2: {"name": "ATS 2", "desc": "Emergency", "color": "#ea580c", "bg": "#fff7ed", "border": "#fed7aa", "target": 10},
    3: {"name": "ATS 3", "desc": "Urgent", "color": "#d97706", "bg": "#fffbeb", "border": "#fde68a", "target": 30},
    4: {"name": "ATS 4", "desc": "Semi-Urgent", "color": "#16a34a", "bg": "#f0fdf4", "border": "#bbf7d0", "target": 60},
    5: {"name": "ATS 5", "desc": "Non-Urgent", "color": "#0284c7", "bg": "#f0f9ff", "border": "#bae6fd", "target": 120},
}

SCENARIO_PROFILES = {
    "normal": {
        "title": "Baseline Monday Operations",
        "badge": "STANDARD OPERATIONS",
        "badge_bg": "#f1f5f9",
        "badge_color": "#334155",
        "border": "#cbd5e1",
        "desc": "Typical diurnal arrival pattern (~200 visits/24h) with standard scheduled staffing. Baseline comparison benchmark.",
    },
    "heatwave": {
        "title": "Extreme Heatwave Surge (Texas 2011 & 2025)",
        "badge": "WEATHER ADVISORY: EXTREME HEAT",
        "badge_bg": "#fffbeb",
        "badge_color": "#b45309",
        "border": "#fde68a",
        "desc": "Nature Health 2025 / CDC Model: 104°F sustained heat. +8.9% elderly (65+) surge with severe dehydration and cardiovascular exacerbations.",
    },
    "hurricane": {
        "title": "Hurricane Evacuee Surge (Harvey Regional Network)",
        "badge": "DISASTER SURGE: HURRICANE INFLUX",
        "badge_bg": "#eff6ff",
        "badge_color": "#1d4ed8",
        "border": "#bfdbfe",
        "desc": "Disaster Medicine Model: +11% sustained demand with +40% afternoon evacuee influx presenting with displaced chronic care, injuries, and medication refills.",
    },
    "compound": {
        "title": "Compound Calamity (Hurricane Beryl 2024)",
        "badge": "COMPOUND DISASTER: STORM + OUTAGE + HEAT",
        "badge_bg": "#fdf2f8",
        "badge_color": "#be185d",
        "border": "#fbcfe8",
        "desc": "Category 1 Hurricane landfall followed by 105°F heat index and 2.2M grid power failure. +35% volume, oxygen/device dependency failures, and acute inpatient boarding gridlock.",
    },
}

PROTOCOLS = [
    "Surge Nurse (+1)",
    "On-Call Attending (+1)",
    "Fast-Track Protocol",
    "Expedite Ward Transfers",
]

HOSPITAL_PRESET_OPTIONS = {
    "Baseline Unit (20 Beds, 5 Docs)": {
        "beds": 20, "docs": 5, "nurses": 2, "scale": 1.0,
        "tag": "Standard Acute Care Benchmark",
        "desc": "Standard single-unit ED: 20 treatment bays, 5 attendings, 2 triage nurses (~200 visits/day).",
    },
    "Small Hospital (18 Beds, 2 Docs)": {
        "beds": 18, "docs": 2, "nurses": 1, "scale": 0.3,
        "tag": "Rural / Critical Access Hospital",
        "desc": "Small community ED: 18 treatment beds, 2 doctors, 1 triage nurse (~60 visits/day).",
    },
    "Medium Hospital (36 Beds, 3 Docs)": {
        "beds": 36, "docs": 3, "nurses": 2, "scale": 0.6,
        "tag": "Suburban Regional Facility",
        "desc": "Medium regional hospital: 36 treatment beds, 3 doctors, 2 triage nurses (~120 visits/day).",
    },
    "Large Hospital (60 Beds, 4 Docs)": {
        "beds": 60, "docs": 4, "nurses": 3, "scale": 1.0,
        "tag": "Tertiary Academic Center (Calibrated)",
        "desc": "Large tertiary academic hospital: 60 treatment beds, 4 doctors, 3 triage nurses (~200 visits/day calibrated against CDC NHAMCS).",
    },
}


def get_early_warning_risk(snapshot, info: dict, volume_scale: float = 1.0) -> dict:
    """Predict 4-hour forward lookahead risk via Model 2 surrogate with safe fallback."""
    try:
        from models.predict_risk import predict_risk
        return predict_risk(snapshot, info, volume_scale=volume_scale)
    except Exception as exc:
        return {
            "risk_urgent_late": 0.0,
            "risk_beds_full": 0.0,
            "risk_lwbs": 0.0,
            "breaks_first": "unavailable",
            "alarm": False,
            "message": f"Early-warning surrogate not active: {exc}",
            "available": False,
        }


def live_metrics(live: LiveED) -> dict:
    """Extract clinical gauges from the live simulation at the current clock."""
    s, now = live.sim, live.sim.now
    seen = [p for p in s.pat.values() if p.st != "future"]
    wait = lambda p: (p.t_doc if p.t_doc is not None else p.t_lwbs if p.t_lwbs is not None else now) - p.t
    a2 = [p for p in seen if p.ats == 2]
    return {
        "bed_pct": 100.0 * s.busy["b"] / max(1, s.cap["b"]),
        "avg_wait": sum(map(wait, seen)) / len(seen) if seen else 0.0,
        "ats2_breach_pct": 100.0 * sum(wait(p) > ATS_TARGET[2] for p in a2) / len(a2) if a2 else 0.0,
        "arrived": len(seen),
        "lwbs": s.cnt["lwbs"],
        "in_beds": s.busy["b"],
        "total_beds": s.cap["b"],
        "active_docs": s.busy["d"],
        "total_docs": s.cap["d"],
        "active_nurses": s.busy["n"],
        "total_nurses": s.cap["n"],
        "waiting_tri": s.cnt["tri_q"],
        "waiting_bed": s.cnt["bed_q"],
        "boarding_count": s.cnt["board"],
    }


def get_hospital_state(live: LiveED) -> dict:
    s, now = live.sim, live.sim.now

    # Triage Intake Queue
    tri_q = []
    for item in sorted(s.q.get("tri_q", []), key=lambda x: (x[0], x[1])):
        p = s.pat.get(item[2])
        if p and p.st == "tri_q":
            w = max(0.0, now - p.t)
            tgt = ATS_TARGET.get(p.ats, 120)
            tri_q.append({"pid": p.pid, "ats": p.ats, "wait": w, "target": tgt, "breached": w > tgt})

    # Bed Waiting Queue (triaged patients awaiting acute bed placement)
    bed_q = []
    for item in sorted(s.q.get("bed_q", []), key=lambda x: (x[0], x[1])):
        p = s.pat.get(item[2])
        if p and p.st == "bed_q":
            w = max(0.0, now - p.t)
            tgt = ATS_TARGET.get(p.ats, 120)
            bed_q.append({"pid": p.pid, "ats": p.ats, "wait": w, "target": tgt, "breached": w > tgt})

    # Acute Treatment Bays (20 Beds Unit)
    in_beds = [p for p in s.pat.values() if p.st in ("doc_q", "consult", "board")]
    in_beds.sort(key=lambda p: (p.t_bed if p.t_bed is not None else p.t, p.pid))

    attending_staff = [
        "Dr. Chen", "Dr. Patel", "Dr. Rivera", "Dr. Hayes", "Dr. Brooks", "Dr. Zhao (On-Call)"
    ]

    beds = []
    for i in range(s.cap["b"]):
        bay_id = i + 1
        if i < len(in_beds):
            p = in_beds[i]
            assigned_doc = attending_staff[i % len(attending_staff)]
            elapsed = max(0.0, now - p.t)
            tgt = ATS_TARGET.get(p.ats, 120)

            if p.st == "consult":
                status = "consult"
                label = "IN CONSULT"
                sublabel = f"Attending: {assigned_doc}"
            elif p.st == "board":
                if p.admit:
                    status = "boarding"
                    hold_time = max(0.0, now - (p.t_doc or p.t))
                    label = "BOARDING BLOCK"
                    sublabel = f"Admitted • Holding Bed ({int(hold_time)}m)"
                else:
                    status = "turnaround"
                    label = "DISCHARGE BUFFER"
                    sublabel = "Labs & Meds Turnaround"
            else:
                status = "awaiting_doc"
                label = "AWAITING ATTENDING"
                sublabel = "Bed Assigned • Charting"

            beds.append({
                "bay": bay_id,
                "status": status,
                "label": label,
                "sublabel": sublabel,
                "pid": p.pid,
                "ats": p.ats,
                "elapsed": elapsed,
                "breached": (elapsed > tgt) if p.t_doc is None else False,
            })
        else:
            beds.append({
                "bay": bay_id,
                "status": "empty",
                "label": "AVAILABLE",
                "sublabel": "Clean & Ready",
                "pid": None,
                "ats": None,
                "elapsed": 0.0,
                "breached": False,
            })

    # Clinical Staff Rosters
    docs = []
    for d_idx in range(s.cap["d"]):
        name = attending_staff[d_idx % len(attending_staff)]
        busy = d_idx < s.busy["d"]
        docs.append({"id": d_idx + 1, "name": name, "busy": busy})

    nurses = []
    nurse_roster = ["Nurse Sarah (Triage Lead)", "Nurse Marcus (Triage)", "Nurse Elena (Surge Triage)"]
    for n_idx in range(s.cap["n"]):
        name = nurse_roster[n_idx % len(nurse_roster)]
        busy = n_idx < s.busy["n"]
        nurses.append({"id": n_idx + 1, "name": name, "busy": busy})

    return {
        "tri_q": tri_q,
        "bed_q": bed_q,
        "beds": beds,
        "docs": docs,
        "nurses": nurses,
    }


def compute_view(key: str, minute: float, seed: int = 42, workers: int = 1, preset_name: str = "Baseline Unit (20 Beds, 5 Docs)") -> dict:
    sc = SCENARIOS[key]
    cfg = HOSPITAL_PRESET_OPTIONS.get(preset_name, HOSPITAL_PRESET_OPTIONS["Baseline Unit (20 Beds, 5 Docs)"])
    live = LiveED(
        sc,
        seed=seed,
        nurses=cfg["nurses"],
        docs=cfg["docs"],
        beds=cfg["beds"],
        volume_scale=cfg["scale"],
    )
    live.advance_to(minute)
    rec = recommend(live.snapshot(), sc, workers=workers)
    info = scenario_info(key)
    risk = get_early_warning_risk(live.snapshot(), info, volume_scale=cfg["scale"])
    hours = {h: {"arrivals": 0, "beds": [], "queue": []} for h in range(int(minute // 60) + 1)}
    for a in live.arrivals:
        if a.t <= minute:
            hours[int(a.t // 60)]["arrivals"] += 1
    for x in live.sim.samples:
        h = hours.get(int(x["t"] // 60))
        if h is not None:
            h["beds"].append(x["beds"])
            h["queue"].append(x["tri_q"] + x["prov_q"])
    avg = lambda v: sum(v) / len(v) if v else 0.0
    series = [
        {"hour": h, "arrivals": d["arrivals"], "bed_occupancy": avg(d["beds"]), "waiting_queue": avg(d["queue"])}
        for h, d in hours.items()
    ]
    return {
        "scenario": sc,
        "metrics": live_metrics(live),
        "rec": rec,
        "series": series,
        "briefing": build_briefing(sc, rec),
        "hospital": get_hospital_state(live),
        "snapshot": live.snapshot(),
        "risk": risk,
        "preset": cfg,
        "preset_name": preset_name,
    }


def render_hospital_twin(h: dict, m: dict):
    """Render structured, clinical light-theme hospital floor plan."""
    import streamlit as st

    col_triage, col_beds = st.columns([1, 2])

    with col_triage:
        nurse_items = []
        for n in h["nurses"]:
            bg = "#ecfdf5" if n["busy"] else "#f8fafc"
            border = "#a7f3d0" if n["busy"] else "#e2e8f0"
            color = "#047857" if n["busy"] else "#475569"
            status_txt = "Active Intake" if n["busy"] else "Intake Ready"
            pulse = "●" if n["busy"] else "○"
            nurse_items.append(
                f"<div style='background:{bg}; border:1px solid {border}; border-radius:6px; padding:6px 10px; font-size:11px; flex:1; min-width:130px; box-sizing:border-box;'>"
                f"<div style='font-weight:600; color:{color};'><span style='margin-right:4px;'>{pulse}</span>{n['name']}</div>"
                f"<div style='color:#64748b; font-size:10px;'>{status_txt}</div></div>"
            )
        nurse_html = f"<div style='display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap;'>{''.join(nurse_items)}</div>"

        # Triage Queue Chips
        tri_items = []
        for p in h["tri_q"]:
            cfg = ATS_CONFIG.get(p["ats"], ATS_CONFIG[5])
            breach_flag = "<span style='color:#dc2626; font-weight:700;'> (BREACH)</span>" if p["breached"] else ""
            tri_items.append(
                f"<div style='background:{cfg['bg']}; border:1px solid {cfg['border']}; border-radius:4px; padding:3px 8px; font-size:11px; display:inline-block; margin:2px; box-sizing:border-box;'>"
                f"<span style='font-weight:600; color:{cfg['color']};'>P#{p['pid']} • {cfg['name']}</span>"
                f"<span style='color:#475569;'> • {int(p['wait'])}m{breach_flag}</span></div>"
            )
        if tri_items:
            tri_q_html = f"<div style='max-height:100px; overflow-y:auto; margin-bottom:12px;'>{''.join(tri_items)}</div>"
        else:
            tri_q_html = "<div style='font-size:11px; color:#94a3b8; font-style:italic; margin-bottom:12px;'>No intake queue at triage.</div>"

        # Acute Bed Wait Queue
        bed_q_items = []
        for p in h["bed_q"][:25]:
            cfg = ATS_CONFIG.get(p["ats"], ATS_CONFIG[5])
            breach_flag = "<span style='color:#dc2626; font-weight:700;'> !</span>" if p["breached"] else ""
            bed_q_items.append(
                f"<div style='background:{cfg['bg']}; border:1px solid {cfg['border']}; border-radius:4px; padding:3px 8px; font-size:11px; display:inline-block; margin:2px; box-sizing:border-box;'>"
                f"<span style='font-weight:600; color:{cfg['color']};'>P#{p['pid']} • {cfg['name']}</span>"
                f"<span style='color:#475569;'> • {int(p['wait'])}m{breach_flag}</span></div>"
            )
        if len(h["bed_q"]) > 25:
            bed_q_items.append(f"<div style='font-size:10px; color:#64748b; display:inline-block; margin:2px;'>+{len(h['bed_q']) - 25} more...</div>")

        if bed_q_items:
            bed_q_html = f"<div style='max-height:160px; overflow-y:auto;'>{''.join(bed_q_items)}</div>"
        else:
            bed_q_html = "<div style='font-size:11px; color:#94a3b8; font-style:italic;'>All triaged patients placed in acute beds.</div>"

        total_waiting = len(h["tri_q"]) + len(h["bed_q"])
        zone_a_html = (
            f"<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.03); box-sizing:border-box;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f1f5f9; padding-bottom:8px; margin-bottom:12px;'>"
            f"<span style='font-weight:700; color:#1e293b; font-size:13px; text-transform:uppercase; letter-spacing:0.5px;'>Waiting Room & Triage Area</span>"
            f"<span style='font-size:11px; background:#f8fafc; border:1px solid #e2e8f0; padding:2px 8px; border-radius:4px; font-weight:600; color:#475569;'>Queue: <b>{total_waiting}</b></span>"
            f"</div>"
            f"<div style='font-size:11px; font-weight:600; color:#475569; margin-bottom:6px;'>Triage Intake Stations:</div>"
            f"{nurse_html}"
            f"<div style='font-size:11px; font-weight:600; color:#475569; margin-bottom:6px;'>Awaiting Triage Assessment ({len(h['tri_q'])}):</div>"
            f"{tri_q_html}"
            f"<div style='font-size:11px; font-weight:600; color:#475569; margin-bottom:6px;'>Triaged & Awaiting Acute Bed Placement ({len(h['bed_q'])}):</div>"
            f"{bed_q_html}"
            f"</div>"
        )
        st.html(zone_a_html)

    with col_beds:
        bed_cards = []
        for b in h["beds"]:
            if b["status"] == "empty":
                bg = "#f8fafc"
                border = "#e2e8f0"
                badge_bg = "#f1f5f9"
                badge_color = "#475569"
                badge_text = "AVAILABLE"
            elif b["status"] == "consult":
                bg = "#eff6ff"
                border = "#bfdbfe"
                badge_bg = "#dbeafe"
                badge_color = "#1d4ed8"
                badge_text = "IN CONSULT"
            elif b["status"] == "boarding":
                bg = "#fef2f2"
                border = "#fecaca"
                badge_bg = "#fee2e2"
                badge_color = "#dc2626"
                badge_text = "BOARDING BLOCK"
            elif b["status"] == "turnaround":
                bg = "#faf5ff"
                border = "#e9d5ff"
                badge_bg = "#f3e8ff"
                badge_color = "#7e22ce"
                badge_text = "DISCHARGE"
            else:
                bg = "#fffbeb"
                border = "#fde68a"
                badge_bg = "#fef3c7"
                badge_color = "#d97706"
                badge_text = "AWAITING MD"

            patient_badge = ""
            if b["pid"] is not None:
                cfg = ATS_CONFIG.get(b["ats"], ATS_CONFIG[5])
                patient_badge = f"<span style='background:{cfg['bg']}; color:{cfg['color']}; border:1px solid {cfg['border']}; padding:1px 5px; border-radius:3px; font-size:10px; font-weight:700;'>P#{b['pid']} {cfg['name']}</span>"

            bed_cards.append(
                f"<div style='background:{bg}; border:1px solid {border}; border-radius:6px; padding:8px 10px; height:82px; box-sizing:border-box; display:flex; flex-direction:column; justify-content:space-between; overflow:hidden;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<span style='font-size:11px; font-weight:700; color:#334155;'>BAY {b['bay']:02d}</span>"
                f"{patient_badge}</div>"
                f"<div>"
                f"<div style='display:inline-block; font-size:9px; font-weight:700; color:{badge_color}; background:{badge_bg}; padding:1px 5px; border-radius:3px; margin-bottom:2px;'>{badge_text}</div>"
                f"<div style='font-size:10px; color:#64748b; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{b['sublabel']}</div>"
                f"</div></div>"
            )

        grid_html = f"<div style='display:grid; grid-template-columns:repeat(auto-fill, minmax(130px, 1fr)); gap:8px;'>{''.join(bed_cards)}</div>"
        zone_b_html = (
            f"<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.03); box-sizing:border-box;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f1f5f9; padding-bottom:8px; margin-bottom:12px;'>"
            f"<span style='font-weight:700; color:#1e293b; font-size:13px; text-transform:uppercase; letter-spacing:0.5px;'>Acute Care Bays ({m['total_beds']} Bed Unit)</span>"
            f"<span style='font-size:11px; background:#f8fafc; border:1px solid #e2e8f0; padding:2px 8px; border-radius:4px; font-weight:600; color:#475569;'>Occupancy: <b>{m['in_beds']}/{m['total_beds']} ({m['bed_pct']:.0f}%)</b></span>"
            f"</div>"
            f"{grid_html}"
            f"</div>"
        )
        st.html(zone_b_html)


def main() -> None:  # pragma: no cover - UI
    import pandas as pd
    import streamlit as st

    st.set_page_config(
        page_title="PulseOps • Emergency Department Digital Twin",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Professional Clean Light Theme CSS (Portfolio Inspired: Minimalist, Sharp, High-Contrast)
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        header[data-testid="stHeader"] {
            background-color: transparent;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1350px;
        }
        /* Metric card styling */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }
        div[data-testid="stMetricLabel"] p {
            color: #475569 !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetricValue"] div {
            color: #0f172a !important;
            font-size: 24px !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 12px !important;
            font-weight: 600 !important;
        }
        /* Buttons */
        button[kind="secondary"] {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            color: #1e293b;
            border-radius: 6px;
            font-weight: 600;
        }
        button[kind="secondary"]:hover {
            border-color: #94a3b8;
            background-color: #f1f5f9;
        }
        /* Clean tabs / expanders */
        .streamlit-expanderHeader {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            color: #1e293b !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    ss = st.session_state
    ss.setdefault("clock", 11.0)
    ss.setdefault("playing", False)
    ss.setdefault("active_protocols", [])

    if ss.pop("tick", False):
        ss.clock = 0.0 if ss.clock >= 24.0 else min(24.0, ss.clock + 0.5)

    # 1. Header (Portfolio inspired minimalist hero)
    header_html = (
        "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:18px 24px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.03); box-sizing:border-box;'>"
        "<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;'>"
        "<div>"
        "<div style='display:flex; align-items:center; gap:8px;'>"
        "<span style='font-size:20px;'>🏥</span>"
        "<span style='font-size:20px; font-weight:800; color:#0f172a; letter-spacing:-0.5px;'>PulseOps • Emergency Department Operations Center</span>"
        "</div>"
        "<div style='color:#64748b; font-size:12px; margin-top:2px;'>Real-Time Decision Support & Surge Resource Optimizer • IBM LinuxONE (s390x) In-Memory Engine</div>"
        "</div>"
        "<div style='display:flex; gap:6px; flex-wrap:wrap; align-items:center;'>"
        "<span style='background:#f1f5f9; border:1px solid #e2e8f0; color:#334155; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:600;'>LinuxONE s390x</span>"
        "<span style='background:#ecfdf5; border:1px solid #a7f3d0; color:#047857; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:600;'>Pure-Python heapq</span>"
        "<span style='background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:600;'>0ms Latency</span>"
        "<span style='background:#f8fafc; border:1px solid #cbd5e1; color:#475569; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:600;'>CRN Validated</span>"
        "</div>"
        "</div>"
        "</div>"
    )
    st.html(header_html)

    # 2. Controls & Scenario Selection Bar
    c1, c2, c3, c4, c5 = st.columns([3, 3, 3, 1, 1])
    key = c1.selectbox("Operational Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k].name)
    preset_key = c2.selectbox("Hospital Scale Preset", list(HOSPITAL_PRESET_OPTIONS), index=0)
    c3.slider("Simulation Clock (24-Hour Horizon)", 0.0, 24.0, step=0.5, key="clock")
    c4.button("Pause" if ss.playing else "Play", on_click=lambda: ss.update(playing=not ss.playing))
    c5.button("+30m Step", on_click=lambda: ss.update(clock=min(24.0, ss.clock + 0.5)))

    # Scenario Clinical Profile Banner
    sc_info = SCENARIO_PROFILES.get(key, SCENARIO_PROFILES["normal"])
    scenario_banner_html = (
        f"<div style='background:#ffffff; border-left:4px solid {sc_info['badge_color']}; border:1px solid {sc_info['border']}; border-left-width:4px; padding:10px 14px; border-radius:6px; margin-bottom:16px; font-size:12px; box-sizing:border-box;'>"
        f"<span style='background:{sc_info['badge_bg']}; color:{sc_info['badge_color']}; font-weight:700; padding:2px 6px; border-radius:3px; margin-right:8px;'>{sc_info['badge']}</span>"
        f"<span style='color:#334155;'><b>{sc_info['title']}:</b> {sc_info['desc']}</span>"
        f"</div>"
    )
    st.html(scenario_banner_html)

    @st.cache_data(show_spinner=False)
    def cached(k: str, m: float, p: str):
        return compute_view(k, m, preset_name=p)

    v = cached(key, ss.clock * 60.0, preset_key)
    m, rec = v["metrics"], v["rec"]
    h = v["hospital"]
    risk = v.get("risk", {})

    # 3. Clinical Telemetry KPIs (01. System Telemetry)
    st.markdown("<div style='font-size:12px; font-weight:700; color:#64748b; letter-spacing:0.5px; text-transform:uppercase; margin-bottom:8px;'>01. Real-Time Emergency Department Telemetry</div>", unsafe_allow_html=True)
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Acute Bed Occupancy", f"{m['bed_pct']:.0f}%", f"{m['in_beds']}/{m['total_beds']} Beds In Use")
    g2.metric("Mean Door-to-Doctor", f"{m['avg_wait']:.0f} min", "Triage to Physician")
    g3.metric("ATS 2 Breach Rate", f"{m['ats2_breach_pct']:.0f}%", "Target: ≤10m")
    g4.metric("Waiting Room Census", f"{len(h['tri_q']) + len(h['bed_q'])}", f"{len(h['bed_q'])} triaged")
    g5.metric("Inpatient Boarding Block", f"{m['boarding_count']}", "Holding Acute Beds")

    # 4. AI Early Warning • 4-Hour Lookahead Risk Engine (ML Surrogate)
    st.markdown("<div style='font-size:12px; font-weight:700; color:#64748b; letter-spacing:0.5px; text-transform:uppercase; margin:18px 0 8px 0;'>02. AI Early Warning • 4-Hour Lookahead Risk Engine (Surrogate ML)</div>", unsafe_allow_html=True)
    if risk.get("available"):
        alarm = risk.get("alarm", False)
        breaks_first = risk.get("breaks_first", "Unknown")
        urgent_late_risk = (risk.get("risk_urgent_late") or 0.0) * 100.0
        beds_full_risk = (risk.get("risk_beds_full") or 0.0) * 100.0
        lwbs_risk = (risk.get("risk_lwbs") or 0.0) * 100.0

        if alarm:
            banner_html = (
                f"<div style='background:#fef2f2; border:1px solid #fecaca; border-left:4px solid #dc2626; border-radius:8px; padding:12px 16px; margin-bottom:12px; box-sizing:border-box;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>"
                f"<div>"
                f"<span style='background:#fee2e2; color:#b91c1c; font-weight:800; font-size:11px; padding:2px 8px; border-radius:4px; text-transform:uppercase; margin-right:8px;'>⚠️ 4-HOUR SURGE ALARM</span>"
                f"<span style='color:#991b1b; font-size:13px; font-weight:700;'>Bottleneck will breach first at: <span style='text-decoration:underline;'>{breaks_first.upper()}</span></span>"
                f"</div>"
                f"<span style='font-size:11px; color:#b91c1c; font-weight:600;'>XGBoost / Logistic Surrogate Active</span>"
                f"</div>"
                f"<div style='color:#7f1d1d; font-size:11.5px; margin-top:6px;'>{risk.get('message', '')}</div>"
                f"</div>"
            )
        else:
            banner_html = (
                f"<div style='background:#f0fdf4; border:1px solid #bbf7d0; border-left:4px solid #16a34a; border-radius:8px; padding:12px 16px; margin-bottom:12px; box-sizing:border-box;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>"
                f"<div>"
                f"<span style='background:#dcfce7; color:#15803d; font-weight:800; font-size:11px; padding:2px 8px; border-radius:4px; text-transform:uppercase; margin-right:8px;'>✓ EARLY WARNING NOMINAL</span>"
                f"<span style='color:#166534; font-size:13px; font-weight:700;'>All operational failure modes are within safe capacity limits for next 4 hours</span>"
                f"</div>"
                f"<span style='font-size:11px; color:#15803d; font-weight:600;'>XGBoost / Logistic Surrogate Active</span>"
                f"</div>"
                f"<div style='color:#14532d; font-size:11.5px; margin-top:6px;'>{risk.get('message', '')}</div>"
                f"</div>"
            )
        st.html(banner_html)

        # 3 Visual Risk Gauges
        rg1, rg2, rg3 = st.columns(3)
        with rg1:
            color = "#dc2626" if urgent_late_risk > 25 else "#d97706" if urgent_late_risk > 15 else "#16a34a"
            tag = "HIGH RISK" if urgent_late_risk > 25 else "ELEVATED" if urgent_late_risk > 15 else "NORMAL"
            st.markdown(
                f"<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px; box-sizing:border-box;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>"
                f"<span style='font-size:11px; font-weight:700; color:#475569; text-transform:uppercase;'>Urgent Seen Late (ATS 1-3)</span>"
                f"<span style='font-size:10px; font-weight:700; color:{color};'>{tag}</span>"
                f"</div>"
                f"<div style='font-size:22px; font-weight:800; color:{color};'>{urgent_late_risk:.1f}%</div>"
                f"<div style='height:6px; background:#f1f5f9; border-radius:3px; margin:6px 0; overflow:hidden;'>"
                f"<div style='width:{min(100.0, urgent_late_risk)}%; height:100%; background:{color};'></div>"
                f"</div>"
                f"<div style='font-size:10px; color:#64748b;'>ATS 1–3 triage breach probability (Alarm line: 25%)</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with rg2:
            color = "#dc2626" if beds_full_risk > 31 else "#d97706" if beds_full_risk > 20 else "#16a34a"
            tag = "HIGH RISK" if beds_full_risk > 31 else "ELEVATED" if beds_full_risk > 20 else "NORMAL"
            st.markdown(
                f"<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px; box-sizing:border-box;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>"
                f"<span style='font-size:11px; font-weight:700; color:#475569; text-transform:uppercase;'>Acute Bed Saturation</span>"
                f"<span style='font-size:10px; font-weight:700; color:{color};'>{tag}</span>"
                f"</div>"
                f"<div style='font-size:22px; font-weight:800; color:{color};'>{beds_full_risk:.1f}%</div>"
                f"<div style='height:6px; background:#f1f5f9; border-radius:3px; margin:6px 0; overflow:hidden;'>"
                f"<div style='width:{min(100.0, beds_full_risk)}%; height:100%; background:{color};'></div>"
                f"</div>"
                f"<div style='font-size:10px; color:#64748b;'>Beds > 95% full probability (Alarm line: 31%)</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with rg3:
            color = "#dc2626" if lwbs_risk > 14 else "#d97706" if lwbs_risk > 8 else "#16a34a"
            tag = "HIGH RISK" if lwbs_risk > 14 else "ELEVATED" if lwbs_risk > 8 else "NORMAL"
            st.markdown(
                f"<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px; box-sizing:border-box;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>"
                f"<span style='font-size:11px; font-weight:700; color:#475569; text-transform:uppercase;'>Walkout / LWBS Queue</span>"
                f"<span style='font-size:10px; font-weight:700; color:{color};'>{tag}</span>"
                f"</div>"
                f"<div style='font-size:22px; font-weight:800; color:{color};'>{lwbs_risk:.1f}%</div>"
                f"<div style='height:6px; background:#f1f5f9; border-radius:3px; margin:6px 0; overflow:hidden;'>"
                f"<div style='width:{min(100.0, lwbs_risk)}%; height:100%; background:{color};'></div>"
                f"</div>"
                f"<div style='font-size:10px; color:#64748b;'>Patients walk out unseen probability (Alarm line: 14%)</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Early-warning surrogate model is currently running in baseline fallback mode.")

    # 4. Tactical Surge Intervention Deck (02. Operational What-If Controls)
    st.markdown("<div style='font-size:12px; font-weight:700; color:#64748b; letter-spacing:0.5px; text-transform:uppercase; margin:18px 0 8px 0;'>02. Tactical Surge Intervention Deck (What-If Controls)</div>", unsafe_allow_html=True)

    deck_box_html = (
        "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px; margin-bottom:10px; box-sizing:border-box;'>"
        "<div style='font-size:12px; color:#475569;'>Toggle tactical staffing and buffer policies below to evaluate real-time capacity and wait-time impacts.</div>"
        "</div>"
    )
    st.html(deck_box_html)

    # Use modern st.pills for clean SaaS toggle pills
    p_col1, p_col2 = st.columns([4, 1])
    selected_pills = p_col1.pills(
        "Active Protocols",
        PROTOCOLS,
        selection_mode="multi",
        default=ss.active_protocols,
        key="pills_protocols",
        label_visibility="collapsed",
    )
    ss.active_protocols = selected_pills or []

    def sync_ai_action():
        active = []
        if "ADD_NURSE" in rec.action:
            active.append("Surge Nurse (+1)")
        if "ADD_PHYSICIAN" in rec.action:
            active.append("On-Call Attending (+1)")
        if "FAST_TRACK" in rec.action:
            active.append("Fast-Track Protocol")
        if "INPATIENT_BUFFER_EXPEDITE" in rec.action:
            active.append("Expedite Ward Transfers")
        ss.active_protocols = active

    p_col2.button("Apply AI Policy", on_click=sync_ai_action, use_container_width=True)

    # Parse active action from pills
    action_parts = []
    if "Surge Nurse (+1)" in ss.active_protocols:
        action_parts.append("ADD_NURSE")
    if "On-Call Attending (+1)" in ss.active_protocols:
        action_parts.append("ADD_PHYSICIAN")
    if "Fast-Track Protocol" in ss.active_protocols:
        action_parts.append("FAST_TRACK")
    if "Expedite Ward Transfers" in ss.active_protocols:
        action_parts.append("INPATIENT_BUFFER_EXPEDITE")
    active_action_str = "+".join(action_parts) or "DO_NOTHING"
    current_action = Action.parse(active_action_str)

    if current_action.name != "DO_NOTHING":
        sim_res = simulate(v["snapshot"], action=current_action, hours=4, scenario=v["scenario"])
        base_wait = rec.base["median_wait"]
        delta_wait = base_wait - sim_res.median_wait
        delta_breaches = rec.base["breaches"] - sim_res.total_breaches

        active_bar_html = (
            f"<div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:8px 14px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center; box-sizing:border-box;'>"
            f"<div><span style='font-weight:700; color:#15803d; font-size:12px;'>ACTIVE PROTOCOL: {pretty(current_action.name)}</span>"
            f"<span style='color:#475569; font-size:11px; margin-left:12px;'>Deployment Cost: <b>${current_action.cost:,.0f}</b></span></div>"
            f"<div style='display:flex; gap:16px; font-size:12px; font-weight:700;'>"
            f"<span style='color:#15803d;'>Projected Wait Saved: -{max(0.0, delta_wait):.0f} min</span>"
            f"<span style='color:#166534;'>Breaches Prevented: {max(0, int(delta_breaches))}</span>"
            f"</div></div>"
        )
        st.html(active_bar_html)

    # 5. Visual Hospital Floor Plan (03. Digital Twin)
    st.markdown("<div style='font-size:12px; font-weight:700; color:#64748b; letter-spacing:0.5px; text-transform:uppercase; margin:18px 0 8px 0;'>03. Physical Hospital Floor Plan (Emergency Department Unit)</div>", unsafe_allow_html=True)
    render_hospital_twin(h, m)

    # 6. Prescriptive Intelligence & Counterfactual Analysis (04. AI Prescriptive Optimization)
    st.markdown("<div style='font-size:12px; font-weight:700; color:#64748b; letter-spacing:0.5px; text-transform:uppercase; margin:18px 0 8px 0;'>04. Prescriptive AI Decision Support (Counterfactual What-If)</div>", unsafe_allow_html=True)

    rec_box_bg = "#f0fdf4" if rec.action == "DO_NOTHING" else "#fffbeb"
    rec_box_border = "#bbf7d0" if rec.action == "DO_NOTHING" else "#fde68a"
    rec_box_color = "#15803d" if rec.action == "DO_NOTHING" else "#b45309"

    rec_html = (
        f"<div style='background:{rec_box_bg}; border:1px solid {rec_box_border}; border-radius:6px; padding:10px 14px; margin-bottom:12px; font-size:12px; box-sizing:border-box;'>"
        f"<span style='font-weight:700; color:{rec_box_color}; margin-right:8px;'>AI PRESCRIBED ACTION:</span>"
        f"<span style='color:#1e293b; font-weight:600;'>{rec.headline().replace('RECOMMENDED: ', '')}</span>"
        f"</div>"
    )
    st.html(rec_html)

    _, lo, hi = rec.breach_reduction_ci
    st.caption(f"Estimated breach reduction 95% Confidence Interval: {lo:.0f}% to {hi:.0f}% | Pareto Frontier: {', '.join(rec.pareto)}")

    # Counterfactual 3-Column Review
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        st.markdown("<div style='font-size:12px; font-weight:700; color:#475569; margin-bottom:6px;'>STATUS QUO (DO NOTHING)</div>", unsafe_allow_html=True)
        st.metric("Projected Breaches (4h)", f"{rec.base['breaches']:.1f}")
        st.metric("Median Wait Time", f"{rec.base['median_wait']:.0f} min")
        st.metric("Left Without Being Seen", f"{rec.base['lwbs']:.1f}")
    with cf2:
        st.markdown(f"<div style='font-size:12px; font-weight:700; color:#1d4ed8; margin-bottom:6px;'>AI OPTIMAL ({rec.action.replace('_', ' ').title()})</div>", unsafe_allow_html=True)
        st.metric("Projected Breaches (4h)", f"{rec.best['breaches']:.1f}", f"-{rec.breach_reduction_pct:.0f}%")
        st.metric("Median Wait Time", f"{rec.best['median_wait']:.0f} min", f"-{rec.wait_reduction:.0f} min")
        st.metric("Left Without Being Seen", f"{rec.best['lwbs']:.1f}", f"-{rec.base['lwbs'] - rec.best['lwbs']:.1f}")
    with cf3:
        st.markdown(f"<div style='font-size:12px; font-weight:700; color:#0f172a; margin-bottom:6px;'>ACTIVE LEVERS ({current_action.name.replace('_', ' ').title()})</div>", unsafe_allow_html=True)
        if current_action.name == "DO_NOTHING":
            st.metric("Projected Breaches (4h)", f"{rec.base['breaches']:.1f}", "Baseline")
            st.metric("Median Wait Time", f"{rec.base['median_wait']:.0f} min", "$0 cost")
            st.metric("Left Without Being Seen", f"{rec.base['lwbs']:.1f}", "Baseline")
        else:
            st.metric("Projected Breaches (4h)", f"{sim_res.total_breaches:.1f}", f"-{max(0.0, delta_breaches):.0f}")
            st.metric("Median Wait Time", f"{sim_res.median_wait:.0f} min", f"-{max(0.0, delta_wait):.0f} min")
            st.metric("Left Without Being Seen", f"{sim_res.lwbs:.1f}", f"${current_action.cost:,.0f} cost")

    # 7. 24-Hour Operations Trend (05. Flow Trajectory)
    st.markdown("<div style='font-size:12px; font-weight:700; color:#64748b; letter-spacing:0.5px; text-transform:uppercase; margin:18px 0 8px 0;'>05. 24-Hour Patient Flow & Capacity Trajectory</div>", unsafe_allow_html=True)
    st.line_chart(pd.DataFrame(v["series"]).set_index("hour"))

    # 8. Clinical Handover & Audit Log Expanders (06. Handover Memo)
    with st.expander("06. Charge Nurse Shift Handover Briefing (Copilot)", expanded=True):
        st.markdown(
            f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:12px 16px; font-size:12px; line-height:1.6; color:#334155; font-family:monospace; white-space:pre-wrap;'>{v['briefing']}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("07. Comprehensive Policy Evaluation (Pareto Frontier Table)"):
        st.dataframe(pd.DataFrame(rec.table), use_container_width=True)

    with st.expander("08. Model Cards, Calibration Evidence & LinuxONE Verification", expanded=False):
        import json
        from pathlib import Path

        root_dir = Path(__file__).resolve().parents[2]
        tab_ew, tab_pf, tab_lx, tab_params = st.tabs([
            "Model 2: 4-Hour Early Warning Surrogate",
            "Model 1: CDC NHAMCS Real Data Calibration",
            "System Performance & LinuxONE Benchmarks",
            "Parameter Provenance Audit (81 Parameters)"
        ])

        with tab_ew:
            st.markdown(
                "**Early Warning AI Surrogate (XGBoost / Logistic Models)**\n"
                "- **Dataset:** 750 simulated operational days × 4 scenario types = 120,000 temporal state snapshots.\n"
                "- **Horizon:** 4-hour forward lookahead across 3 critical failure targets: `urgent_late` (ATS 1–3), `beds_full` (>95%), and `lwbs` (walkout).\n"
                "- **False Alarm Control:** Fixed at ≤ 5% false alarm rate on baseline days."
            )
            stage_csv = root_dir / "models" / "stage_results.csv"
            if stage_csv.exists():
                sdf = pd.read_csv(stage_csv)
                cols = [c for c in ("stage", "train_on", "test_on", "target", "auc_roc", "auc_roc_lo", "auc_roc_hi", "recall_at_5pct_fpr", "rule_recall_at_5pct_fpr", "median_lead_min") if c in sdf.columns]
                st.markdown("**Stage-by-Stage Progressive Generalization (AUC ROC & Early-Warning Lead Time):**")
                st.dataframe(sdf[cols].round(3), hide_index=True, use_container_width=True)
            stage_png = root_dir / "results" / "stage_chart.png"
            if stage_png.exists():
                st.image(str(stage_png), caption="Progressive Learning Stages: Stage 1 (Normal Only) vs Stage 2 (Unseen Surge Generalization) vs Stage 3 (Retrained with Surge Data)")

        with tab_pf:
            st.markdown(
                "**Model 1: CDC NHAMCS 2022 Ground-Truth Survey Calibration**\n"
                "- **Empirical Source:** CDC National Hospital Ambulatory Medical Care Survey (NHAMCS 2022 Public Use File).\n"
                "- **Cohort:** 10,125 triaged visits across 147 hospitals; held-out hospitals for out-of-sample testing.\n"
                "- **Admission Model (HGB):** Test ROC AUC **0.845** (95% CI 0.790–0.878) vs 0.732 baseline triage-only model."
            )
            cal_png = root_dir / "results" / "for_slides" / "calibration_real_vs_simulated.png"
            if cal_png.exists():
                st.image(str(cal_png), caption="Calibration Curve: Real NHAMCS 2022 Patient Flow Distribution vs Simulated Monday")
            cal_csv = root_dir / "results" / "calibration.csv"
            if cal_csv.exists():
                st.markdown("**Empirical vs Simulated Normal Day Benchmark:**")
                st.dataframe(pd.read_csv(cal_csv), hide_index=True, use_container_width=True)

        with tab_lx:
            st.markdown(
                "**In-Memory Microsecond Execution & IBM LinuxONE (s390x) Validation**\n"
                "- **Engine:** Pure-Python in-memory discrete-event simulation with CRN (Common Random Numbers).\n"
                "- **Throughput:** Average **6.62ms** to **14.93ms** per full 24-hour hospital day.\n"
                "- **Architecture:** s390x big-endian mainframe enterprise server & x86 multi-worker multiprocessing support."
            )
            timing_png = root_dir / "results" / "for_slides" / "timing_workers.png"
            if timing_png.exists():
                st.image(str(timing_png), caption="Multiprocessing Scalability Across Concurrent Workers (Days/Second Throughput)")
            local_json = root_dir / "results" / "local_run.json"
            if local_json.exists():
                try:
                    lj = json.loads(local_json.read_text(encoding="utf-8"))
                    pt = lj.get("parallel_timing")
                    if pt:
                        st.markdown("**Parallel Worker Benchmark Run:**")
                        st.dataframe(pd.DataFrame(pt), hide_index=True, use_container_width=True)
                except Exception:
                    pass

        with tab_params:
            st.markdown(
                "**Clinical Parameter Transparency & Evidence Audit (81 Parameters)**\n"
                "- **[DATA]:** Measured directly from CDC NHAMCS survey dataset.\n"
                "- **[EVIDENCE]:** Sourced from peer-reviewed clinical studies (Nature Health 2025, Texas DSHS, Disaster Medicine).\n"
                "- **[ASSUMPTION]:** Explicit operational boundaries and baseline staffing models."
            )
            all_params = [
                {"Parameter": p["name"], "Value": json.dumps(p["value"], default=str)[:120], "Tag": p.get("tag", "ASSUMPTION"), "Status": p.get("status", "ok"), "Clinical Note / Source": p.get("note", "")}
                for p in PARAMETER_TABLE + HOSPITAL_PARAMETERS + OPTIMISER_PARAMETERS
            ]
            ptdf = pd.DataFrame(all_params)
            tag_filter = st.multiselect("Filter Provenance Tags", ["DATA", "EVIDENCE", "ASSUMPTION"], default=["DATA", "EVIDENCE", "ASSUMPTION"], key="filter_param_tags")
            st.dataframe(ptdf[ptdf["Tag"].isin(tag_filter)], hide_index=True, use_container_width=True)

    if ss.playing:
        time.sleep(1.0)
        ss.tick = True
        st.rerun()


if __name__ == "__main__":
    main()
