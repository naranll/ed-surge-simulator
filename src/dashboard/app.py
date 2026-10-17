"""Emergency Department Real-Time Decision Support System.
Clean clinical operations dashboard for hospital capacity and surge management.
Run: streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explainer.briefing import build_briefing  # noqa: E402
from src.optimization.optimizer import recommend, pretty  # noqa: E402
from src.simulation.ed_simulator import Action, LiveED, simulate  # noqa: E402
from src.simulation.scenarios import ATS_TARGET, SCENARIOS  # noqa: E402

# Clinical Triage Standards (Australasian Triage Scale / Emergency Severity Index)
ATS_CONFIG = {
    1: {"name": "ATS 1", "desc": "Resuscitation", "color": "#b91c1c", "bg": "#fef2f2", "border": "#fecaca", "target": 0},
    2: {"name": "ATS 2", "desc": "Emergency", "color": "#c2410c", "bg": "#fff7ed", "border": "#fed7aa", "target": 10},
    3: {"name": "ATS 3", "desc": "Urgent", "color": "#a16207", "bg": "#fefce8", "border": "#fef08a", "target": 30},
    4: {"name": "ATS 4", "desc": "Semi-Urgent", "color": "#15803d", "bg": "#f0fdf4", "border": "#bbf7d0", "target": 60},
    5: {"name": "ATS 5", "desc": "Non-Urgent", "color": "#0369a1", "bg": "#f0f9ff", "border": "#bae6fd", "target": 120},
}

SCENARIO_PROFILES = {
    "normal": {
        "title": "Baseline Monday Operations",
        "badge": "STANDARD OPERATIONS",
        "badge_bg": "#f1f5f9",
        "badge_color": "#475569",
        "border": "#cbd5e1",
        "desc": "Typical diurnal arrival pattern (~200 visits/24h) with standard scheduled staffing. Baseline comparison point.",
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

    # Triage Queue
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

    # Acute Treatment Bays (20 Beds)
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
                    label = "INPATIENT BOARDING"
                    sublabel = f"Admitted • Holding Bed ({int(hold_time)}m)"
                else:
                    status = "turnaround"
                    label = "DISCHARGE BUFFER"
                    sublabel = "Awaiting Labs / Transport"
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


def compute_view(key: str, minute: float, seed: int = 42, workers: int = 1) -> dict:
    sc = SCENARIOS[key]
    live = LiveED(sc, seed)
    live.advance_to(minute)
    rec = recommend(live.snapshot(), sc, workers=workers)
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
            color = "#065f46" if n["busy"] else "#475569"
            status_txt = "Active Intake" if n["busy"] else "Intake Ready"
            pulse = "●" if n["busy"] else "○"
            nurse_items.append(
                f"<div style='background:{bg}; border:1px solid {border}; border-radius:6px; padding:6px 10px; font-size:11px; flex:1; min-width:130px; box-sizing:border-box;'>"
                f"<div style='font-weight:600; color:{color};'><span style='color:{color}; margin-right:4px;'>{pulse}</span>{n['name']}</div>"
                f"<div style='color:#64748b; font-size:10px;'>{status_txt}</div></div>"
            )
        nurse_html = f"<div style='display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap;'>{''.join(nurse_items)}</div>"

        # Triage Queue Chips
        tri_items = []
        for p in h["tri_q"]:
            cfg = ATS_CONFIG.get(p["ats"], ATS_CONFIG[5])
            breach_flag = "<span style='color:#b91c1c; font-weight:700;'> (BREACH)</span>" if p["breached"] else ""
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
            breach_flag = "<span style='color:#b91c1c; font-weight:700;'> !</span>" if p["breached"] else ""
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
                badge_color = "#b91c1c"
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
                badge_color = "#b45309"
                badge_text = "AWAITING MD"

            patient_badge = ""
            if b["pid"] is not None:
                cfg = ATS_CONFIG.get(b["ats"], ATS_CONFIG[5])
                patient_badge = f"<span style='background:{cfg['bg']}; color:{cfg['color']}; border:1px solid {cfg['border']}; padding:1px 4px; border-radius:3px; font-size:10px; font-weight:700;'>P#{b['pid']} {cfg['name']}</span>"

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

        grid_html = f"<div style='display:grid; grid-template-columns:repeat(4, 1fr); gap:8px;'>{''.join(bed_cards)}</div>"
        zone_b_html = (
            f"<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.03); box-sizing:border-box;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f1f5f9; padding-bottom:8px; margin-bottom:12px;'>"
            f"<span style='font-weight:700; color:#1e293b; font-size:13px; text-transform:uppercase; letter-spacing:0.5px;'>Acute Care Bays (20 Bed Unit)</span>"
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
        page_title="Emergency Department Decision Support System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Professional Clean Light Theme CSS
    st.markdown(
        """
        <style>
        /* Base typography and clean clinical theme */
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
        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        div[data-testid="stMetricLabel"] p {
            color: #64748b !important;
            font-size: 11px !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        div[data-testid="stMetricValue"] div {
            color: #0f172a !important;
            font-size: 22px !important;
            font-weight: 700 !important;
        }
        /* Buttons and inputs */
        button[kind="secondary"] {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            color: #334155;
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
    ss.setdefault("add_nurse", False)
    ss.setdefault("add_doc", False)
    ss.setdefault("fast_track", False)
    ss.setdefault("expedite", False)

    if ss.pop("tick", False):
        ss.clock = 0.0 if ss.clock >= 24.0 else min(24.0, ss.clock + 0.5)

    # 1. Clinical Header
    header_html = (
        "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px 20px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.03); box-sizing:border-box;'>"
        "<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;'>"
        "<div>"
        "<div style='font-size:18px; font-weight:800; color:#0f172a; letter-spacing:-0.3px;'>Metropolitan Emergency Department Operations Center</div>"
        "<div style='color:#64748b; font-size:12px; margin-top:2px;'>Real-Time Decision Support & Surge Resource Optimizer • IBM LinuxONE (s390x) In-Memory Engine</div>"
        "</div>"
        "<div style='display:flex; gap:8px; flex-wrap:wrap; align-items:center;'>"
        "<span style='background:#f1f5f9; border:1px solid #e2e8f0; color:#475569; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:600;'>LinuxONE Native Engine</span>"
        "<span style='background:#f0fdf4; border:1px solid #bbf7d0; color:#15803d; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:600;'>System Online</span>"
        "<span style='background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:600;'>CRN Validated</span>"
        "</div>"
        "</div>"
        "</div>"
    )
    st.html(header_html)

    # 2. Controls & Scenario Selection Bar
    c1, c2, c3, c4 = st.columns([4, 4, 1, 1])
    key = c1.selectbox("Operational Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k].name)
    c2.slider("Simulation Clock (24-Hour Horizon)", 0.0, 24.0, step=0.5, key="clock")
    c3.button("Pause" if ss.playing else "Play", on_click=lambda: ss.update(playing=not ss.playing))
    c4.button("+30m Step", on_click=lambda: ss.update(clock=min(24.0, ss.clock + 0.5)))

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
    def cached(k: str, m: float):
        return compute_view(k, m)

    v = cached(key, ss.clock * 60.0)
    m, rec = v["metrics"], v["rec"]
    h = v["hospital"]

    # 3. Clinical Telemetry KPIs
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Acute Bed Occupancy", f"{m['bed_pct']:.0f}%", f"{m['in_beds']}/{m['total_beds']} Beds")
    g2.metric("Mean Door-to-Doctor", f"{m['avg_wait']:.0f} min", "Triage to MD")
    g3.metric("ATS 2 Breach Rate", f"{m['ats2_breach_pct']:.0f}%", "Target: ≤10m")
    g4.metric("Waiting Room Census", f"{len(h['tri_q']) + len(h['bed_q'])}", f"{len(h['bed_q'])} triaged")
    g5.metric("Inpatient Boarding Block", f"{m['boarding_count']}", "Holding Acute Beds")

    # 4. Tactical Intervention Command Deck
    deck_header_html = (
        "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px; margin:16px 0 10px 0; box-shadow:0 1px 2px rgba(0,0,0,0.02); box-sizing:border-box;'>"
        "<div style='display:flex; justify-content:space-between; align-items:center;'>"
        "<div>"
        "<span style='font-weight:700; color:#1e293b; font-size:13px; text-transform:uppercase; letter-spacing:0.5px;'>Operational Intervention Deck</span>"
        "<span style='color:#64748b; font-size:12px; margin-left:10px;'>Simulate staffing surges and discharge buffer protocols</span>"
        "</div>"
        "</div>"
        "</div>"
    )
    st.html(deck_header_html)

    w1, w2, w3, w4, w5 = st.columns(5)
    t_nurse = w1.checkbox("Surge Nurse (+1)", value=ss.add_nurse, key="chk_nurse")
    t_doc = w2.checkbox("On-Call Attending (+1)", value=ss.add_doc, key="chk_doc")
    t_fast = w3.checkbox("Fast-Track Protocol", value=ss.fast_track, key="chk_fast")
    t_exp = w4.checkbox("Expedite Ward Transfers", value=ss.expedite, key="chk_exp")

    ss.add_nurse, ss.add_doc, ss.fast_track, ss.expedite = t_nurse, t_doc, t_fast, t_exp

    def apply_ai_rec():
        ss.add_nurse = "ADD_NURSE" in rec.action
        ss.add_doc = "ADD_PHYSICIAN" in rec.action
        ss.fast_track = "FAST_TRACK" in rec.action
        ss.expedite = "INPATIENT_BUFFER_EXPEDITE" in rec.action

    w5.button("Apply AI Recommendation", on_click=apply_ai_rec, use_container_width=True)

    # Active Intervention Evaluation
    action_parts = []
    if ss.add_nurse:
        action_parts.append("ADD_NURSE")
    if ss.add_doc:
        action_parts.append("ADD_PHYSICIAN")
    if ss.fast_track:
        action_parts.append("FAST_TRACK")
    if ss.expedite:
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

    # 5. Visual Hospital Floor Plan
    render_hospital_twin(h, m)

    # 6. Prescriptive Intelligence & Counterfactual Analysis
    st.markdown("#### Prescriptive AI Decision Support")
    
    rec_box_bg = "#f0fdf4" if rec.action == "DO_NOTHING" else "#fffbeb"
    rec_box_border = "#bbf7d0" if rec.action == "DO_NOTHING" else "#fde68a"
    rec_box_color = "#15803d" if rec.action == "DO_NOTHING" else "#b45309"
    
    rec_html = (
        f"<div style='background:{rec_box_bg}; border:1px solid {rec_box_border}; border-radius:6px; padding:10px 14px; margin-bottom:12px; font-size:12px; box-sizing:border-box;'>"
        f"<span style='font-weight:700; color:{rec_box_color}; margin-right:8px;'>AI RECOMMENDATION:</span>"
        f"<span style='color:#1e293b; font-weight:600;'>{rec.headline()}</span>"
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

    # 7. 24-Hour Operations Trend
    st.markdown("#### 24-Hour Patient Flow & Capacity Trajectory")
    st.line_chart(pd.DataFrame(v["series"]).set_index("hour"))

    # 8. Clinical Handover & Audit Log Expanders
    with st.expander("Charge Nurse Shift Handover Briefing (Copilot)", expanded=True):
        st.markdown(
            f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:12px 16px; font-size:12px; line-height:1.6; color:#334155; font-family:monospace; white-space:pre-wrap;'>{v['briefing']}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Comprehensive Policy Evaluation (Pareto Frontier)"):
        st.dataframe(pd.DataFrame(rec.table), use_container_width=True)

    if ss.playing:
        time.sleep(1.0)
        ss.tick = True
        st.rerun()


if __name__ == "__main__":
    main()
