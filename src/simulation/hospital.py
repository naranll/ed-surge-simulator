"""HospitalConfig + a day runner. Standard library only. Additive: defaults equal today's behaviour.

How the planner-friendly inputs map onto the simulator (also shown in the dashboard):
  beds, rooms          -> beds the simulator can use = the SMALLER of the two (a bed needs a room)
  triage stations/nurses -> triage capacity = the SMALLER of the two (a nurse needs a station)
  doctors              -> physicians seeing patients at the same time
  fast-track bays      -> 0 = off; 1 or more = fast track ON all day (levels 4-5 need 40% less doctor time).
                          More than one bay does NOT change the effect (the simulator has one fast-track switch)
  boarding cut         -> admitted patients leave their bed 0 / 25 / 50% sooner, all day (faster ward transfers)
  extra staff shift    -> one extra doctor for 6 hours from the chosen start time
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

from src.optimization.optimizer import DAY_MIN, Plan, boarding_cut_shim, compute_metrics
from src.simulation import patient_flow
from src.simulation.ed_simulator import ATS_TARGET, Action, EDSim
from src.simulation.scenarios import CAPACITY

URGENT = (1, 2, 3)


@dataclass(frozen=True)
class HospitalConfig:
    beds: int = CAPACITY["beds"]
    rooms: Optional[int] = None
    triage_stations: Optional[int] = None
    triage_nurses: int = CAPACITY["nurses"]
    doctors: int = CAPACITY["docs"]
    fast_track_bays: int = 0
    boarding_cut: float = 0.0                 # 0, 0.25 or 0.5
    extra_staff_start: Optional[float] = None  # minute of the day; None = no extra shift
    extra_staff_hours: float = 6.0
    patient_flow: bool = False                # True = real-data admission and care times (needs the lookup file)
    volume_scale: float = 1.0                 # 1.0 = the ~200 patients/day reference; below 1 keeps that share of patients

    def __post_init__(self):
        if min(self.beds, self.triage_nurses, self.doctors) < 1:
            raise ValueError("beds, triage nurses and doctors must each be at least 1")
        if not 0.0 < self.volume_scale <= 1.0:
            raise ValueError("volume_scale must be above 0 and at most 1")
        if not 0.0 <= self.boarding_cut < 1.0:
            raise ValueError("boarding_cut must be from 0 to under 1")

    @property
    def effective_beds(self) -> int:
        return min(self.beds, self.rooms) if self.rooms else self.beds

    @property
    def triage_capacity(self) -> int:
        return min(self.triage_stations, self.triage_nurses) if self.triage_stations else self.triage_nurses

    def what_changes(self) -> dict:
        return {
            "beds_used": f"{self.effective_beds} (the smaller of beds {self.beds} and rooms {self.rooms or 'not limited'})",
            "triage_capacity": f"{self.triage_capacity} (the smaller of nurses {self.triage_nurses} and stations {self.triage_stations or 'not limited'})",
            "doctors": self.doctors,
            "fast_track": "ON all day (levels 4-5 need 40% less doctor time)" if self.fast_track_bays > 0 else "off",
            "boarding": f"admitted patients leave {self.boarding_cut * 100:.0f}% sooner" if self.boarding_cut else "normal ward transfers",
            "extra_shift": (f"+1 doctor for {self.extra_staff_hours:g} h from {int(self.extra_staff_start // 60):02d}:{int(self.extra_staff_start % 60):02d}"
                            if self.extra_staff_start is not None else "none"),
        }


# Reference hospitals. Capacities are ASSUMPTIONS (NHAMCS has no bed counts). "Large" was set by calibration
# (results/calibration.csv); Small and Medium are the same design scaled by patient volume (30% / 60%) and are NOT separately calibrated.
PRESETS = {
    "Small": HospitalConfig(beds=18, triage_nurses=1, doctors=2, patient_flow=True, volume_scale=0.3),     # ~60 patients/day
    "Medium": HospitalConfig(beds=36, triage_nurses=2, doctors=3, patient_flow=True, volume_scale=0.6),    # ~120 patients/day
    "Large": HospitalConfig(beds=60, triage_nurses=3, doctors=4, patient_flow=True, volume_scale=1.0),     # ~200/day: the calibrated one
}


def prepare(cfg: HospitalConfig, patients: Sequence) -> list:
    """Apply the real-data patient-flow tables when the config asks for them (else leave patients as they are)."""
    pats = list(patients)
    if cfg.volume_scale < 1.0:   # keep each patient with probability volume_scale (keyed by the patient, so it is repeatable)
        pats = [a for a in pats if random.Random(f"thin:{a.pid}:{a.t:.4f}").random() < cfg.volume_scale]
    if cfg.patient_flow:
        pats = patient_flow.apply_patient_flow(pats)
    return pats


def build_sim(cfg: HospitalConfig, patients: Sequence) -> EDSim:
    sim = EDSim(nurses=cfg.triage_capacity, docs=cfg.doctors, beds=cfg.effective_beds, per_patient_stay=cfg.patient_flow)
    sim.add_arrivals(patients)
    if cfg.fast_track_bays > 0:
        sim.ft_until = 1e12
    return sim


def _events(cfg: HospitalConfig, plan: Optional[Plan]) -> list:
    ev = []   # (minute, Action, boarding_cut)
    if cfg.boarding_cut > 0:
        ev.append((0.0, Action(expedite=True, duration=DAY_MIN), cfg.boarding_cut))
    if cfg.extra_staff_start is not None:
        ev.append((float(cfg.extra_staff_start), Action(doc=1, duration=cfg.extra_staff_hours * 60.0), 0.0))
    if plan is not None and not plan.is_nothing:
        ev.append((plan.start, plan.action(), plan.boarding_cut))
    return sorted(ev, key=lambda e: e[0])


def run_day(cfg: HospitalConfig, patients: Sequence, plan: Optional[Plan] = None,
            prepared: bool = False, stop_every: Optional[float] = None, on_stop=None) -> EDSim:
    """One whole day (00:00-24:00). ``on_stop(sim)`` is called every ``stop_every`` minutes (read only)."""
    pats = list(patients) if prepared else prepare(cfg, patients)
    sim = build_sim(cfg, pats)
    cut = max([e[2] for e in _events(cfg, plan)] or [0.0])
    with boarding_cut_shim(cut):
        marks = sorted({e[0] for e in _events(cfg, plan)} | (
            {k * stop_every for k in range(0, int(DAY_MIN // stop_every) + 1)} if stop_every else set()) | {DAY_MIN})
        pending = _events(cfg, plan)
        for m in marks:
            sim.run_until(m)
            while pending and pending[0][0] <= m + 1e-9:
                sim.apply(pending.pop(0)[1])
            if on_stop is not None and stop_every and abs(m / stop_every - round(m / stop_every)) < 1e-9 and m < DAY_MIN + 1e-9:
                on_stop(sim)
    return sim


def runner_for(cfg: HospitalConfig):
    """A ``runner(plan, patients)`` for the optimiser's plan_search / select_plan (so plans are scored on THIS hospital)."""
    def run(plan: Plan, patients: Sequence):
        sim = run_day(cfg, patients, plan)
        return compute_metrics(sim, 0.0, DAY_MIN)
    return run


# ------------------------------------------------------------------ measuring
def waits_by_level(sim: EDSim, t_end: float = DAY_MIN) -> dict:
    """{level: [wait minutes]} for every patient who has arrived (still-waiting patients count up to t_end)."""
    out = {l: [] for l in range(1, 6)}
    for p in sim.pat.values():
        if p.st == "future":
            continue
        w = (p.t_doc - p.t) if p.t_doc is not None else (p.t_lwbs - p.t) if p.t_lwbs is not None else t_end - p.t
        out[p.ats].append(w)
    return out


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return 0.0 if n == 0 else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def day_metrics(sim: EDSim, cfg: HospitalConfig, t_end: float = DAY_MIN) -> dict:
    m = compute_metrics(sim, 0.0, t_end)
    waits = waits_by_level(sim, t_end)
    n_all = sum(len(v) for v in waits.values())
    beds = max(cfg.effective_beds, 1)
    s = sim.samples
    return {
        "patients": n_all,
        "urgent_n": m.urgent_n,
        "urgent_late_pct": 100.0 * (1.0 - m.urgent_share),
        "lwbs": m.lwbs,
        "lwbs_pct": 100.0 * m.lwbs / n_all if n_all else 0.0,
        "median_wait_by_level": {l: _median(v) for l, v in waits.items()},
        "nonurgent_median_wait": m.nonurgent_median_wait,
        "peak_bed_occupancy_pct": 100.0 * max((x["beds"] for x in s), default=0) / beds,
        "longest_queue": max((x["tri_q"] + x["prov_q"] for x in s), default=0),
        "timeline": {"hour": [x["t"] / 60.0 for x in s],
                     "queue": [x["tri_q"] + x["prov_q"] for x in s],
                     "bed_pct": [100.0 * x["beds"] / beds for x in s],
                     "boarders": [x["board"] for x in s]},
    }


def snapshot_features(sim: EDSim, cfg: HospitalConfig, arrivals_last_hour: int) -> dict:
    """What a hospital can SEE at this moment (queues, beds, staff). Never reads the future."""
    now = sim.now
    q = {l: [] for l in range(1, 6)}
    boarders = 0
    for p in sim.pat.values():
        if p.st in ("tri_q", "bed_q", "doc_q"):
            q[p.ats].append(now - p.t)
        elif p.st == "board" and p.admit:
            boarders += 1
    f = {}
    for l in range(1, 6):
        f[f"queue_l{l}"] = len(q[l])
        f[f"longest_wait_l{l}"] = max(q[l], default=0.0)
    f["queue_total"] = sum(len(v) for v in q.values())
    f["bed_occupancy"] = sim.busy["b"] / max(sim.cap["b"], 1)
    f["boarders"] = boarders
    f["doctors_busy_frac"] = sim.busy["d"] / max(sim.cap["d"], 1)
    f["nurses_busy_frac"] = sim.busy["n"] / max(sim.cap["n"], 1)
    f["staff_doctors"] = sim.cap["d"]
    f["staff_nurses"] = sim.cap["n"]
    f["beds_total"] = sim.cap["b"]
    f["hour_of_day"] = (now / 60.0) % 24
    f["minutes_since_start"] = now
    f["arrivals_last_hour"] = arrivals_last_hour
    return f


# ------------------------------------------------------------------ a day with a chosen surge size
def generate_day(name: str, seed: int, intensity: Optional[float] = None, burst: bool = False):
    """Like scenarios.generate(), but the surge size can be FIXED by the user.

    ``intensity`` is the extra share of patients (0.10 = +10%). It must stay inside the published range
    (heatwave +6% to +13%, hurricane +9% to +13%); anything else is refused. None = sample like generate().
    """
    from src.simulation import scenarios as S
    if intensity is None or name not in S.SURGE_RANGES or S.USE_LEGACY_PARAMS:
        return S.generate(name, seed, burst=burst)
    r = S.SURGE_RANGES[name]
    m = 1.0 + float(intensity)
    if not (r["low"] - 1e-9 <= m <= r["high"] + 1e-9):
        raise ValueError(f"{name} surge must stay between +{(r['low'] - 1) * 100:.0f}% and +{(r['high'] - 1) * 100:.0f}% (published range)")
    sc = S._CURRENT_SCENARIOS[name]
    volume = S.BASE_VOLUME * S.weekday_factor("monday")
    vs = random.Random(f"{seed}:volume")
    vs.random()                      # the surge draw is skipped: the size is fixed
    factors = S.hourly_multipliers(name, m, hourly=S._CURRENT_HOURLY)
    rows = []
    for h in range(24):
        count = S._poisson_from_u(vs.random(), volume * S._CURRENT_HOURLY[h] * factors[h])
        rows += S._hour_rows(seed, h, count, sc.acuity, sc.admit, sc.boarding_mult)
    if burst:
        rows += S._burst_rows(seed)
    return S._to_arrivals(rows), S.scenario_info(name)


# ------------------------------------------------------------------ parameters added with the hospital / patient-flow / warning layers
HOSPITAL_PARAMETERS = [
    {"name": "Hospital presets (beds / nurses / doctors / volume)", "tag": "ASSUMPTION", "status": "unverified",
     "value": {k: [c.beds, c.triage_nurses, c.doctors, c.volume_scale] for k, c in PRESETS.items()},
     "note": "NHAMCS has no bed or staff counts. 'Large' (60 beds, 3 nurses, 4 doctors, ~200 patients/day) was picked by scripts/calibrate.py to match NHAMCS waits, "
             "left-unseen share and care times; Small and Medium are the same design scaled by volume and are NOT separately calibrated."},
    {"name": "Admission chance by triage level x age band", "tag": "DATA", "status": "verified",
     "value": "models/patient_flow/admission_lookup.json",
     "note": "NHAMCS 2022, PATWT-weighted, ADMITHOS == 1; small cells blended toward the triage rate (shrinkage k=10, an ASSUMPTION)."},
    {"name": "Care time by triage level and admitted / discharged (log-normal)", "tag": "DATA", "status": "verified",
     "value": "models/patient_flow/admission_lookup.json",
     "note": "LOV - WAITTIME is a PROXY (includes tests and boarding), not pure service time. Weighted by PATWT."},
    {"name": "Bed stay after the consult = care time - consult time (min 5 min, care time capped at 24 h)", "tag": "ASSUMPTION", "status": "unverified",
     "value": [5.0, 1440.0], "note": "Needed because the simulator holds a bed after the doctor consult."},
    {"name": "Fast-track bays", "tag": "ASSUMPTION", "status": "unverified", "value": "1+ bays = fast track on all day",
     "note": "The simulator has one fast-track switch (-40% doctor time for levels 4-5, an old placeholder). More bays do not change the effect."},
    {"name": "Ward-transfer speed", "tag": "ASSUMPTION", "status": "unverified", "value": "admitted patients leave 0 / 25 / 50% sooner, all day",
     "note": "Same mechanism as the optimiser's boarding cut. The plan search starts with this set to 0 and adds its own."},
    {"name": "Dashboard surge slider", "tag": "EVIDENCE", "status": "cited_by_team_not_independently_checked",
     "value": {"heatwave": "+6% to +13%", "hurricane": "+9% to +13%"},
     "note": "generate_day(intensity=...) refuses values outside the published range. Weather, FEMA and hurricane records never set a surge size."},
    {"name": "Calibration targets (urgent median wait, left-unseen share)", "tag": "DATA", "status": "verified",
     "value": {"urgent_median_wait_min": 15.0, "lwbs_pct": 1.89},
     "note": "Computed from NHAMCS 2022 (levels 1-3 WAITTIME, PATWT-weighted; LWBS flag). The simulated normal day gives about 5 min and 1.6%: see results/calibration.csv."},
    {"name": "Model seeds", "tag": "ASSUMPTION", "status": "unverified",
     "value": {"A1 split": 20261017, "A2 split": 20261018, "truth seeds": "100000 + day", "look-ahead seeds": "200000 + day"},
     "note": "Fixed so results repeat; the two seed ranges never overlap."},
    {"name": "Early-warning breach thresholds", "tag": "ASSUMPTION", "status": "unverified",
     "value": {"urgent_late": ">=2 late urgent patients and >=25% of urgent in the window; level-1 grace 10 min",
               "beds_full": "occupancy above 95%", "lwbs": ">=1 patient leaves unseen", "horizon": "next 4 h", "alarm line": "5% false alarms on normal days"},
     "note": "Our own definitions. The model learns the simulator, not a real hospital."},
]
