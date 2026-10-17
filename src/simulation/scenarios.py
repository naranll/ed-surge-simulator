"""Clinical constants, scenario profiles and the arrival generators.

Plain-English map of this file
------------------------------
* Constants such as BASE_VOLUME, ATS_TARGET or GAMMA_SHAPE are the numbers the simulator uses.
  PARAMETER_TABLE (bottom of the file) lists every one of them with a tag:
    DATA        computed from real data (only if the parameter file was loaded and verified)
    EVIDENCE    taken from a published study (cited by the team, not independently checked)
    ASSUMPTION  our own placeholder, stated openly
* Real parameters (hourly shape, triage mix) are read from
  src/simulation/params/scenario_parameters.json when that file exists and its entries are
  marked "verified". Otherwise the old hand-typed values are used and a warning is logged.
* The legacy switch (USE_LEGACY_PARAMS, env ED_USE_LEGACY_PARAMS=1, or use_legacy_params=True)
  brings back the ORIGINAL hand-typed numbers exactly.
* generate() builds the "true day" for the simulator. Surge sizes come only from published
  studies (SURGE_RANGES), never from NHAMCS, NOAA, FEMA or HURDAT2.
* Standard library only, so it runs on IBM LinuxONE (s390x).
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import NamedTuple, Optional

_log = logging.getLogger("src.simulation.scenarios")

# --------------------------------------------------------------------------------------
# 1. Hand-typed constants (the original numbers). All are ASSUMPTION / unverified.
# --------------------------------------------------------------------------------------
# Raw hourly shares (hand-typed, from a third-party 2018-22 summary, NOT the official NHAMCS file).
HOURLY_ARRIVAL_PROPORTIONS = [
    0.024, 0.017, 0.013, 0.010, 0.009, 0.010,
    0.015, 0.025, 0.038, 0.052, 0.061, 0.065,
    0.064, 0.063, 0.063, 0.063, 0.062, 0.061,
    0.062, 0.061, 0.057, 0.048, 0.041, 0.030,
]
_S = sum(HOURLY_ARRIVAL_PROPORTIONS)
_LEGACY_HOURLY = [x / _S for x in HOURLY_ARRIVAL_PROPORTIONS]  # normalised to sum to 1

BASE_VOLUME = 200  # patients / 24h on a MONDAY (baseline range 180-220)
_LEGACY_BASE_ACUITY = (0.012, 0.131, 0.384, 0.368, 0.105)
BASE_ADMIT = (0.45, 0.45, 0.25, 0.05, 0.05)
ATS_TARGET = {1: 0.0, 2: 10.0, 3: 30.0, 4: 60.0, 5: 120.0}  # minutes
ATS_WEIGHT = {1: 100, 2: 25, 3: 10, 4: 3, 5: 1}
GAMMA_SHAPE = 3.0
GAMMA_SCALE = {1: 15.0, 2: 10.0, 3: 6.67, 4: 4.0, 5: 2.67}
TURNAROUND = 45.0   # test/treatment turnaround for discharged patients (min)
LWBS_MIN = 180.0    # wait before provider consult that triggers LWBS
CAPACITY = {"nurses": 2, "docs": 5, "beds": 20}
TRIAGE_MINUTES = (2.0, 4.0)    # triage nurse time, uniform (min)
HOLD_MINUTES = (120.0, 240.0)  # inpatient boarding hold before scenario scaling, uniform (min)

TRIAGE_LEVEL_NOTE = ("Patient data uses NHAMCS triage levels 1-5; ATS time targets "
                     "(immediate/10/30/60/120 min) are applied to them as an ASSUMPTION.")

# Optional mass-casualty style burst (ASSUMPTION, stress test only, off by default).
BURST_SIZE_RANGE = (10, 20)     # number of very urgent arrivals, whole numbers, both ends included
BURST_HOUR_RANGE = (12, 21)     # the burst falls inside ONE hour, chosen between these hours
BURST_LEVEL1_SHARE = 0.25       # share of burst patients that are level 1 (rest are level 2)

# --------------------------------------------------------------------------------------
# 2. Published surge sizes (EVIDENCE). The ONLY source of surge sizes in this project.
# --------------------------------------------------------------------------------------
# Truth samples one multiplier per seed from [low, high]. The forecaster may use only the
# midpoint (public literature knowledge), never the sampled value.
# TODO (not modelled): older-patient shift. Houston 2011 heat-wave study: 65+ visits +8.9%
#   (5.1-12.8%) vs all ages +3.6%. We do NOT change the patient mix for it.
SURGE_RANGES = {
    "heatwave": {
        "low": 1.06, "high": 1.13, "midpoint": (1.06 + 1.13) / 2, "window": (12, 20),
        "tag": "EVIDENCE",
        "evidence_status": "cited_by_team_not_independently_checked",
        "source": ("Nature Health study of summer-2025 US heatwaves (8.3-9.2% of ED visits "
                   "attributable to heat, converted with 1/(1-AF)); team range +6% to +13% "
                   "all-cause visits."),
        "note": ("Attributable fraction includes lagged effects up to 4 days, so this is an "
                 "approximation. Extra arrivals are spread over hours 12-19 in proportion to "
                 "the baseline hourly shape."),
    },
    "hurricane": {
        "low": 1.09, "high": 1.13, "midpoint": (1.09 + 1.13) / 2, "window": (0, 24),
        "tag": "EVIDENCE",
        "evidence_status": "cited_by_team_not_independently_checked",
        "source": ("Hurricane Harvey, receiving hospital in Dallas-Fort Worth (Disaster Medicine "
                   "and Public Health Preparedness): +11% (9-13%) sustained."),
        "note": "Applied evenly over all 24 hours of the day.",
    },
}

# --------------------------------------------------------------------------------------
# 3. Loading the real parameter file (hourly shape, triage mix, weekday shares)
# --------------------------------------------------------------------------------------
_ENV_PARAMS = "ED_SCENARIO_PARAMS"
_ENV_LEGACY = "ED_USE_LEGACY_PARAMS"
_DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "params" / "scenario_parameters.json"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def params_path() -> Path:
    """Where the parameter file is looked for (env ED_SCENARIO_PARAMS overrides the default)."""
    override = os.environ.get(_ENV_PARAMS, "").strip()
    return Path(override) if override else _DEFAULT_PARAMS_PATH


def _check_distribution(entry, size: int, label: str):
    """Return (values, reason). values is None unless the entry is usable."""
    if not isinstance(entry, dict):
        return None, f"entry {label!r} is missing"
    if entry.get("status") != "verified":
        return None, f"status is {entry.get('status')!r}, not 'verified'"
    if entry.get("tag") != "DATA":
        return None, f"tag is {entry.get('tag')!r}, not 'DATA'"
    value = entry.get("value")
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None, f"value is not a list of {size} numbers"
    for x in value:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
            return None, "value contains something that is not a number >= 0"
    total = float(sum(value))
    if abs(total - 1.0) > 1e-6:
        return None, f"values sum to {total!r}, not 1"
    return tuple(float(x) / total for x in value), ""   # tiny re-normalise so sums are exactly 1


def _entry_info(entry) -> dict:
    keep = ("tag", "status", "weighting", "denominator", "source", "note")
    return {k: entry.get(k) for k in keep} if isinstance(entry, dict) else {}


def load_real_parameters(path=None, warn: bool = True) -> dict:
    """Read the parameter file (standard library json only). Never raises for bad content.

    Returns a dict. "hourly" / "acuity" are tuples when usable, else None (then the caller
    keeps the old constant and a warning that names the parameter is logged).
    """
    path = Path(path) if path is not None else params_path()
    out = {"path": str(path), "found": False, "error": "", "hourly": None, "acuity": None,
           "hourly_info": {}, "acuity_info": {}, "hourly_reason": "", "acuity_reason": "",
           "weekday_shares": None, "monday_factor": None, "source": None}
    raw = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            out["error"] = "top level of the file is not an object"
            raw = None
        else:
            out["found"] = True
    except FileNotFoundError:
        out["error"] = "file not found"
    except (OSError, ValueError) as exc:     # unreadable file or invalid JSON
        out["error"] = f"cannot read file ({exc.__class__.__name__})"
    raw = raw or {}
    out["hourly"], out["hourly_reason"] = _check_distribution(raw.get("hourly_shape"), 24, "hourly_shape")
    out["acuity"], out["acuity_reason"] = _check_distribution(raw.get("triage_mix"), 5, "triage_mix")
    out["hourly_info"] = _entry_info(raw.get("hourly_shape"))
    out["acuity_info"] = _entry_info(raw.get("triage_mix"))
    out["weekday_shares"] = raw.get("weekday_shares")
    out["monday_factor"] = raw.get("monday_factor")
    out["source"] = raw.get("source") if isinstance(raw.get("source"), dict) else None
    if warn:
        for key, name, old in (("hourly", "hourly_shape", "HOURLY"), ("acuity", "triage_mix", "BASE_ACUITY")):
            if out[key] is None:
                why = out["error"] or out[key + "_reason"]
                _log.warning("scenario parameter %r not loaded from %s (%s); using the hand-typed "
                             "placeholder %s (ASSUMPTION, unverified)", name, path, why, old)
    return out


_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def weekday_factor(weekday: str = "monday", params: Optional[dict] = None) -> float:
    """Volume factor for a weekday. BASE_VOLUME is a Monday figure, so Monday is always 1.0.

    Other days = share_of_that_day / share_of_Monday, but ONLY when the parameter file has
    verified weekday_shares (with value labels naming the days) and a verified monday_factor.
    Otherwise a ValueError explains what is missing (we never guess a weekday effect).
    """
    day = str(weekday).strip().lower()
    if day not in _WEEKDAYS:
        raise ValueError(f"unknown weekday {weekday!r}; use one of {_WEEKDAYS}")
    if day == "monday":
        return 1.0
    real = _REAL if params is None else params
    where = real.get("path", "the parameter file")

    def bad(why):
        return ValueError(f"no verified weekday factor for {day!r} ({why}); parameter file: {where}")

    shares, mf = real.get("weekday_shares"), real.get("monday_factor")
    if not isinstance(shares, dict) or shares.get("status") != "verified":
        raise bad("weekday_shares is missing or not 'verified'")
    if not isinstance(mf, dict) or mf.get("status") != "verified" or not isinstance(mf.get("value"), (int, float)):
        raise bad("monday_factor is missing, null or not 'verified'")
    values, labels = shares.get("value"), shares.get("code_labels")
    if not isinstance(values, dict) or not isinstance(labels, dict):
        raise bad("weekday_shares needs value and code_labels")

    def code_for(name):
        hits = [c for c, lab in labels.items() if name in str(lab).lower()]
        return hits[0] if len(hits) == 1 else None

    c_day, c_mon = code_for(day), code_for("monday")
    if c_day is None or c_mon is None or c_day not in values or c_mon not in values:
        raise bad("cannot match the day names to the value labels")
    share_day, share_mon = float(values[c_day]), float(values[c_mon])
    if share_mon <= 0 or share_day <= 0:
        raise bad("a weekday share is not positive")
    if abs(share_mon * 7 - float(mf["value"])) > 1e-3:
        raise bad("monday_factor does not match the Monday share")
    return share_day / share_mon


# --------------------------------------------------------------------------------------
# 4. Active values: loaded (if verified) or the old constants; legacy switch
# --------------------------------------------------------------------------------------
USE_LEGACY_PARAMS = _env_flag(_ENV_LEGACY)
_REAL = load_real_parameters(warn=not USE_LEGACY_PARAMS)

_CURRENT_HOURLY = list(_REAL["hourly"]) if _REAL["hourly"] else list(_LEGACY_HOURLY)
_CURRENT_ACUITY = tuple(_REAL["acuity"]) if _REAL["acuity"] else _LEGACY_BASE_ACUITY

# The names other modules import. They follow the switch value read when this module loads.
HOURLY = list(_LEGACY_HOURLY) if USE_LEGACY_PARAMS else list(_CURRENT_HOURLY)  # 24 floats, sum 1
BASE_ACUITY = _LEGACY_BASE_ACUITY if USE_LEGACY_PARAMS else _CURRENT_ACUITY    # levels 1-5, sum 1


def hourly_multipliers(name: str, m: float, window: Optional[tuple] = None,
                       hourly: Optional[list] = None) -> list:
    """The ONE place that turns (scenario, surge multiplier m) into 24 hourly factors.

    Used by the truth (sampled m) and by the forecaster (midpoint m), so both agree on
    WHEN the extra patients arrive. factor[h] = expected arrivals in hour h / baseline arrivals.
    The extra visits (m - 1) x daily volume are placed inside the window in proportion to the
    baseline hourly shape, so the whole-day total is exactly m x baseline.
    Scenarios without a published surge (and no window given) get all 1.0.
    """
    if window is None:
        window = SURGE_RANGES[name]["window"] if name in SURGE_RANGES else None
    if window is None:
        return [1.0] * 24
    start, end = window
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= 24):
        raise ValueError(f"window must be (start, end) hours with 0 <= start < end <= 24, got {window!r}")
    if not (isinstance(m, (int, float)) and math.isfinite(m) and m > 0):
        raise ValueError(f"surge multiplier must be a positive number, got {m!r}")
    shape = HOURLY if hourly is None else hourly
    inside = sum(shape[start:end])
    if inside <= 0:
        raise ValueError("baseline arrivals inside the window are zero")
    boost = 1.0 + (m - 1.0) / inside
    return [boost if start <= h < end else 1.0 for h in range(24)]


# --------------------------------------------------------------------------------------
# 5. Patients and scenarios
# --------------------------------------------------------------------------------------
class Arrival(NamedTuple):
    """One patient with every random draw pre-sampled (enables CRN)."""
    pid: int
    t: float         # arrival minute (absolute)
    ats: int
    tri: float       # triage service time
    consult: float   # physician face-to-face time
    admit: bool
    hold: float      # inpatient boarding hold (already scenario-scaled)


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    description: str
    volume_mult: float = 1.0
    acuity: tuple = BASE_ACUITY
    admit: tuple = BASE_ADMIT
    boarding_mult: float = 1.0
    burst: Optional[tuple] = None   # (start_hour, end_hour, multiplier)  [legacy hand-typed burst]
    elderly_surge_pct: float = 0.0
    hourly_mult: Optional[tuple] = None   # 24 factors from hourly_multipliers() at the midpoint

    def hour_mult(self, h: int) -> float:
        m = self.hourly_mult[h % 24] if self.hourly_mult else 1.0
        if self.burst and self.burst[0] <= h < self.burst[1]:
            m *= self.burst[2]
        return m

    def expected_daily(self, base: float = BASE_VOLUME, hourly: Optional[list] = None) -> float:
        shape = HOURLY if hourly is None else hourly
        return base * self.volume_mult * sum(shape[h] * self.hour_mult(h) for h in range(24))


_HEATWAVE_NAME = "Extreme Heatwave (Houston 2011 / Nature Health 2025)"
_HEATWAVE_DESC = ("Heat exhaustion, dehydration and cardiovascular distress. Size: all-cause visits "
                  "+6% to +13% (Nature Health study of summer-2025 US heatwaves; includes lagged "
                  "effects up to 4 days), concentrated in the afternoon. The older-patient shift "
                  "(65+ visits +8.9%, Houston 2011 heat-wave study) is NOT modelled yet.")
_COMPOUND_NAME = "Compound Calamity (assumed stress test)"
_COMPOUND_DESC = ("Storm aftermath with mass power outage in extreme heat: device failure, heat "
                  "exposure, boarding gridlock. Every number is a hand-typed ASSUMPTION.")


def _legacy_scenarios() -> dict:
    """The original hand-typed profiles (numbers unchanged)."""
    return {
        "normal": Scenario("normal", "Normal Monday Baseline",
                           "Standard diurnal curve, triage mix and admission rates.",
                           acuity=_LEGACY_BASE_ACUITY),
        "heatwave": Scenario(
            "heatwave", _HEATWAVE_NAME, _HEATWAVE_DESC,
            volume_mult=1.036, acuity=(0.012, 0.140, 0.420, 0.340, 0.088),
            admit=(0.50, 0.50, 0.30, 0.07, 0.07), boarding_mult=1.25, elderly_surge_pct=8.9),
        "hurricane": Scenario(
            "hurricane", "Hurricane Evacuee Influx (Harvey / DFW)",
            "Sustained demand lift with an afternoon burst of refills, lacerations and trauma.",
            volume_mult=1.11, acuity=(0.012, 0.120, 0.370, 0.380, 0.118), burst=(14, 20, 1.40)),
        "compound": Scenario(
            "compound", _COMPOUND_NAME, _COMPOUND_DESC,
            volume_mult=1.35, acuity=(0.015, 0.170, 0.440, 0.290, 0.085),
            admit=(0.55, 0.55, 0.33, 0.08, 0.08), boarding_mult=1.5, elderly_surge_pct=8.9),
    }


def _current_scenarios() -> dict:
    """Profiles without hand-typed clinical shifts: surge sizes come from SURGE_RANGES only."""
    def surge(name):   # midpoint = the expected value, used by old-style callers (no sampling)
        r = SURGE_RANGES[name]
        return tuple(hourly_multipliers(name, r["midpoint"], hourly=_CURRENT_HOURLY))

    return {
        "normal": Scenario("normal", "Normal Monday Baseline",
                           "Standard diurnal curve, triage mix and admission rates.",
                           acuity=_CURRENT_ACUITY),
        "heatwave": Scenario("heatwave", _HEATWAVE_NAME, _HEATWAVE_DESC, acuity=_CURRENT_ACUITY,
                             hourly_mult=surge("heatwave")),     # 65+ shift not modelled -> 0.0
        "hurricane": Scenario(
            "hurricane", "Hurricane Evacuee Influx (Harvey / DFW)",
            "Sustained +9% to +13% demand lift at a receiving hospital (Hurricane Harvey, "
            "Dallas-Fort Worth study). An optional burst of very urgent arrivals is an "
            "ASSUMPTION stress test.",
            acuity=_CURRENT_ACUITY, hourly_mult=surge("hurricane")),
        "compound": Scenario(
            "compound", _COMPOUND_NAME, _COMPOUND_DESC,
            volume_mult=1.35, acuity=(0.015, 0.170, 0.440, 0.290, 0.085),
            admit=(0.55, 0.55, 0.33, 0.08, 0.08), boarding_mult=1.5),
    }


_LEGACY_SCENARIOS = _legacy_scenarios()
_CURRENT_SCENARIOS = _current_scenarios()
SCENARIOS = _LEGACY_SCENARIOS if USE_LEGACY_PARAMS else _CURRENT_SCENARIOS


# --------------------------------------------------------------------------------------
# 6. Old arrival generator (unchanged logic; callers: ed_simulator, optimizer, run.py)
# --------------------------------------------------------------------------------------
def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 500:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    lim, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= lim:
            return k
        k += 1


def generate_arrivals(sc: Scenario, seed: int, t0: float = 0.0, t1: float = 1440.0,
                      noise: float = 0.0, base_pid: int = 0,
                      volume: float = BASE_VOLUME) -> list[Arrival]:
    """Non-homogeneous Poisson stream on [t0, t1) minutes.

    ``noise`` > 0 multiplies each hourly rate by lognormal noise (the "true day");
    forecasts use noise=0. Same (sc, seed, window) always yields the same list.
    """
    if t1 < t0:
        raise ValueError("t1 must be >= t0")
    rng, out, pid = random.Random(seed), [], base_pid
    for H in range(int(t0 // 60), int(math.ceil(t1 / 60))):
        f = math.exp(rng.gauss(0.0, noise)) if noise > 0 else 1.0
        a, b = max(t0, H * 60.0), min(t1, (H + 1) * 60.0)
        if b <= a:
            continue
        h = H % 24
        lam = volume * sc.volume_mult * HOURLY[h] * sc.hour_mult(h) * f * (b - a) / 60.0
        for ts in sorted(rng.uniform(a, b) for _ in range(_poisson(rng, lam))):
            ats = rng.choices(range(1, 6), sc.acuity)[0]
            out.append(Arrival(pid, ts, ats, rng.uniform(*TRIAGE_MINUTES),
                               rng.gammavariate(GAMMA_SHAPE, GAMMA_SCALE[ats]),
                               rng.random() < sc.admit[ats - 1],
                               rng.uniform(*HOLD_MINUTES) * sc.boarding_mult))
            pid += 1
    return out


# --------------------------------------------------------------------------------------
# 7. New interface: true day (generate), seeds, forecast patients
# --------------------------------------------------------------------------------------
_SEED_SPAN = 100_000
_TRUTH_BASE = 100_000
_LOOKAHEAD_BASE = 200_000


def _check_n(n) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n < _SEED_SPAN:
        raise ValueError(f"n must be a whole number with 0 <= n < {_SEED_SPAN}, got {n!r}")
    return n


def truth_seed(n: int) -> int:
    """Seed for a TRUE day: 100000 + n. Never overlaps lookahead_seed."""
    return _TRUTH_BASE + _check_n(n)


def lookahead_seed(n: int) -> int:
    """Seed for an optimiser look-ahead run: 200000 + n. Never overlaps truth_seed."""
    return _LOOKAHEAD_BASE + _check_n(n)


def list_scenarios() -> list:
    """Scenarios offered to users. 'compound' still works by name but is not listed."""
    return ["normal", "heatwave", "hurricane"]


# What a hospital knows in advance. A pure function of the scenario NAME.
_ADVANCE_INFO = {
    "normal": (False, "n/a", None, False),
    "heatwave": (True, "day_before", SURGE_RANGES["heatwave"]["window"], False),
    "hurricane": (True, "day_before", SURGE_RANGES["hurricane"]["window"], True),
    "compound": (True, "day_before", (0, 24), True),
}


def scenario_info(name: str) -> dict:
    """Advance information for a scenario. Never depends on seed, burst flag or sampled values."""
    if name not in _ADVANCE_INFO:
        raise ValueError(f"unknown scenario {name!r}; known: {sorted(_ADVANCE_INFO)}")
    known, source, window, burst_possible = _ADVANCE_INFO[name]
    return {"scenario": name, "surge_known": known, "known_from": source,
            "window": window, "burst_possible": burst_possible}


def _multiplier_from_u(name: str, u: float) -> float:
    r = SURGE_RANGES[name]
    return r["low"] + (r["high"] - r["low"]) * u      # same as random.uniform(low, high)


def _sample_multiplier(name: str, seed) -> float:
    """The surge multiplier the truth uses for this seed (private: tests only)."""
    return _multiplier_from_u(name, random.Random(f"{seed}:volume").random())


def _poisson_from_u(u: float, lam: float) -> int:
    """Poisson count by inverse CDF: ONE uniform number per hour keeps seeds comparable."""
    if lam <= 0:
        return 0
    if lam > 500:
        z = NormalDist().inv_cdf(min(max(u, 1e-12), 1.0 - 1e-12))
        return max(0, round(lam + math.sqrt(lam) * z))
    k, p = 0, math.exp(-lam)
    cum, limit = p, int(lam + 12.0 * math.sqrt(lam) + 50)
    while cum < u and k < limit:
        k += 1
        p *= lam / k
        cum += p
    return k


def _level_from_u(u: float, cumulative: list) -> int:
    for level, c in enumerate(cumulative, start=1):
        if u < c:
            return level
    return len(cumulative)


def _cumulative(mix) -> list:
    out, total = [], 0.0
    for x in mix:
        total += x
        out.append(total)
    return out


def _hour_rows(seed, abs_hour: int, count: int, acuity, admit, boarding_mult: float) -> list:
    """Patients for one hour as (t, ats, tri, consult, admit, hold) tuples.

    Timing, acuity and care each have their own stream, restarted every hour, so the i-th
    patient of an hour keeps the same random numbers in every scenario (common random numbers).
    """
    if count <= 0:
        return []
    a, b = abs_hour * 60.0, (abs_hour + 1) * 60.0
    timing = random.Random(f"{seed}:timing:{abs_hour}")
    mix = random.Random(f"{seed}:acuity:{abs_hour}")
    care = random.Random(f"{seed}:care:{abs_hour}")
    cum = _cumulative(acuity)
    rows = []
    for _ in range(count):
        t = timing.uniform(a, b)
        ats = _level_from_u(mix.random(), cum)
        tri = care.uniform(*TRIAGE_MINUTES)
        consult = GAMMA_SCALE[ats] * care.gammavariate(GAMMA_SHAPE, 1.0)
        admitted = care.random() < admit[ats - 1]
        hold = care.uniform(*HOLD_MINUTES) * boarding_mult
        rows.append((t, ats, tri, consult, admitted, hold))
    return rows


def _burst_rows(seed) -> list:
    """Optional burst: 10-20 level 1-2 patients inside one hour. Burst stream only. ASSUMPTION."""
    rng = random.Random(f"{seed}:burst")
    hour = rng.randint(*BURST_HOUR_RANGE)
    count = rng.randint(*BURST_SIZE_RANGE)
    rows = []
    for _ in range(count):
        t = rng.uniform(hour * 60.0, (hour + 1) * 60.0)
        ats = 1 if rng.random() < BURST_LEVEL1_SHARE else 2
        tri = rng.uniform(*TRIAGE_MINUTES)
        consult = GAMMA_SCALE[ats] * rng.gammavariate(GAMMA_SHAPE, 1.0)
        admitted = rng.random() < BASE_ADMIT[ats - 1]
        hold = rng.uniform(*HOLD_MINUTES)
        rows.append((t, ats, tri, consult, admitted, hold))
    return rows


def _to_arrivals(rows: list, base_pid: int = 0) -> list:
    rows = sorted(rows, key=lambda r: r[0])           # stable: ties keep draw order
    return [Arrival(base_pid + i, *r) for i, r in enumerate(rows)]


def generate(name: str, seed: int, burst: bool = False, use_legacy_params: Optional[bool] = None,
             weekday: str = "monday"):
    """Build the TRUE day for the live simulator.

    Returns (truth, info). truth: list of Arrival for 00:00-24:00, sorted by time, pid 0..n-1.
    info: what the hospital knows in advance (see scenario_info); it never contains the
    sampled surge size, the seed or the burst hour, so the optimiser cannot peek.

    Random streams (each random.Random(f"{seed}:{stream}")): volume (surge size and hourly
    counts), timing, acuity, care (consult / admit / boarding) and burst. Per-hour streams
    add ":{hour}" so the same seed gives common random numbers across scenarios.
    """
    legacy = USE_LEGACY_PARAMS if use_legacy_params is None else bool(use_legacy_params)
    scenarios = _LEGACY_SCENARIOS if legacy else _CURRENT_SCENARIOS
    if name not in scenarios:
        raise ValueError(f"unknown scenario {name!r}; known: {sorted(scenarios)}")
    sc = scenarios[name]
    hourly = _LEGACY_HOURLY if legacy else _CURRENT_HOURLY
    volume = BASE_VOLUME * weekday_factor(weekday)

    volume_stream = random.Random(f"{seed}:volume")
    u_surge = volume_stream.random()          # always drawn, so every scenario stays aligned
    if not legacy and name in SURGE_RANGES:
        factors = hourly_multipliers(name, _multiplier_from_u(name, u_surge), hourly=hourly)
    else:
        factors = [sc.volume_mult * sc.hour_mult(h) for h in range(24)]
    rows = []
    for h in range(24):
        count = _poisson_from_u(volume_stream.random(), volume * hourly[h] * factors[h])
        rows += _hour_rows(seed, h, count, sc.acuity, sc.admit, sc.boarding_mult)
    if burst:
        rows += _burst_rows(seed)
    return _to_arrivals(rows), scenario_info(name)


def patients_from_rates(rates, seed, start_hour: int = 0, base_pid: int = 0) -> list:
    """Turn expected arrivals per hour into patients (used for forecast / look-ahead runs).

    rates[i] is the expected number of arrivals in hour start_hour + i. Each hour gets a
    Poisson count, uniform times inside the hour, levels from the triage mix and care draws
    like generate(). Times are absolute minutes (hour 25 starts at minute 1500). Forecast
    patients should use base_pid >= 10**6 so they never clash with live patients.
    """
    if isinstance(start_hour, bool) or not isinstance(start_hour, int) or start_hour < 0:
        raise ValueError(f"start_hour must be a whole number >= 0, got {start_hour!r}")
    lams = []
    for r in rates:
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r < 0:
            raise ValueError(f"every rate must be a number >= 0, got {r!r}")
        lams.append(float(r))
    volume_stream = random.Random(f"{seed}:volume")
    rows = []
    for i, lam in enumerate(lams):
        count = _poisson_from_u(volume_stream.random(), lam)
        rows += _hour_rows(seed, start_hour + i, count, BASE_ACUITY, BASE_ADMIT, 1.0)
    return _to_arrivals(rows, base_pid)


# --------------------------------------------------------------------------------------
# 8. Parameter table: every parameter in this file, with tag and status
# --------------------------------------------------------------------------------------
_PLACEHOLDER = "placeholder"
_CITED = "cited_by_team_not_independently_checked"


def parameter_table(legacy: Optional[bool] = None) -> list:
    """List of {name, value, tag, status, note} dicts covering every parameter constant."""
    legacy = USE_LEGACY_PARAMS if legacy is None else bool(legacy)
    rows = []

    def add(name, value, tag="ASSUMPTION", status="unverified", note=_PLACEHOLDER):
        rows.append({"name": name, "value": value, "tag": tag, "status": status, "note": note})

    def data_or_placeholder(key, label):
        """(tag, status, note) for a value that may have been loaded from the parameter file."""
        loaded = (not legacy) and _REAL[key] is not None
        if loaded:
            info = _REAL[key + "_info"]
            note = (f"loaded from {_REAL['path']}; variable {info.get('source')}; weighting "
                    f"{info.get('weighting')}; denominator {info.get('denominator')}; "
                    "re-normalised to sum exactly 1")
            return "DATA", "verified", note
        why = "legacy switch is ON" if legacy else (_REAL["error"] or _REAL[key + "_reason"])
        return "ASSUMPTION", "unverified", f"{_PLACEHOLDER}: hand-typed {label} ({why})"

    add("HOURLY_ARRIVAL_PROPORTIONS", HOURLY_ARRIVAL_PROPORTIONS, note=(
        f"{_PLACEHOLDER}: hand-typed raw hourly shares from a third-party 2018-22 summary, "
        "not the official NHAMCS file"))
    tag, status, note = data_or_placeholder("hourly", "hourly arrival shape")
    add("HOURLY", list(_LEGACY_HOURLY if legacy else _CURRENT_HOURLY), tag, status, note)
    add("BASE_VOLUME", BASE_VOLUME, note=f"{_PLACEHOLDER}: patients per MONDAY (range 180-220)")
    a_tag, a_status, a_note = data_or_placeholder("acuity", "triage mix")
    add("BASE_ACUITY", _LEGACY_BASE_ACUITY if legacy else _CURRENT_ACUITY, a_tag, a_status, a_note)
    add("BASE_ADMIT", BASE_ADMIT, note=f"{_PLACEHOLDER}: admission probability by level")
    add("ATS_TARGET", ATS_TARGET, note=f"{_PLACEHOLDER}: see TRIAGE_LEVEL_NOTE")
    add("TRIAGE_LEVEL_NOTE", TRIAGE_LEVEL_NOTE, note="assumption statement")
    add("ATS_WEIGHT", ATS_WEIGHT, note=f"{_PLACEHOLDER}: loss weight by level")
    add("GAMMA_SHAPE", GAMMA_SHAPE, note=f"{_PLACEHOLDER}: doctor consult time shape (no consult field in NHAMCS)")
    add("GAMMA_SCALE", GAMMA_SCALE, note=f"{_PLACEHOLDER}: doctor consult time scale by level")
    add("TURNAROUND", TURNAROUND, note=f"{_PLACEHOLDER}: minutes a discharged patient keeps the bed")
    add("LWBS_MIN", LWBS_MIN, note=f"{_PLACEHOLDER}: minutes waiting before a patient leaves unseen")
    add("CAPACITY", CAPACITY, note=f"{_PLACEHOLDER}: nurses, doctors, beds on shift")
    add("TRIAGE_MINUTES", TRIAGE_MINUTES, note=f"{_PLACEHOLDER}: triage nurse time (confirm with a clinician)")
    add("HOLD_MINUTES", HOLD_MINUTES, note=f"{_PLACEHOLDER}: inpatient boarding hold range")
    add("BURST_SIZE_RANGE", BURST_SIZE_RANGE, note=f"{_PLACEHOLDER}: stress test only, 10-20 level 1-2 arrivals")
    add("BURST_HOUR_RANGE", BURST_HOUR_RANGE, note=f"{_PLACEHOLDER}: burst falls in one hour chosen between these hours")
    add("BURST_LEVEL1_SHARE", BURST_LEVEL1_SHARE, note=f"{_PLACEHOLDER}: share of burst patients at level 1")
    add("SURGE_RANGES", {k: (r["low"], r["high"]) for k, r in SURGE_RANGES.items()}, "EVIDENCE", _CITED,
        "published all-cause ED visit multipliers; the only source of surge sizes (never NHAMCS, "
        "NOAA, FEMA or HURDAT2). The truth samples one value per seed, the forecaster uses the midpoint")
    for name, r in SURGE_RANGES.items():
        add(f"SURGE_RANGES.{name}", (r["low"], r["high"]), r["tag"], r["evidence_status"],
            f"{r['source']} {r['note']}")
    add("weekday_factor", "Monday = 1.0",
        note=f"{_PLACEHOLDER}: other weekdays need verified weekday_shares and monday_factor in the "
             "parameter file; without them weekday_factor() raises an error")

    for key, sc in (_LEGACY_SCENARIOS if legacy else _CURRENT_SCENARIOS).items():
        shared_mix = (not legacy) and key != "compound"
        surge_model = (not legacy) and key in SURGE_RANGES
        add(f"SCENARIOS.{key}.volume_mult", sc.volume_mult, note=(
            f"{_PLACEHOLDER}: neutral, surge applied through hourly_mult (SURGE_RANGES)" if surge_model
            else f"{_PLACEHOLDER}: hand-typed point value" + (" (legacy only)" if legacy and key != "normal" else "")))
        if sc.hourly_mult:
            add(f"SCENARIOS.{key}.hourly_mult", list(sc.hourly_mult), SURGE_RANGES[key]["tag"], _CITED,
                "SURGE_RANGES midpoint spread by hourly_multipliers(); old-style callers use this expected value")
        if shared_mix:
            add(f"SCENARIOS.{key}.acuity", sc.acuity, a_tag, a_status, f"same mix as BASE_ACUITY. {a_note}")
        else:
            add(f"SCENARIOS.{key}.acuity", sc.acuity, note=f"{_PLACEHOLDER}: hand-typed mix, no study supports it")
        add(f"SCENARIOS.{key}.admit", sc.admit, note=f"{_PLACEHOLDER}: hand-typed admission probabilities")
        add(f"SCENARIOS.{key}.boarding_mult", sc.boarding_mult, note=f"{_PLACEHOLDER}: hand-typed boarding scale")
        add(f"SCENARIOS.{key}.burst", sc.burst, note=(
            f"{_PLACEHOLDER}: hand-typed hourly burst multiplier (legacy only)" if sc.burst
            else f"{_PLACEHOLDER}: none; the optional burst uses the BURST_* values"))
        add(f"SCENARIOS.{key}.elderly_surge_pct", sc.elderly_surge_pct, note=(
            f"{_PLACEHOLDER}: legacy label only, NOT modelled (study figure: 65+ visits +8.9%, Houston 2011)"
            if sc.elderly_surge_pct else
            f"{_PLACEHOLDER}: 0 = older-patient shift NOT modelled (TODO; study figure 65+ visits +8.9%, Houston 2011)"))
    return rows


PARAMETER_TABLE = parameter_table()
