"""Clinical constants, literature-grounded scenario profiles and the arrival generator.

Provenance: hourly/acuity shares follow the NHAMCS 2018-22 sample distribution;
service times, admission rates and boarding durations are literature/synthetic
assumptions (the NHAMCS extract has no such fields).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import NamedTuple, Optional

HOURLY_ARRIVAL_PROPORTIONS = [
    0.024, 0.017, 0.013, 0.010, 0.009, 0.010,
    0.015, 0.025, 0.038, 0.052, 0.061, 0.065,
    0.064, 0.063, 0.063, 0.063, 0.062, 0.061,
    0.062, 0.061, 0.057, 0.048, 0.041, 0.030,
]
_S = sum(HOURLY_ARRIVAL_PROPORTIONS)
HOURLY = [x / _S for x in HOURLY_ARRIVAL_PROPORTIONS]  # normalised to sum to 1

BASE_VOLUME = 200  # patients / 24h (baseline range 180-220)
BASE_ACUITY = (0.012, 0.131, 0.384, 0.368, 0.105)
BASE_ADMIT = (0.45, 0.45, 0.25, 0.05, 0.05)
ATS_TARGET = {1: 0.0, 2: 10.0, 3: 30.0, 4: 60.0, 5: 120.0}  # minutes
ATS_WEIGHT = {1: 100, 2: 25, 3: 10, 4: 3, 5: 1}
GAMMA_SHAPE = 3.0
GAMMA_SCALE = {1: 15.0, 2: 10.0, 3: 6.67, 4: 4.0, 5: 2.67}
TURNAROUND = 45.0   # test/treatment turnaround for discharged patients (min)
LWBS_MIN = 180.0    # wait before provider consult that triggers LWBS
CAPACITY = {"nurses": 2, "docs": 5, "beds": 20}


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
    burst: Optional[tuple] = None   # (start_hour, end_hour, multiplier)
    elderly_surge_pct: float = 0.0

    def hour_mult(self, h: int) -> float:
        if self.burst and self.burst[0] <= h < self.burst[1]:
            return self.burst[2]
        return 1.0

    def expected_daily(self, base: float = BASE_VOLUME) -> float:
        return base * self.volume_mult * sum(HOURLY[h] * self.hour_mult(h) for h in range(24))


SCENARIOS = {
    "normal": Scenario("normal", "Normal Monday Baseline",
                       "Standard diurnal curve, triage mix and admission rates."),
    "heatwave": Scenario(
        "heatwave", "Extreme Heatwave (Houston 2011 / Nature Health 2025)",
        "Heat exhaustion, dehydration and cardiovascular distress; elderly surge, inpatient bed strain.",
        volume_mult=1.036, acuity=(0.012, 0.140, 0.420, 0.340, 0.088),
        admit=(0.50, 0.50, 0.30, 0.07, 0.07), boarding_mult=1.25, elderly_surge_pct=8.9),
    "hurricane": Scenario(
        "hurricane", "Hurricane Evacuee Influx (Harvey / DFW)",
        "Sustained demand lift with an afternoon burst of refills, lacerations and trauma.",
        volume_mult=1.11, acuity=(0.012, 0.120, 0.370, 0.380, 0.118), burst=(14, 20, 1.40)),
    "compound": Scenario(
        "compound", "Compound Calamity (Hurricane Beryl 2024)",
        "Storm aftermath with mass power outage in extreme heat: device failure, heat exposure, boarding gridlock.",
        volume_mult=1.35, acuity=(0.015, 0.170, 0.440, 0.290, 0.085),
        admit=(0.55, 0.55, 0.33, 0.08, 0.08), boarding_mult=1.5, elderly_surge_pct=8.9),
}


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
            out.append(Arrival(pid, ts, ats, rng.uniform(2.0, 4.0),
                               rng.gammavariate(GAMMA_SHAPE, GAMMA_SCALE[ats]),
                               rng.random() < sc.admit[ats - 1],
                               rng.uniform(120.0, 240.0) * sc.boarding_mult))
            pid += 1
    return out
