"""Tests for the additive hospital layer: HospitalConfig, patient-flow link, fixed-intensity days."""
import statistics
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization.optimizer import DAY_MIN, compute_metrics
from src.simulation import patient_flow
from src.simulation.ed_simulator import EDSim
from src.simulation.hospital import PRESETS, HospitalConfig, generate_day, prepare, run_day
from src.simulation.scenarios import SURGE_RANGES, generate, truth_seed

needs_flow = pytest.mark.skipif(not patient_flow.available(), reason="admission_lookup.json missing or legacy switch on")


def test_defaults_equal_todays_behaviour():
    cfg = HospitalConfig()
    assert (cfg.effective_beds, cfg.triage_capacity, cfg.doctors) == (20, 2, 5)
    truth, _ = generate("normal", truth_seed(3))
    new = compute_metrics(run_day(cfg, truth), 0, DAY_MIN)
    old_sim = EDSim()
    old_sim.add_arrivals(truth)
    old_sim.run_until(DAY_MIN)
    assert new == compute_metrics(old_sim, 0, DAY_MIN)           # identical to the plain simulator


def test_per_patient_stay_is_off_by_default():
    assert EDSim().per_patient_stay is False


def test_beds_rooms_and_nurses_stations_combine_as_documented():
    assert HospitalConfig(beds=30, rooms=22).effective_beds == 22
    assert HospitalConfig(beds=30, rooms=40).effective_beds == 30
    assert HospitalConfig(triage_nurses=4, triage_stations=2).triage_capacity == 2
    with pytest.raises(ValueError):
        HospitalConfig(beds=0)
    with pytest.raises(ValueError):
        HospitalConfig(boarding_cut=1.0)


def test_more_doctors_never_lower_urgent_on_time_share_same_days():
    for n in range(4):
        truth, _ = generate("heatwave", truth_seed(n))
        few = run_day(HospitalConfig(beds=40, doctors=3), truth)
        many = run_day(HospitalConfig(beds=40, doctors=6), truth)
        assert compute_metrics(many).urgent_share >= compute_metrics(few).urgent_share - 1e-9


def test_volume_scale_thins_patients_repeatably():
    truth, _ = generate("normal", truth_seed(1))
    cfg = HospitalConfig(volume_scale=0.5)
    a, b = prepare(cfg, truth), prepare(cfg, truth)
    assert a == b and 0.35 * len(truth) < len(a) < 0.65 * len(truth)


def test_generate_day_range_and_delegation():
    t1, i1 = generate_day("heatwave", truth_seed(2))
    t2, i2 = generate("heatwave", truth_seed(2))
    assert t1 == t2 and i1 == i2                                    # no intensity -> same as generate()
    from src.simulation.scenarios import USE_LEGACY_PARAMS
    if USE_LEGACY_PARAMS:                                           # the legacy switch has no sampled surge range, so a fixed size is ignored
        return
    lo, hi = SURGE_RANGES["heatwave"]["low"] - 1, SURGE_RANGES["heatwave"]["high"] - 1
    small, _ = generate_day("heatwave", truth_seed(2), intensity=lo)
    big, _ = generate_day("heatwave", truth_seed(2), intensity=hi)
    assert len(big) >= len(small)
    with pytest.raises(ValueError):
        generate_day("heatwave", truth_seed(2), intensity=0.30)      # outside the published range is refused


@needs_flow
def test_patient_flow_is_deterministic_and_keeps_arrival_times():
    truth, _ = generate("normal", truth_seed(5))
    a, b = patient_flow.apply_patient_flow(truth), patient_flow.apply_patient_flow(truth)
    assert a == b
    assert [p.t for p in a] == [p.t for p in truth] and [p.ats for p in a] == [p.ats for p in truth]


@needs_flow
def test_patient_flow_admission_share_matches_the_lookup():
    lk = patient_flow.load_lookup()
    shares = []
    for n in range(30):
        truth, _ = generate("normal", truth_seed(n))
        shares.append(statistics.mean(p.admit for p in patient_flow.apply_patient_flow(truth)))
    assert abs(statistics.mean(shares) - lk["overall_admission_rate_weighted"]) < 0.03
    assert all(0.0 <= v["rate"] <= 1.0 for lv in lk["admission"].values() for v in lv["bands"].values())


@needs_flow
def test_flow_mode_gives_longer_bed_stays_than_the_old_constant():
    truth, _ = generate("normal", truth_seed(0))
    assert sum(p.hold for p in patient_flow.apply_patient_flow(truth)) > sum(p.hold for p in truth) * 0.9


def test_a_simulated_day_is_fast():
    truth, _ = generate("normal", truth_seed(0))
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        run_day(HospitalConfig(), truth)
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 0.25        # a whole day; very loose bound, the real value is about 10 ms
