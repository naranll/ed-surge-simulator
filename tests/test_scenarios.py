"""Tests for the scenario generator (truth day) and the forecaster.

Most tests use PRIVATE COPIES of src/simulation/scenarios.py loaded with a clean environment
(no ED_* variables, parameter file pointing nowhere), so results do not depend on whether the
real parameter file exists or on the legacy switch. A few smoke tests use the real modules.
No test reads data/ or anything outside this repo.

Run:  py -3.12 -B -m unittest tests.test_scenarios
"""
import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
import logging
import math
import os
import platform
import random
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from src.simulation import forecaster as real_forecaster
from src.simulation import scenarios as real_scenarios
from src.simulation.ed_simulator import EDSim

LOGGER = "src.simulation.scenarios"
SCENARIOS_PATH = Path(real_scenarios.__file__)
FORECASTER_PATH = Path(real_forecaster.__file__)
ENV_PARAMS, ENV_LEGACY = "ED_SCENARIO_PARAMS", "ED_USE_LEGACY_PARAMS"
INFO_KEYS = {"scenario", "surge_known", "known_from", "window", "burst_possible"}

# ---------------------------------------------------------------------------------------
# Golden values captured from the ORIGINAL scenarios.py (commit 480f9eb), before any edit.
# ---------------------------------------------------------------------------------------
GOLDEN_HOURLY = [
    0.023668639053254437, 0.01676528599605523, 0.01282051282051282, 0.009861932938856016,
    0.008875739644970413, 0.009861932938856016, 0.014792899408284023, 0.02465483234714004,
    0.03747534516765286, 0.05128205128205128, 0.06015779092702169, 0.06410256410256411,
    0.0631163708086785, 0.0621301775147929, 0.0621301775147929, 0.0621301775147929,
    0.0611439842209073, 0.06015779092702169, 0.0611439842209073, 0.06015779092702169,
    0.05621301775147929, 0.047337278106508875, 0.04043392504930966, 0.029585798816568046]
GOLDEN_BASE_ACUITY = (0.012, 0.131, 0.384, 0.368, 0.105)
GOLDEN_SCENARIOS = {   # key: (volume_mult, acuity, admit, boarding_mult, burst, elderly_surge_pct, expected_daily)
    "normal": (1.0, (0.012, 0.131, 0.384, 0.368, 0.105), (0.45, 0.45, 0.25, 0.05, 0.05), 1.0, None, 0.0, 200.0),
    "heatwave": (1.036, (0.012, 0.14, 0.42, 0.34, 0.088), (0.5, 0.5, 0.3, 0.07, 0.07), 1.25, None, 8.9,
                 207.20000000000002),
    "hurricane": (1.11, (0.012, 0.12, 0.37, 0.38, 0.118), (0.45, 0.45, 0.25, 0.05, 0.05), 1.0, (14, 20, 1.4), 0.0,
                  254.57751479289942),
    "compound": (1.35, (0.015, 0.17, 0.44, 0.29, 0.085), (0.55, 0.55, 0.33, 0.08, 0.08), 1.5, None, 8.9, 270.0),
}
# (scenario, seed, kwargs for generate_arrivals) -> (patients, sha256 of repr(list))
GOLDEN_CASES = {
    "normal_7": ("normal", 7, {}),
    "heatwave_3": ("heatwave", 3, {}),
    "hurricane_4": ("hurricane", 4, {}),
    "compound_5": ("compound", 5, {}),
    "compound_5_noise": ("compound", 5, {"noise": 0.08}),
    "normal_9_window": ("normal", 9, {"t0": 240.0, "t1": 600.0, "base_pid": 1000000}),
}
GOLDEN_HASHES = {
    "normal_7": (187, "028cab360fb1abebdeff3a9fbcbd4f3297b82b9d41476db71914c85fe20eb96f"),
    "heatwave_3": (206, "c87f00148aab2b92222712eb908c435a555553e91f313d12e0eb716ffa8cd894"),
    "hurricane_4": (256, "f7f2834f309854a4dcdff1d348ccad0cfcc4aac0da4ff26c8c3e9eb2649b9713"),
    "compound_5": (289, "b3eb5a129a31aecbb060d3ebc969a575b52e735467e793e52d33c0280910a5a3"),
    "compound_5_noise": (228, "82a514074d7ef83f51711e5cac36a2d1e4d6734ff4863d53c97a4be4acf3330a"),
    "normal_9_window": (30, "d97da7fcea248575c2f642d9f5594632f6a973fc89cd4f520c2f5af11b4d8a33"),
}
_REF_GAMMA_SCALE = {1: 15.0, 2: 10.0, 3: 6.67, 4: 4.0, 5: 2.67}


def _ref_poisson(rng, lam):
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


def reference_generate_arrivals(arrival_cls, key, seed, t0=0.0, t1=1440.0, noise=0.0, base_pid=0, volume=200):
    """Frozen copy of the ORIGINAL generator, driven by the golden constants above."""
    vol_mult, acuity, admit, boarding, burst, _, _ = GOLDEN_SCENARIOS[key]

    def hour_mult(h):
        return burst[2] if burst and burst[0] <= h < burst[1] else 1.0

    rng, out, pid = random.Random(seed), [], base_pid
    for H in range(int(t0 // 60), int(math.ceil(t1 / 60))):
        f = math.exp(rng.gauss(0.0, noise)) if noise > 0 else 1.0
        a, b = max(t0, H * 60.0), min(t1, (H + 1) * 60.0)
        if b <= a:
            continue
        h = H % 24
        lam = volume * vol_mult * GOLDEN_HOURLY[h] * hour_mult(h) * f * (b - a) / 60.0
        for ts in sorted(rng.uniform(a, b) for _ in range(_ref_poisson(rng, lam))):
            ats = rng.choices(range(1, 6), acuity)[0]
            out.append(arrival_cls(pid, ts, ats, rng.uniform(2.0, 4.0),
                                   rng.gammavariate(3.0, _REF_GAMMA_SCALE[ats]),
                                   rng.random() < admit[ats - 1],
                                   rng.uniform(120.0, 240.0) * boarding))
            pid += 1
    return out


# ---------------------------------------------------------------------------------------
# Helpers: stub parameter files, private module copies, warning capture
# ---------------------------------------------------------------------------------------
STUB_MIX = [0.014985, 0.154190, 0.507351, 0.296424, 0.027050]
_RAW_HOURLY = [25, 21, 17, 16, 14, 15, 18, 26, 40, 52, 60, 61, 60, 60, 58, 57, 57, 58, 57, 56, 53, 46, 41, 31]
STUB_HOURLY = [round(x / sum(_RAW_HOURLY), 9) for x in _RAW_HOURLY]
STUB_WEEKDAY_SHARES = {"1": 0.125, "2": 0.169, "3": 0.150, "4": 0.140, "5": 0.135, "6": 0.146, "7": 0.135}
STUB_WEEKDAY_LABELS = {"1": "Sunday", "2": "Monday", "3": "Tuesday", "4": "Wednesday",
                       "5": "Thursday", "6": "Friday", "7": "Saturday"}


def _entry(value, status="verified", tag="DATA", **extra):
    entry = {"value": value, "tag": tag, "status": status, "weighting": "PATWT", "denominator": 1234,
             "source": "stub variable", "note": "test stub"}
    entry.update(extra)
    return entry


def stub_params(**override):
    data = {
        "schema_version": "1.0",
        "source": {"dataset": "stub", "file": "stub", "sha256": "0" * 64, "records": 1, "computed_on": "test"},
        "hourly_shape": _entry(STUB_HOURLY),
        "weekday_shares": _entry(STUB_WEEKDAY_SHARES, code_labels=STUB_WEEKDAY_LABELS),
        "monday_factor": _entry(round(STUB_WEEKDAY_SHARES["2"] * 7, 6)),
        "triage_mix": _entry(STUB_MIX),
        "notes_check": [],
        "context": {"tag": "CONTEXT"},
    }
    data.update(override)
    return data


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def capture_warnings():
    logger = logging.getLogger(LOGGER)
    handler, old_level = ListHandler(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


_copy_counter = [0]


def load_copy(path, params_path=None, legacy=False, module_path=None, patch_scenarios=None):
    """Execute a private copy of a source file under a fresh module name."""
    _copy_counter[0] += 1
    name = f"_private_copy_{_copy_counter[0]}"
    env = {k: v for k, v in os.environ.items() if k not in (ENV_PARAMS, ENV_LEGACY)}
    env[ENV_PARAMS] = str(params_path if params_path is not None else NOWHERE)
    if legacy:
        env[ENV_LEGACY] = "1"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        patches = {"src.simulation.scenarios": patch_scenarios} if patch_scenarios is not None else {}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules, patches):
            spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_scenarios(params_path=None, legacy=False):
    with capture_warnings():
        return load_copy(SCENARIOS_PATH, params_path, legacy)


def strip_pid(patients):
    return [tuple(p)[1:] for p in patients]


def hour_counts(patients):
    counts = [0] * 24
    for p in patients:
        counts[int(p.t // 60)] += 1
    return counts


TMP = None
NOWHERE = None
M = None   # clean private copy of scenarios (hand-typed placeholders, non-legacy default)
F = None   # private copy of forecaster bound to M


def setUpModule():
    global TMP, NOWHERE, M, F
    TMP = tempfile.TemporaryDirectory()
    NOWHERE = Path(TMP.name) / "does_not_exist.json"
    M = load_scenarios()
    F = load_copy(FORECASTER_PATH, patch_scenarios=M)


def tearDownModule():
    TMP.cleanup()


def write_json(name, data, raw_text=None):
    path = Path(TMP.name) / name
    path.write_text(raw_text if raw_text is not None else json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------
# Truth day
# ---------------------------------------------------------------------------------------
class TruthDayTests(unittest.TestCase):
    def test_same_seed_identical_and_different_seeds_differ(self):
        for name in ("normal", "heatwave", "hurricane", "compound"):
            a, _ = M.generate(name, 11)
            b, _ = M.generate(name, 11)
            self.assertEqual(a, b)
            self.assertNotEqual(a, M.generate(name, 12)[0])

    def test_truth_shape(self):
        truth, _ = M.generate("heatwave", 5)
        self.assertEqual([p.pid for p in truth], list(range(len(truth))))
        self.assertTrue(all(x.t <= y.t for x, y in zip(truth, truth[1:])))
        self.assertTrue(all(0.0 <= p.t < 1440.0 for p in truth))
        self.assertTrue(all(p.ats in (1, 2, 3, 4, 5) and isinstance(p.admit, bool) for p in truth))
        self.assertEqual(M.Arrival._fields, ("pid", "t", "ats", "tri", "consult", "admit", "hold"))

    def test_unknown_scenario_and_no_global_random(self):
        with self.assertRaises(ValueError):
            M.generate("earthquake", 1)
        state = random.getstate()
        M.generate("hurricane", 3, burst=True)
        self.assertEqual(state, random.getstate())     # the global random module is never touched

    def test_sampled_multiplier_is_inside_published_range_and_varies(self):
        for name in ("heatwave", "hurricane"):
            lo, hi = M.SURGE_RANGES[name]["low"], M.SURGE_RANGES[name]["high"]
            values = [M._sample_multiplier(name, M.truth_seed(i)) for i in range(50)]
            self.assertTrue(all(lo <= v <= hi for v in values))
            self.assertGreater(len(set(values)), 40)

    def test_mean_uplift_inside_published_range(self):
        seeds = [M.truth_seed(i) for i in range(50)]
        normal = sum(len(M.generate("normal", s)[0]) for s in seeds)
        for name in ("heatwave", "hurricane"):
            total = sum(len(M.generate(name, s)[0]) for s in seeds)
            ratio = total / normal
            self.assertGreaterEqual(ratio, M.SURGE_RANGES[name]["low"], name)
            self.assertLessEqual(ratio, M.SURGE_RANGES[name]["high"], name)

    def test_heatwave_extra_arrivals_fall_in_hours_12_to_19(self):
        extra_in, extra_out = 0, 0
        for i in range(50):
            seed = M.truth_seed(i)
            normal = hour_counts(M.generate("normal", seed)[0])
            heat = hour_counts(M.generate("heatwave", seed)[0])
            for h in range(24):
                if 12 <= h < 20:
                    extra_in += heat[h] - normal[h]
                else:
                    extra_out += heat[h] - normal[h]
        self.assertGreater(extra_in, 100)
        self.assertLessEqual(abs(extra_out), 0.05 * extra_in)   # "mostly" in 12-19 (here: exactly none)

    def test_hurricane_extra_arrivals_spread_over_the_whole_day(self):
        night, day = 0, 0
        for i in range(50):
            seed = M.truth_seed(i)
            normal = hour_counts(M.generate("normal", seed)[0])
            storm = hour_counts(M.generate("hurricane", seed)[0])
            night += sum(storm[h] - normal[h] for h in range(0, 12))
            day += sum(storm[h] - normal[h] for h in range(12, 24))
        self.assertGreater(night, 30)
        self.assertGreater(day, 30)

    def test_common_random_numbers_across_scenarios(self):
        # Same seed: the normal day's patients are a subset of the heatwave / hurricane day's.
        for seed in (3, 4, 5):
            normal = Counter(strip_pid(M.generate("normal", seed)[0]))
            for name in ("heatwave", "hurricane"):
                other = Counter(strip_pid(M.generate(name, seed)[0]))
                self.assertEqual(normal - other, Counter(), name)

    def test_shared_helper_keeps_truth_and_forecast_consistent(self):
        factors = M.hourly_multipliers("heatwave", 1.10)
        self.assertAlmostEqual(sum(M.HOURLY[h] * factors[h] for h in range(24)), 1.10, places=9)
        self.assertEqual([f for h, f in enumerate(factors) if not 12 <= h < 20], [1.0] * 16)
        self.assertEqual(M.hourly_multipliers("hurricane", 1.1)[5], M.hourly_multipliers("hurricane", 1.1)[20])
        self.assertEqual(M.hourly_multipliers("normal", 1.5), [1.0] * 24)
        for bad in ((0, 0), (5, 3), (-1, 4), (3, 25)):
            with self.assertRaises(ValueError):
                M.hourly_multipliers("heatwave", 1.1, window=bad)
        with self.assertRaises(ValueError):
            M.hourly_multipliers("heatwave", -1.0)

    def test_surge_ranges_are_tagged_evidence(self):
        self.assertEqual(set(M.SURGE_RANGES), {"heatwave", "hurricane"})
        self.assertEqual((M.SURGE_RANGES["heatwave"]["low"], M.SURGE_RANGES["heatwave"]["high"]), (1.06, 1.13))
        self.assertEqual((M.SURGE_RANGES["hurricane"]["low"], M.SURGE_RANGES["hurricane"]["high"]), (1.09, 1.13))
        for entry in M.SURGE_RANGES.values():
            self.assertEqual(entry["tag"], "EVIDENCE")
            self.assertEqual(entry["evidence_status"], "cited_by_team_not_independently_checked")


class BurstTests(unittest.TestCase):
    def test_burst_only_when_flagged_and_has_right_shape(self):
        hours_seen, sizes_seen = set(), set()
        for name in ("normal", "heatwave", "hurricane", "compound"):
            for seed in range(1, 9):
                plain = M.generate(name, seed)[0]
                self.assertEqual(plain, M.generate(name, seed, burst=False)[0])
                flagged = M.generate(name, seed, burst=True)[0]
                self.assertEqual(Counter(strip_pid(plain)) - Counter(strip_pid(flagged)), Counter())
                extra = list(Counter(strip_pid(flagged)) - Counter(strip_pid(plain)))
                self.assertTrue(10 <= len(extra) <= 20, (name, seed, len(extra)))
                self.assertTrue(all(row[1] in (1, 2) for row in extra))           # levels 1-2 only
                hours = {int(row[0] // 60) for row in extra}
                self.assertEqual(len(hours), 1)                                   # inside ONE hour
                self.assertTrue(12 <= min(hours) <= 21)
                hours_seen |= hours
                sizes_seen.add(len(extra))
        self.assertGreater(len(hours_seen), 2)        # the hour really is chosen at random
        self.assertGreater(len(sizes_seen), 2)

    def test_burst_does_not_disturb_the_other_streams(self):
        # Base patients are identical with and without the burst (separate burst stream).
        plain = M.generate("heatwave", 21)[0]
        flagged = M.generate("heatwave", 21, burst=True)[0]
        self.assertEqual(sum(p.ats <= 2 for p in flagged) - sum(p.ats <= 2 for p in plain),
                         len(flagged) - len(plain))


class InfoTests(unittest.TestCase):
    def test_info_is_a_pure_function_of_the_name(self):
        expected = {"normal": (False, "n/a", None, False), "heatwave": (True, "day_before", (12, 20), False),
                    "hurricane": (True, "day_before", (0, 24), True), "compound": (True, "day_before", (0, 24), True)}
        for name, (known, source, window, burst_possible) in expected.items():
            reference = {"scenario": name, "surge_known": known, "known_from": source,
                         "window": window, "burst_possible": burst_possible}
            for seed in (1, 2, 99_999, 123_456):
                for burst in (False, True):
                    info = M.generate(name, seed, burst=burst)[1]
                    self.assertEqual(info, reference)
                    self.assertEqual(set(info), INFO_KEYS)
            self.assertEqual(M.scenario_info(name), reference)

    def test_info_has_no_sampled_values(self):
        for name in ("heatwave", "hurricane"):
            info = M.generate(name, 7, burst=True)[1]
            self.assertFalse(any(isinstance(v, float) for v in info.values()))
            self.assertNotIn(M._sample_multiplier(name, 7), info.values())

    def test_info_is_a_fresh_dict(self):
        info = M.scenario_info("heatwave")
        info["surge_known"] = False
        self.assertTrue(M.scenario_info("heatwave")["surge_known"])


class SeedTests(unittest.TestCase):
    def test_truth_and_lookahead_never_overlap(self):
        self.assertEqual(real_scenarios.truth_seed(0), 100_000)
        self.assertEqual(real_scenarios.truth_seed(99_999), 199_999)
        self.assertEqual(real_scenarios.lookahead_seed(0), 200_000)
        self.assertEqual(real_scenarios.lookahead_seed(99_999), 299_999)
        truth = {real_scenarios.truth_seed(n) for n in range(0, 100_000, 7)}
        look = {real_scenarios.lookahead_seed(n) for n in range(0, 100_000, 7)}
        self.assertEqual(truth & look, set())
        self.assertLess(real_scenarios.truth_seed(99_999), real_scenarios.lookahead_seed(0))

    def test_out_of_range_raises(self):
        for fn in (real_scenarios.truth_seed, real_scenarios.lookahead_seed):
            for bad in (-1, 100_000, 10 ** 9, 1.5, "3", None, True):
                with self.assertRaises(ValueError, msg=repr(bad)):
                    fn(bad)


class ListAndCompoundTests(unittest.TestCase):
    def test_compound_hidden_but_callable(self):
        self.assertEqual(real_scenarios.list_scenarios(), ["normal", "heatwave", "hurricane"])
        self.assertIn("compound", real_scenarios.SCENARIOS)
        truth, info = M.generate("compound", 4)
        self.assertGreater(len(truth), 200)
        self.assertEqual(info["scenario"], "compound")

    def test_compound_is_all_assumption_and_cleaned(self):
        sc = M.SCENARIOS["compound"]
        self.assertTrue(sc.name.startswith("Compound Calamity"))
        source = SCENARIOS_PATH.read_text(encoding="utf-8")
        for word in ("Beryl", "2.2M", "105F"):
            self.assertNotIn(word, source)
        rows = [r for r in M.PARAMETER_TABLE if r["name"].startswith("SCENARIOS.compound.")]
        self.assertGreaterEqual(len(rows), 6)
        for row in rows:
            self.assertEqual((row["tag"], row["status"]), ("ASSUMPTION", "unverified"), row["name"])

    def test_heatwave_description_does_not_credit_nature_health_for_the_older_patient_figure(self):
        text = M.SCENARIOS["heatwave"].description
        sentences = [s for s in re.split(r"(?<=\.)\s+", text) if "8.9" in s]
        self.assertTrue(sentences)
        for sentence in sentences:
            self.assertIn("Houston", sentence)
            self.assertNotIn("Nature Health", sentence)
        self.assertEqual(M.SCENARIOS["heatwave"].elderly_surge_pct, 0.0)    # not modelled -> not claimed


class WeekdayTests(unittest.TestCase):
    def test_monday_is_one_and_default_is_monday(self):
        self.assertEqual(M.weekday_factor("monday"), 1.0)
        self.assertEqual(M.weekday_factor("Monday"), 1.0)
        self.assertEqual(M.generate("normal", 8)[0], M.generate("normal", 8, weekday="monday")[0])

    def test_other_days_need_verified_parameters(self):
        for day in ("saturday", "sunday"):
            with self.assertRaises(ValueError):
                M.weekday_factor(day)
        with self.assertRaises(ValueError):
            M.generate("normal", 8, weekday="saturday")
        with self.assertRaises(ValueError):
            M.weekday_factor("funday")

    def test_unverified_weekday_data_is_refused(self):
        path = write_json("wd_unverified.json", stub_params(
            weekday_shares=_entry(STUB_WEEKDAY_SHARES, status="unverified", code_labels=STUB_WEEKDAY_LABELS)))
        mod = load_scenarios(path)
        with self.assertRaises(ValueError):
            mod.weekday_factor("saturday")
        path = write_json("wd_null_monday.json", stub_params(monday_factor=_entry(None, status="unverified")))
        with self.assertRaises(ValueError):
            load_scenarios(path).weekday_factor("saturday")
        path = write_json("wd_no_labels.json", stub_params(weekday_shares=_entry(STUB_WEEKDAY_SHARES)))
        with self.assertRaises(ValueError):
            load_scenarios(path).weekday_factor("saturday")

    def test_verified_weekday_factor_changes_volume(self):
        mod = load_scenarios(write_json("wd_ok.json", stub_params()))
        self.assertAlmostEqual(mod.weekday_factor("saturday"), 0.135 / 0.169, places=9)
        self.assertEqual(mod.weekday_factor("monday"), 1.0)
        monday = sum(len(mod.generate("normal", s)[0]) for s in range(30))
        sunday = sum(len(mod.generate("normal", s, weekday="sunday")[0]) for s in range(30))
        self.assertAlmostEqual(sunday / monday, 0.125 / 0.169, delta=0.08)


# ---------------------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------------------
class ForecasterTests(unittest.TestCase):
    def test_deterministic_and_correct_length(self):
        info = M.scenario_info("heatwave")
        a, b = F.forecast(info, 6, 10), F.forecast(dict(info), 6, 10)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)
        self.assertEqual(F.forecast(info, 3, 0), [])

    def test_hour_index_wraps_around_24(self):
        info = M.scenario_info("normal")
        day = F.forecast(info, 0, 24)
        self.assertEqual(F.forecast(info, 22, 4), [day[22], day[23], day[0], day[1]])
        self.assertEqual(F.forecast(info, 0, 48), day + day)

    def test_sum_matches_expected_daily_volume(self):
        for name in ("normal", "heatwave", "hurricane"):
            total = sum(F.forecast(M.scenario_info(name), 0, 24))
            self.assertAlmostEqual(total, M.SCENARIOS[name].expected_daily(), places=6, msg=name)
        self.assertAlmostEqual(sum(F.forecast(M.scenario_info("normal"), 0, 24)), 200.0, places=6)
        self.assertAlmostEqual(sum(F.forecast(M.scenario_info("heatwave"), 0, 24)),
                               200.0 * M.SURGE_RANGES["heatwave"]["midpoint"], places=6)

    def test_surge_is_applied_only_inside_the_window(self):
        base = F.forecast(M.scenario_info("normal"), 0, 24)
        heat = F.forecast(M.scenario_info("heatwave"), 0, 24)
        for h in range(24):
            if 12 <= h < 20:
                self.assertGreater(heat[h], base[h])
            else:
                self.assertEqual(heat[h], base[h])

    def test_unknown_surge_means_baseline(self):
        info = M.scenario_info("heatwave")
        info["surge_known"] = False
        self.assertEqual(F.forecast(info, 0, 24), F.forecast(M.scenario_info("normal"), 0, 24))
        # compound has no published range: the forecaster does not invent one
        self.assertEqual(F.forecast(M.scenario_info("compound"), 0, 24), F.forecast(M.scenario_info("normal"), 0, 24))

    def test_forecast_ignores_burst_flag_and_extra_keys(self):
        info = M.scenario_info("hurricane")
        quiet = dict(info, burst_possible=False)
        loud = dict(info, burst_possible=True, sampled_multiplier=1.5, seed=3, truth=[1, 2, 3])
        self.assertEqual(F.forecast(quiet, 0, 24), F.forecast(loud, 0, 24))

    def test_truth_average_matches_forecast(self):
        # Averaged over many days, the truth follows the forecast (which uses the range midpoint).
        for name in ("heatwave", "hurricane"):
            total = [0.0] * 24
            for i in range(50):
                for h, c in enumerate(hour_counts(M.generate(name, M.truth_seed(i))[0])):
                    total[h] += c / 50
            forecast_total = sum(F.forecast(M.scenario_info(name), 0, 24))
            self.assertAlmostEqual(sum(total) / forecast_total, 1.0, delta=0.08, msg=name)

    def test_forecaster_cannot_see_truth(self):
        self.assertEqual(list(inspect.signature(F.forecast).parameters), ["info", "start_hour", "hours"])
        tree = ast.parse(FORECASTER_PATH.read_text(encoding="utf-8"))
        modules, names = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules.add(node.module)
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.Import):
                modules |= {a.name for a in node.names}
        self.assertEqual(modules, {"__future__", "src.simulation.scenarios"})
        self.assertLessEqual(names, {"annotations", "BASE_VOLUME", "HOURLY", "SURGE_RANGES", "hourly_multipliers"})

    def test_bad_input_is_rejected(self):
        info = M.scenario_info("normal")
        for args in ((None, 0, 4), ({"scenario": "normal"}, 0, 4), (info, 0.5, 4), (info, 0, 2.5), (info, 0, -1)):
            with self.assertRaises(ValueError):
                F.forecast(*args)


# ---------------------------------------------------------------------------------------
# patients_from_rates
# ---------------------------------------------------------------------------------------
class PatientsFromRatesTests(unittest.TestCase):
    def test_deterministic_and_seed_sensitive(self):
        rates = [8.0, 9.5, 12.0, 11.0]
        a = M.patients_from_rates(rates, M.lookahead_seed(1), start_hour=10, base_pid=10 ** 6)
        self.assertEqual(a, M.patients_from_rates(rates, M.lookahead_seed(1), start_hour=10, base_pid=10 ** 6))
        self.assertNotEqual(a, M.patients_from_rates(rates, M.lookahead_seed(2), start_hour=10, base_pid=10 ** 6))

    def test_times_pids_and_zero_rates(self):
        patients = M.patients_from_rates([0.0, 30.0, 0.0, 30.0], 5, start_hour=22, base_pid=10 ** 6)
        self.assertTrue(patients)
        self.assertEqual([p.pid for p in patients], list(range(10 ** 6, 10 ** 6 + len(patients))))
        self.assertTrue(all(x.t <= y.t for x, y in zip(patients, patients[1:])))
        hours = {int(p.t // 60) for p in patients}
        self.assertEqual(hours, {23, 25})                       # absolute hours; none in 22 or 24
        self.assertEqual(M.patients_from_rates([], 5), [])
        self.assertEqual(M.patients_from_rates([0.0, 0.0], 5), [])

    def test_counts_follow_the_rates(self):
        rates = [20.0, 5.0, 40.0]
        totals = [0, 0, 0]
        for seed in range(40):
            for p in M.patients_from_rates(rates, seed):
                totals[int(p.t // 60)] += 1
        for hour, rate in enumerate(rates):
            self.assertAlmostEqual(totals[hour] / 40, rate, delta=max(2.0, 0.2 * rate))

    def test_levels_follow_triage_mix(self):
        counts = Counter(p.ats for s in range(30) for p in M.patients_from_rates([60.0] * 6, s))
        total = sum(counts.values())
        for level, share in enumerate(M.BASE_ACUITY, start=1):
            self.assertAlmostEqual(counts[level] / total, share, delta=0.02)

    def test_bad_rates_rejected(self):
        for bad in ([-1.0], [float("nan")], [float("inf")], ["3"], [None]):
            with self.assertRaises(ValueError):
                M.patients_from_rates(bad, 1)
        with self.assertRaises(ValueError):
            M.patients_from_rates([1.0], 1, start_hour=-1)

    def test_forecast_feeds_straight_into_patients(self):
        rates = F.forecast(M.scenario_info("heatwave"), 12, 4)
        patients = M.patients_from_rates(rates, M.lookahead_seed(0), start_hour=12, base_pid=10 ** 6)
        self.assertTrue(all(12 * 60 <= p.t < 16 * 60 for p in patients))


# ---------------------------------------------------------------------------------------
# Legacy switch: the old numbers come back exactly
# ---------------------------------------------------------------------------------------
class LegacySwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = load_scenarios(legacy=True)
        cls.legacy_with_json = load_scenarios(write_json("legacy_stub.json", stub_params()), legacy=True)

    def _check_constants(self, mod):
        self.assertEqual(mod.HOURLY, GOLDEN_HOURLY)
        self.assertEqual(tuple(mod.BASE_ACUITY), GOLDEN_BASE_ACUITY)
        self.assertEqual(set(mod.SCENARIOS), set(GOLDEN_SCENARIOS))
        for key, (vol, acuity, admit, boarding, burst, elderly, daily) in GOLDEN_SCENARIOS.items():
            sc = mod.SCENARIOS[key]
            self.assertEqual((sc.volume_mult, sc.acuity, sc.admit, sc.boarding_mult, sc.burst,
                              sc.elderly_surge_pct), (vol, acuity, admit, boarding, burst, elderly), key)
            self.assertEqual(sc.expected_daily(), daily, key)

    def test_constants_equal_the_original(self):
        self._check_constants(self.legacy)
        self._check_constants(self.legacy_with_json)     # legacy wins even if a verified file exists

    def test_arrivals_equal_a_frozen_copy_of_the_original_generator(self):
        for mod in (self.legacy, self.legacy_with_json):
            for label, (key, seed, kwargs) in GOLDEN_CASES.items():
                got = mod.generate_arrivals(mod.SCENARIOS[key], seed, **kwargs)
                want = reference_generate_arrivals(mod.Arrival, key, seed, **kwargs)
                self.assertEqual(got, want, label)

    @unittest.skipUnless(sys.version_info[:2] == (3, 12), "golden hashes were captured with Python 3.12")
    @unittest.skipUnless(platform.machine().lower() in ("amd64", "x86_64"),
                         "golden hashes were captured on x86-64; other CPUs (e.g. s390x) can differ in the last digit "
                         "of libm results, which changes repr(). The frozen-reference test above checks exactness everywhere.")
    def test_arrivals_match_golden_hashes(self):
        for mod in (self.legacy, self.legacy_with_json):
            for label, (key, seed, kwargs) in GOLDEN_CASES.items():
                got = mod.generate_arrivals(mod.SCENARIOS[key], seed, **kwargs)
                digest = hashlib.sha256(repr(got).encode()).hexdigest()
                self.assertEqual((len(got), digest), GOLDEN_HASHES[label], label)

    def test_per_call_switch_matches_process_wide_switch(self):
        for name in ("normal", "heatwave", "hurricane", "compound"):
            self.assertEqual(M.generate(name, 9, use_legacy_params=True)[0], self.legacy.generate(name, 9)[0])
            self.assertEqual(M.generate(name, 9, use_legacy_params=False)[0], M.generate(name, 9)[0])
        self.assertNotEqual(M.generate("heatwave", 9, use_legacy_params=True)[0], M.generate("heatwave", 9)[0])

    def test_legacy_heatwave_uses_the_old_point_value(self):
        self.assertEqual(M._LEGACY_SCENARIOS["heatwave"].volume_mult, 1.036)
        self.assertEqual(M._LEGACY_SCENARIOS["heatwave"].acuity, (0.012, 0.14, 0.42, 0.34, 0.088))

    def test_non_legacy_has_no_hand_typed_shifts_for_the_three_listed_scenarios(self):
        base = M.SCENARIOS["normal"]
        for name in ("heatwave", "hurricane"):
            sc = M.SCENARIOS[name]
            self.assertEqual((sc.acuity, sc.admit, sc.boarding_mult, sc.burst, sc.volume_mult),
                             (base.acuity, base.admit, 1.0, None, 1.0))
        self.assertEqual(M.SCENARIOS["compound"].acuity, (0.015, 0.170, 0.440, 0.290, 0.085))

    def test_legacy_switch_logs_no_warning(self):
        with capture_warnings() as messages:
            load_copy(SCENARIOS_PATH, legacy=True)
        self.assertEqual(messages, [])


# ---------------------------------------------------------------------------------------
# Loading the real parameter file
# ---------------------------------------------------------------------------------------
class ParameterLoadingTests(unittest.TestCase):
    def test_missing_file_falls_back_and_warns_with_parameter_names(self):
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            mod = load_copy(SCENARIOS_PATH, params_path=NOWHERE)
        text = "\n".join(logs.output)
        self.assertIn("hourly_shape", text)
        self.assertIn("triage_mix", text)
        self.assertEqual(mod.HOURLY, GOLDEN_HOURLY)
        self.assertEqual(tuple(mod.BASE_ACUITY), GOLDEN_BASE_ACUITY)

    def test_unverified_entry_falls_back_and_warns(self):
        path = write_json("unverified.json", stub_params(hourly_shape=_entry(STUB_HOURLY, status="unverified")))
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            mod = load_copy(SCENARIOS_PATH, params_path=path)
        text = "\n".join(logs.output)
        self.assertIn("hourly_shape", text)
        self.assertNotIn("triage_mix", text)                    # that one is fine
        self.assertEqual(mod.HOURLY, GOLDEN_HOURLY)             # fell back
        self.assertEqual(list(mod.BASE_ACUITY), [round(x, 6) for x in STUB_MIX])   # loaded

    def test_bad_entries_fall_back(self):
        nan_list = STUB_MIX[:4] + [float("nan")]
        bad_files = {
            "short": stub_params(hourly_shape=_entry(STUB_HOURLY[:23])),
            "sum": stub_params(hourly_shape=_entry([x * 1.1 for x in STUB_HOURLY])),
            "negative": stub_params(hourly_shape=_entry([-0.01, 0.01] + STUB_HOURLY[2:])),
            "nan": stub_params(triage_mix=_entry(nan_list)),
            "wrong_tag": stub_params(triage_mix=_entry(STUB_MIX, tag="ASSUMPTION")),
            "missing": {"schema_version": "1.0"},
            "list_top": [1, 2, 3],
        }
        for label, data in bad_files.items():
            path = write_json(f"bad_{label}.json", data)
            with self.assertLogs(LOGGER, level="WARNING") as logs:
                mod = load_copy(SCENARIOS_PATH, params_path=path)
            self.assertTrue(any("hourly_shape" in m or "triage_mix" in m for m in logs.output), label)
            if label in ("short", "sum", "negative", "missing", "list_top"):
                self.assertEqual(mod.HOURLY, GOLDEN_HOURLY, label)
            if label in ("nan", "wrong_tag", "missing", "list_top"):
                self.assertEqual(tuple(mod.BASE_ACUITY), GOLDEN_BASE_ACUITY, label)

    def test_broken_json_falls_back(self):
        path = write_json("broken.json", None, raw_text="{this is not json")
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            mod = load_copy(SCENARIOS_PATH, params_path=path)
        self.assertIn("hourly_shape", "\n".join(logs.output))
        self.assertEqual(mod.HOURLY, GOLDEN_HOURLY)

    def test_verified_file_really_changes_the_values(self):
        path = write_json("verified.json", stub_params())
        with capture_warnings() as messages:
            mod = load_copy(SCENARIOS_PATH, params_path=path)
        self.assertEqual(messages, [])
        self.assertNotEqual(mod.HOURLY, GOLDEN_HOURLY)
        for got, want in zip(mod.HOURLY, STUB_HOURLY):
            self.assertAlmostEqual(got, want, places=7)
        self.assertAlmostEqual(sum(mod.HOURLY), 1.0, places=12)
        for got, want in zip(mod.BASE_ACUITY, STUB_MIX):
            self.assertAlmostEqual(got, want, places=7)
        self.assertAlmostEqual(sum(mod.BASE_ACUITY), 1.0, places=12)
        for name in ("normal", "heatwave", "hurricane"):         # the three listed scenarios share the loaded mix
            self.assertEqual(mod.SCENARIOS[name].acuity, mod.BASE_ACUITY)
        self.assertEqual(mod.SCENARIOS["compound"].acuity, (0.015, 0.170, 0.440, 0.290, 0.085))
        self.assertAlmostEqual(mod.SCENARIOS["normal"].expected_daily(), 200.0, places=6)
        # The truth day really follows the loaded hourly shape: the quiet 04:00 hour is quieter.
        counts = [sum(hour_counts(mod.generate("normal", s)[0])[h] for s in range(40)) for h in (4, 11)]
        self.assertLess(counts[0], 0.35 * counts[1])
        # ... and the old numbers are still one switch away.
        self.assertEqual(mod._LEGACY_HOURLY, GOLDEN_HOURLY)

    def test_load_real_parameters_function_and_env_override(self):
        path = write_json("direct.json", stub_params())
        info = real_scenarios.load_real_parameters(path, warn=False)
        self.assertTrue(info["found"])
        self.assertEqual(len(info["hourly"]), 24)
        self.assertEqual(len(info["acuity"]), 5)
        missing = real_scenarios.load_real_parameters(NOWHERE, warn=False)
        self.assertFalse(missing["found"])
        self.assertIsNone(missing["hourly"])
        with mock.patch.dict(os.environ, {ENV_PARAMS: str(path)}):
            self.assertEqual(real_scenarios.params_path(), path)
        self.assertEqual(real_scenarios._DEFAULT_PARAMS_PATH.name, "scenario_parameters.json")
        self.assertEqual(real_scenarios._DEFAULT_PARAMS_PATH.parent.name, "params")

    def test_verified_params_do_not_touch_surge_sizes(self):
        mod = load_scenarios(write_json("surge_check.json", stub_params()))
        self.assertEqual(mod.SURGE_RANGES["heatwave"]["low"], 1.06)
        self.assertEqual(mod.SURGE_RANGES["hurricane"]["high"], 1.13)


# ---------------------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------------------
class ParameterTableTests(unittest.TestCase):
    NOT_PARAMETERS = {"PARAMETER_TABLE", "SCENARIOS", "USE_LEGACY_PARAMS"}   # table itself, container, switch

    def test_every_parameter_constant_is_in_the_table(self):
        for mod in (M, real_scenarios):
            names = {r["name"] for r in mod.PARAMETER_TABLE}
            constants = [n for n in vars(mod) if re.fullmatch(r"[A-Z][A-Z0-9_]*", n) and n not in self.NOT_PARAMETERS]
            self.assertGreater(len(constants), 15)
            for name in constants:
                self.assertIn(name, names, f"{name} is missing from PARAMETER_TABLE")
            for needed in ("HOURLY", "BASE_VOLUME", "BASE_ACUITY", "BASE_ADMIT", "ATS_TARGET", "ATS_WEIGHT",
                           "GAMMA_SHAPE", "GAMMA_SCALE", "TURNAROUND", "LWBS_MIN", "CAPACITY", "SURGE_RANGES"):
                self.assertIn(needed, names)

    def test_every_scenario_value_is_in_the_table(self):
        names = {r["name"] for r in M.PARAMETER_TABLE}
        for key in M.SCENARIOS:
            for field in ("volume_mult", "acuity", "admit", "boarding_mult", "burst", "elderly_surge_pct"):
                self.assertIn(f"SCENARIOS.{key}.{field}", names)

    def test_row_format_and_tags(self):
        for table in (M.PARAMETER_TABLE, M.parameter_table(legacy=True)):
            for row in table:
                self.assertEqual(set(row), {"name", "value", "tag", "status", "note"}, row["name"])
                self.assertIn(row["tag"], ("DATA", "EVIDENCE", "ASSUMPTION"), row["name"])
                self.assertTrue(row["note"], row["name"])
                if row["tag"] == "DATA":
                    self.assertEqual(row["status"], "verified", row["name"])
                if row["tag"] == "ASSUMPTION":
                    self.assertEqual(row["status"], "unverified", row["name"])
                if row["status"] == "unverified":
                    self.assertEqual(row["tag"], "ASSUMPTION", row["name"])
                json.dumps(row["value"])        # values can be shown on the dashboard

    def test_nothing_is_data_unless_it_was_loaded(self):
        self.assertEqual([r["name"] for r in M.PARAMETER_TABLE if r["tag"] == "DATA"], [])
        self.assertEqual([r["name"] for r in M.parameter_table(legacy=True) if r["tag"] == "DATA"], [])
        loaded = load_scenarios(write_json("table_stub.json", stub_params()))
        data_rows = {r["name"]: r for r in loaded.PARAMETER_TABLE if r["tag"] == "DATA"}
        self.assertEqual(set(data_rows), {"HOURLY", "BASE_ACUITY", "SCENARIOS.normal.acuity",
                                          "SCENARIOS.heatwave.acuity", "SCENARIOS.hurricane.acuity"})
        self.assertEqual(data_rows["HOURLY"]["status"], "verified")
        # the legacy view of the same module never claims DATA
        self.assertEqual([r for r in loaded.parameter_table(legacy=True) if r["tag"] == "DATA"], [])

    def test_placeholders_say_placeholder(self):
        by_name = {r["name"]: r for r in M.PARAMETER_TABLE}
        for name in ("BASE_VOLUME", "BASE_ADMIT", "ATS_WEIGHT", "GAMMA_SHAPE", "GAMMA_SCALE", "TURNAROUND",
                     "LWBS_MIN", "CAPACITY", "HOURLY", "BASE_ACUITY"):
            self.assertIn("placeholder", by_name[name]["note"], name)
            self.assertEqual(by_name[name]["tag"], "ASSUMPTION")

    def test_surge_rows_are_evidence_and_unchecked(self):
        for row in M.PARAMETER_TABLE:
            if row["name"].startswith("SURGE_RANGES"):
                self.assertEqual(row["tag"], "EVIDENCE", row["name"])
                self.assertEqual(row["status"], "cited_by_team_not_independently_checked", row["name"])

    def test_required_triage_level_line(self):
        line = ("Patient data uses NHAMCS triage levels 1-5; ATS time targets "
                "(immediate/10/30/60/120 min) are applied to them as an ASSUMPTION.")
        rows = [r for r in M.PARAMETER_TABLE if r["value"] == line]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tag"], "ASSUMPTION")


# ---------------------------------------------------------------------------------------
# The existing simulator accepts the truth, and old callers still work
# ---------------------------------------------------------------------------------------
class SimulatorIntegrationTests(unittest.TestCase):
    def test_truth_runs_through_the_existing_simulator(self):
        for name in ("normal", "heatwave", "hurricane", "compound"):
            for burst in (False, True):
                truth, _ = M.generate(name, M.truth_seed(3), burst=burst)
                sim = EDSim()
                sim.add_arrivals(truth)
                sim.run_until(1440)
                self.assertEqual(sum(sim.cnt.values()), len(truth), name)
                self.assertEqual(sim.cnt["future"], 0)
                self.assertGreater(sim.cnt["done"] + sim.cnt["lwbs"] + sim.cnt["board"], 0)

    def test_real_modules_smoke(self):
        for name in real_scenarios.list_scenarios():
            truth, info = real_scenarios.generate(name, real_scenarios.truth_seed(1), use_legacy_params=False)
            self.assertTrue(truth)
            rates = real_forecaster.forecast(info, 0, 24)
            self.assertEqual(len(rates), 24)
            self.assertTrue(real_scenarios.patients_from_rates(rates, real_scenarios.lookahead_seed(1),
                                                               base_pid=10 ** 6))

    def test_old_names_and_meanings_survive(self):
        s = real_scenarios
        self.assertAlmostEqual(sum(s.HOURLY), 1.0)
        self.assertEqual(len(s.HOURLY), 24)
        self.assertEqual(s.BASE_VOLUME, 200)
        self.assertAlmostEqual(sum(s.BASE_ACUITY), 1.0)
        self.assertEqual(set(s.SCENARIOS), {"normal", "heatwave", "hurricane", "compound"})
        self.assertEqual(s.Arrival._fields, ("pid", "t", "ats", "tri", "consult", "admit", "hold"))
        self.assertAlmostEqual(s.SCENARIOS["normal"].expected_daily(), 200.0)
        v = {k: sc.expected_daily() for k, sc in s.SCENARIOS.items()}
        self.assertGreater(v["heatwave"], v["normal"])
        self.assertGreater(v["hurricane"], v["heatwave"])
        self.assertGreater(v["compound"], v["hurricane"])
        for sc in s.SCENARIOS.values():
            self.assertAlmostEqual(sum(sc.acuity), 1.0)
        a = s.generate_arrivals(s.SCENARIOS["normal"], 7)
        self.assertEqual(a, s.generate_arrivals(s.SCENARIOS["normal"], 7))
        self.assertTrue(150 < len(a) < 260)
        self.assertEqual(s.ATS_TARGET, {1: 0.0, 2: 10.0, 3: 30.0, 4: 60.0, 5: 120.0})


if __name__ == "__main__":
    unittest.main()
