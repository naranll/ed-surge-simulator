"""Tests for the version 1 plan search in src/optimization/optimizer.py.

These tests use a small STUB of the scenario API (generate, forecast, truth_seed,
lookahead_seed, patients_from_rates) so they do not depend on the real scenarios.py /
forecaster.py or on any data file. In the stub the TRUE surge differs from the
FORECAST surge, so the tests mean something. Run:
    py -3.12 -B -m unittest tests.test_optimizer_v1
"""
import contextlib
import inspect
import io
import json
import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.optimization import optimizer as opt
from src.optimization.optimizer import (DO_NOTHING, FORECAST_PID, Plan, RunMetrics, Settings, boarding_cut_shim,
                                        compute_metrics, confidence_range, evaluate_on_truth, fairness_ok,
                                        main, plan_grid, plan_search, rolling_plan, run_batches, run_plan,
                                        score_run, select_plan, stability_penalty, t_critical)
from src.simulation import ed_simulator
from src.simulation.ed_simulator import Action, EDSim
from src.simulation.scenarios import Arrival

# ---------------------------------------------------------------------------
# The stub scenario API (same signatures as interface notes [2] and [3])
# ---------------------------------------------------------------------------
_RAW_HOURLY = [0.024, 0.017, 0.013, 0.010, 0.009, 0.010, 0.015, 0.025, 0.038, 0.052, 0.061, 0.065,
               0.064, 0.063, 0.063, 0.063, 0.062, 0.061, 0.062, 0.061, 0.057, 0.048, 0.041, 0.030]
STUB_HOURLY = [x / sum(_RAW_HOURLY) for x in _RAW_HOURLY]
STUB_ACUITY = (0.012, 0.131, 0.384, 0.368, 0.105)
STUB_ADMIT = (0.45, 0.45, 0.25, 0.05, 0.05)
STUB_SCALE = {1: 15.0, 2: 10.0, 3: 6.67, 4: 4.0, 5: 2.67}

# forecast_* is what the hospital expects; truth_* is what really happens.
STUB_SCENARIOS = {
    # quiet day: nothing helps, the optimiser must stay quiet
    "calm": dict(forecast_volume=50, truth_volume=50, window=None, forecast_mult=1.0, truth_mult=(1.0, 1.0)),
    "normal": dict(forecast_volume=200, truth_volume=200, window=None, forecast_mult=1.0, truth_mult=(1.0, 1.0)),
    # big surge, known in advance; the real surge is BIGGER than the forecast midpoint
    "heatwave": dict(forecast_volume=200, truth_volume=200, window=(10, 20), forecast_mult=1.25,
                     truth_mult=(1.3, 1.45)),
    # forecast says "big surge" but the real day is quiet: plans look good, then fail on the truth
    "false_alarm": dict(forecast_volume=200, truth_volume=60, window=(10, 20), forecast_mult=1.6,
                        truth_mult=(1.0, 1.0)),
    "hurricane": dict(forecast_volume=200, truth_volume=200, window=(0, 24), forecast_mult=1.2,
                      truth_mult=(1.3, 1.5), burst_possible=True),
}


def _poisson(rng, lam):
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


class StubAPI:
    """Fake scenario API. Records every call so tests can spy on it."""

    def __init__(self, overlap_seeds=False):
        self.calls = {"generate": [], "forecast": [], "patients_from_rates": []}
        self.overlap_seeds = overlap_seeds

    # ---- seeds ----
    def truth_seed(self, n):
        if not 0 <= n < 100_000:
            raise ValueError("n out of range")
        return 100_000 + n

    def lookahead_seed(self, n):
        if not 0 <= n < 100_000:
            raise ValueError("n out of range")
        return (100_000 if self.overlap_seeds else 200_000) + n

    def list_scenarios(self):
        return ["normal", "heatwave", "hurricane"]

    # ---- patients ----
    def _patients(self, rates, seed, start_hour, base_pid):
        timing, acuity = random.Random(f"{seed}:timing"), random.Random(f"{seed}:acuity")
        care = random.Random(f"{seed}:care")
        out, pid = [], base_pid
        for k, lam in enumerate(rates):
            hour = start_hour + k
            times = sorted(60.0 * hour + 60.0 * timing.random() for _ in range(_poisson(timing, lam)))
            for t in times:
                ats = acuity.choices(range(1, 6), STUB_ACUITY)[0]
                out.append(Arrival(pid, t, ats, care.uniform(2.0, 4.0), care.gammavariate(3.0, STUB_SCALE[ats]),
                                   care.random() < STUB_ADMIT[ats - 1], care.uniform(120.0, 240.0)))
                pid += 1
        return out

    def patients_from_rates(self, rates, seed, start_hour=0, base_pid=0):
        self.calls["patients_from_rates"].append({"seed": seed, "start_hour": start_hour, "base_pid": base_pid})
        return self._patients(rates, seed, start_hour, base_pid)

    # ---- the true day ----
    def info_for(self, name):
        cfg = STUB_SCENARIOS[name]
        return {"scenario": name, "surge_known": cfg["window"] is not None,
                "known_from": "day_before" if cfg["window"] else "n/a", "window": cfg["window"],
                "burst_possible": cfg.get("burst_possible", False)}

    def generate(self, name, seed, burst=False, use_legacy_params=None, weekday="monday"):
        self.calls["generate"].append({"name": name, "seed": seed, "burst": burst})
        cfg = STUB_SCENARIOS[name]
        lo, hi = cfg["truth_mult"]
        mult = random.Random(f"{seed}:volume").uniform(lo, hi)
        rates = [cfg["truth_volume"] * STUB_HOURLY[h] * (mult if cfg["window"] and cfg["window"][0] <= h < cfg["window"][1] else 1.0)
                 for h in range(24)]
        truth = self._patients(rates, seed, 0, 0)
        if burst:
            hour = random.Random(f"{seed}:burst").randrange(12, 22)
            rng = random.Random(f"{seed}:burstcare")
            extra = [Arrival(0, 60.0 * hour + 5.0 * i, 2, 3.0, rng.gammavariate(3.0, 10.0), True, 180.0)
                     for i in range(12)]
            truth = sorted(truth + extra, key=lambda a: a.t)
            truth = [a._replace(pid=i) for i, a in enumerate(truth)]
        return truth, self.info_for(name)

    # ---- the forecast: built ONLY from info (never from truth, seeds or burst) ----
    def forecast(self, info, start_hour, hours):
        self.calls["forecast"].append({"info": dict(info), "start_hour": start_hour, "hours": hours})
        cfg = STUB_SCENARIOS[info["scenario"]]
        out = []
        for k in range(hours):
            h = (start_hour + k) % 24
            surge = info["surge_known"] and info["window"][0] <= h < info["window"][1]
            out.append(cfg["forecast_volume"] * STUB_HOURLY[h] * (cfg["forecast_mult"] if surge else 1.0))
        return out




def stub_search(name, n_seeds=5, quick=True, **kw):
    api = StubAPI()
    return api, plan_search(name, n_seeds=n_seeds, quick=quick, api=api, **kw)


def fake_metrics(on_time=70, wait=10.0, lwbs=0):
    return RunMetrics(100, on_time, 50, wait, lwbs)


class LiveSpy(opt.TruthLive):
    """A live ED that remembers the biggest number of clinicians it ever had."""

    def __init__(self, truth):
        super().__init__(truth)
        self.max_docs = self.sim.cap["d"]

    def advance_to(self, t):
        super().advance_to(t)
        self.max_docs = max(self.max_docs, self.sim.cap["d"])


# ---------------------------------------------------------------------------
class SelectionUsesOnlyTheForecast(unittest.TestCase):
    def test_select_plan_never_calls_generate(self):
        api = StubAPI()
        info = api.info_for("heatwave")
        api.calls["generate"].clear()
        sel = select_plan(info, 3, plan_grid(quick=True), api)
        self.assertEqual(api.calls["generate"], [])                     # the spy saw ZERO generate calls
        self.assertEqual([c["info"] for c in api.calls["forecast"]], [info])
        self.assertEqual(len(api.calls["patients_from_rates"]), 3)
        self.assertIn(sel.best.plan, plan_grid(quick=True))

    def test_select_plan_signature_has_no_truth(self):
        for fn in (select_plan, rolling_plan):
            names = list(inspect.signature(fn).parameters)
            self.assertFalse([n for n in names if "truth" in n.lower()], names)
            self.assertNotIn("true_day", names)

    def test_seed_streams_never_overlap_and_come_from_the_api(self):
        api, res = stub_search("heatwave")
        look = [c["seed"] for c in api.calls["patients_from_rates"]]
        truth = [c["seed"] for c in api.calls["generate"]]
        self.assertEqual(look, [api.lookahead_seed(i) for i in range(5)])
        self.assertEqual(sorted(set(truth)), [api.truth_seed(i) for i in range(5)])   # info call re-uses seed 0
        self.assertTrue(set(look).isdisjoint(truth))
        self.assertTrue(all(c["start_hour"] == 0 and c["base_pid"] >= FORECAST_PID
                            for c in api.calls["patients_from_rates"]))
        self.assertEqual(res.n_seeds, 5)

    def test_overlapping_seed_streams_are_refused(self):
        with self.assertRaises(ValueError):
            plan_search("normal", n_seeds=3, quick=True, api=StubAPI(overlap_seeds=True))

    def test_truth_is_used_only_for_the_chosen_plan_and_do_nothing(self):
        seen = []

        def runner(plan, patients):
            seen.append((plan, patients[0].pid >= FORECAST_PID))
            return run_plan(plan, patients)

        res = plan_search("heatwave", quick=True, api=StubAPI(), runner=runner)
        truth_plans = {p for p, is_lookahead in seen if not is_lookahead}
        self.assertEqual(truth_plans, {DO_NOTHING, res.selected})
        look_plans = {p for p, is_lookahead in seen if is_lookahead}
        self.assertEqual(look_plans, set(plan_grid(quick=True)))
        self.assertEqual(sum(1 for _p, la in seen if la), 29 * 5)

    def test_truth_runs_are_paired_on_the_same_day(self):
        ids = {}

        def runner(plan, patients):
            ids.setdefault(id(patients), []).append(plan)
            return fake_metrics()

        evaluate_on_truth(Plan(720.0, 1), "normal", 4, StubAPI(), runner=runner)
        self.assertEqual(len(ids), 4)                                   # 4 truth days
        self.assertTrue(all(v == [DO_NOTHING, Plan(720.0, 1)] for v in ids.values()))

    def test_burst_goes_to_the_true_days_only(self):
        api, res = stub_search("hurricane", burst=True)
        calls = api.calls["generate"]
        self.assertFalse(calls[0]["burst"])                             # the info call
        self.assertTrue(all(c["burst"] for c in calls[1:]))
        self.assertIn("burst", res.caution)
        _api, calm = stub_search("normal")
        self.assertEqual(calm.caution, "")


class Recommendation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.heat = stub_search("heatwave")[1]
        cls.calm = stub_search("calm")[1]
        cls.false_alarm = stub_search("false_alarm")[1]

    def test_clear_gain_recommends_a_plan(self):
        r = self.heat
        self.assertFalse(r.recommended.is_nothing)
        self.assertEqual(r.recommended, r.selected)
        self.assertGreater(r.truth["gain_lo"], 0.0)                     # the whole 95% range is above zero
        self.assertTrue(r.truth["fairness_ok"])
        self.assertGreater(r.truth["plan_on_time_pct"], r.truth["base_on_time_pct"])
        row = r.recommended_row
        self.assertTrue(row["recommended"] and row["selected"])
        self.assertGreater(row["truth_gain_lo"], 0.0)
        self.assertEqual(sum(1 for x in r.rows if x["recommended"]), 1)

    def test_truth_differs_from_forecast_in_the_stub(self):
        api = StubAPI()
        info = api.info_for("heatwave")
        forecast_total = sum(api.forecast(info, 0, 24))
        sizes = [len(api.generate("heatwave", api.truth_seed(i))[0]) for i in range(8)]
        self.assertGreater(sum(sizes) / 8, 1.05 * forecast_total)      # the real surge is bigger than the forecast

    def test_nothing_helps_so_do_nothing(self):
        r = self.calm
        self.assertTrue(r.selected.is_nothing and r.recommended.is_nothing)
        self.assertIn("worth its cost", r.reason)
        self.assertTrue(r.recommended_row["recommended"])

    def test_range_that_does_not_clear_zero_gives_do_nothing(self):
        r = self.false_alarm        # forecast says big surge, the real day is quiet
        self.assertFalse(r.selected.is_nothing)                         # the look-ahead was fooled ...
        self.assertTrue(r.recommended.is_nothing)                       # ... the truth check caught it
        self.assertLessEqual(r.truth["gain_lo"], 0.0)
        self.assertIn("95% range", r.reason)
        picked = [x for x in r.rows if x["selected"]][0]
        self.assertFalse(picked["recommended"])
        self.assertLess(picked["truth_gain"], 0.0)

    def test_every_plan_has_a_row_and_one_pick(self):
        for r in (self.heat, self.calm, self.false_alarm):
            self.assertEqual(len(r.rows), len(plan_grid(quick=True)))
            self.assertEqual(sum(1 for x in r.rows if x["selected"]), 1)
            self.assertEqual(len({x["plan"] for x in r.rows}), len(r.rows))

    def test_quick_uses_five_seeds_and_the_coarse_grid(self):
        api = StubAPI()
        r = plan_search("normal", n_seeds=50, quick=True, api=api)      # quick overrides n_seeds
        self.assertEqual((r.n_seeds, len(r.rows)), (5, 29))
        self.assertEqual(len(api.calls["patients_from_rates"]), 5)
        full = plan_search("normal", n_seeds=2, api=StubAPI())
        self.assertEqual(len(full.rows), 78)


class FairnessAndScore(unittest.TestCase):
    def test_fairness_function(self):
        s = Settings(fairness_limit=0.15, fairness_slack_min=1.0)
        self.assertTrue(fairness_ok(11.5, 10.0, Settings(0.15, 0.0)))
        self.assertFalse(fairness_ok(11.6, 10.0, Settings(0.15, 0.0)))
        self.assertTrue(fairness_ok(9.0, 10.0, s))                      # faster is always fine
        self.assertFalse(fairness_ok(40.0, 10.0, s))

    def test_fairness_limit_blocks_a_plan_that_hurts_non_urgent_patients(self):
        def runner(plan, patients):
            if plan.is_nothing:
                return fake_metrics(70, 10.0)
            if plan.boarding_cut == 0.5:                                # great for urgent, bad for non-urgent
                return fake_metrics(95, 40.0)
            return fake_metrics(70 + 8 * plan.clinician + 2 * plan.fast_track, 10.0)

        api = StubAPI()
        info = api.info_for("normal")
        sel = select_plan(info, 3, plan_grid(quick=True), api, runner=runner)
        blocked = [s for s in sel.stats if s.plan.boarding_cut == 0.5]
        self.assertTrue(blocked and not any(s.feasible for s in blocked))
        self.assertTrue(all(s.nonurgent_change_pct > 100 for s in blocked))
        self.assertEqual(sel.best.plan.boarding_cut, 0.0)               # the best FAIR plan wins ...
        self.assertEqual((sel.best.plan.clinician, sel.best.plan.fast_track), (1, True))
        # ... and with the limit lifted the 50% plan would win, so fairness was the reason
        lifted = select_plan(info, 3, plan_grid(quick=True), api, settings=Settings(fairness_limit=100.0),
                             runner=runner)
        self.assertEqual(lifted.best.plan.boarding_cut, 0.5)

    def test_fairness_is_checked_again_on_the_truth_days(self):
        def runner(plan, patients):                                     # fair in the forecast, unfair on truth
            truth_day = patients[0].pid < FORECAST_PID
            if plan.is_nothing:
                return fake_metrics(70, 10.0)
            return fake_metrics(95, 40.0 if truth_day else 10.0)

        res = plan_search("normal", quick=True, api=StubAPI(), runner=runner)
        self.assertFalse(res.selected.is_nothing)
        self.assertTrue(res.recommended.is_nothing)
        self.assertFalse(res.truth["fairness_ok"])
        self.assertGreater(res.truth["gain_lo"], 0.0)                   # the gain itself looked fine
        self.assertIn("fairness", res.reason)

    def test_lwbs_counts_against_the_score(self):
        a, b = fake_metrics(80, 10.0, lwbs=0), fake_metrics(80, 10.0, lwbs=6)
        self.assertAlmostEqual(score_run(a, 0.0) - score_run(b, 0.0), 6 * opt.LWBS_PENALTY_PP)
        self.assertGreater(opt.LWBS_PENALTY_PP, 0.0)
        heavy = Settings(lwbs_penalty_pp=10.0)
        self.assertAlmostEqual(score_run(a, 0.0, heavy) - score_run(b, 0.0, heavy), 60.0)

        def runner(plan, patients):                                     # same on-time share, fewer LWBS
            return fake_metrics(80, 10.0, lwbs=0 if plan.fast_track else 6)

        api = StubAPI()
        sel = select_plan(api.info_for("normal"), 2, plan_grid(quick=True), api, runner=runner)
        self.assertTrue(sel.best.plan.fast_track)
        self.assertEqual(sel.best.plan.clinician, 0)                    # fast track is the cheapest fix

    def test_cost_counts_against_the_score(self):
        m = fake_metrics(80)
        self.assertAlmostEqual(score_run(m, 0.0) - score_run(m, 1000.0), opt.COST_PENALTY_PP_PER_1000_USD)

    def test_compute_metrics_matches_the_simulator_and_counts_lwbs(self):
        sim = EDSim(nurses=1, docs=1, beds=2)
        sim.add_arrivals([Arrival(i, float(i), 1 + i % 5, 2.0, 40.0, False, 0.0) for i in range(14)])
        sim.run_until(600.0)
        r, m = sim.result(0.0), compute_metrics(sim, 0.0, 600.0)
        self.assertGreater(m.lwbs, 0)
        self.assertEqual(m.lwbs, r.lwbs)
        self.assertEqual(m.urgent_n, sum(r.totals[i] for i in (1, 2, 3)))
        self.assertEqual(m.urgent_n - m.urgent_on_time, sum(r.breaches[i] for i in (1, 2, 3)))
        self.assertEqual(m.nonurgent_n, r.totals[4] + r.totals[5])
        waits = sorted((p.t_doc if p.t_doc is not None else p.t_lwbs if p.t_lwbs is not None else 600.0) - p.t
                       for p in sim.pat.values() if p.ats >= 4)
        mid = len(waits) // 2
        median = waits[mid] if len(waits) % 2 else (waits[mid - 1] + waits[mid]) / 2
        self.assertAlmostEqual(m.nonurgent_median_wait, median)


class ShimTests(unittest.TestCase):
    def test_shim_sets_and_restores_the_factor(self):
        before = ed_simulator.EXPEDITE_FACTOR
        with boarding_cut_shim(0.25):
            self.assertAlmostEqual(ed_simulator.EXPEDITE_FACTOR, 0.75)
            with boarding_cut_shim(0.5):
                self.assertAlmostEqual(ed_simulator.EXPEDITE_FACTOR, 0.5)
            self.assertAlmostEqual(ed_simulator.EXPEDITE_FACTOR, 0.75)
        self.assertEqual(ed_simulator.EXPEDITE_FACTOR, before)
        with boarding_cut_shim(0.0):
            self.assertEqual(ed_simulator.EXPEDITE_FACTOR, before)

    def test_shim_restores_the_factor_even_after_an_exception(self):
        before = ed_simulator.EXPEDITE_FACTOR
        with self.assertRaises(RuntimeError):
            with boarding_cut_shim(0.5):
                raise RuntimeError("boom")
        self.assertEqual(ed_simulator.EXPEDITE_FACTOR, before)
        with self.assertRaises(ValueError):
            with boarding_cut_shim(1.5):
                pass
        self.assertEqual(ed_simulator.EXPEDITE_FACTOR, before)

    def test_shim_really_changes_the_boarding_time(self):
        for cut, factor in ((0.25, 0.75), (0.5, 0.5)):
            sim = EDSim()
            sim.add_arrivals([Arrival(0, 0.0, 2, 2.0, 10.0, True, 200.0)])
            with boarding_cut_shim(cut):
                sim.apply(Action(expedite=True, duration=360.0))
                sim.run_until(400.0)
            self.assertAlmostEqual(sim.pat[0].t_dep, 2.0 + 10.0 + 200.0 * factor, places=6)
        self.assertEqual(ed_simulator.EXPEDITE_FACTOR, 0.7)

    def test_run_plan_leaves_the_factor_alone(self):
        before = ed_simulator.EXPEDITE_FACTOR
        patients = StubAPI().generate("normal", 100_001)[0]
        run_plan(Plan(720.0, 1, True, 0.5), patients)
        self.assertEqual(ed_simulator.EXPEDITE_FACTOR, before)

    def test_boarding_cut_helps_when_the_ed_is_congested(self):
        api = StubAPI()
        truth = api.generate("heatwave", api.truth_seed(1))[0]
        base, cut = run_plan(DO_NOTHING, truth), run_plan(Plan(720.0, 0, False, 0.5), truth)
        self.assertGreaterEqual(cut.urgent_share, base.urgent_share)
        self.assertLessEqual(cut.lwbs, base.lwbs)


class PlansAndStatistics(unittest.TestCase):
    def test_plan_grid(self):
        full, quick = plan_grid(), plan_grid(quick=True)
        self.assertEqual((len(full), len(quick)), (78, 29))
        self.assertEqual(full[0], DO_NOTHING)
        self.assertEqual(len(set(full)), len(full))
        self.assertTrue(all(p.start + p.duration <= 1440.0 for p in full))
        self.assertEqual(sorted({p.start / 60 for p in full if not p.is_nothing}), [0, 3, 6, 9, 12, 15, 18])
        self.assertEqual(sorted({p.start / 60 for p in quick if not p.is_nothing}), [0, 6, 12, 18])
        self.assertEqual(sum(1 for p in full if p.is_nothing), 1)

    def test_plan_costs_and_labels(self):
        self.assertEqual(Plan(720.0, 1).cost, 900.0)                    # 6 h x 150
        self.assertEqual(Plan(720.0, 1).staff_hours, 6.0)
        self.assertEqual(Plan(720.0, 0, True).cost, 150.0)
        self.assertEqual(Plan(720.0, 0, False, 0.5).cost, 500.0)
        self.assertEqual(Plan(720.0, 0, False, 0.25).cost, 250.0)
        self.assertEqual(Plan(720.0, 1, True, 0.5).cost, 1550.0)
        self.assertEqual(Plan(0.0, 1, True, 0.5, duration=240.0).cost, 600.0 + 150.0 + 500.0)
        self.assertEqual(Plan(999.0).label, "Do nothing")
        self.assertEqual(Plan(999.0), DO_NOTHING)
        self.assertEqual(Plan(720.0, 1, True, 0.5).label, "12:00-18:00 +1 clinician, fast track, boarding -50%")
        a = Plan(720.0, 1, False, 0.25).action()
        self.assertEqual((a.doc, a.fast_track, a.expedite, a.duration), (1, False, True, 360.0))
        with self.assertRaises(ValueError):
            Plan(0.0, 0, False, 1.0)

    def test_confidence_range(self):
        m, lo, hi = confidence_range([1.0, 2.0, 3.0])
        self.assertAlmostEqual(m, 2.0)
        self.assertAlmostEqual(hi - m, 4.303 / math.sqrt(3), places=6)
        self.assertAlmostEqual(m - lo, hi - m)
        self.assertEqual(confidence_range([5.0]), (5.0, None, None))
        self.assertEqual(confidence_range([2.0, 2.0, 2.0, 2.0]), (2.0, 2.0, 2.0))
        self.assertAlmostEqual(t_critical(1), 12.706)
        self.assertEqual(t_critical(1000), 1.96)
        self.assertTrue(all(t_critical(d) >= t_critical(d + 1) for d in range(1, 200)))
        with self.assertRaises(ValueError):
            confidence_range([])

    def test_parameter_table_tags(self):
        names = [p["name"] for p in opt.OPTIMISER_PARAMETERS]
        self.assertEqual(len(names), len(set(names)))
        for p in opt.OPTIMISER_PARAMETERS:
            self.assertEqual((p["tag"], p["status"]), ("ASSUMPTION", "unverified"), p["name"])
            self.assertTrue(p["note"])
        for needed in ("START_HOURS", "FAIRNESS_LIMIT", "CLINICIAN_COST_PER_HOUR", "LWBS_PENALTY_PP",
                       "STABILITY_PENALTY_PP", "DEFAULT_SEEDS", "COST_PENALTY_PP_PER_1000_USD"):
            self.assertIn(needed, names)
        ats = next(p for p in opt.OPTIMISER_PARAMETERS if p["name"] == "ATS_TARGET")
        self.assertIn("NHAMCS", ats["note"])

    def test_existing_public_names_are_still_importable(self):
        from src.optimization.optimizer import (Recommendation as OldRecommendation, SEEDS, candidate_actions,
                                                clinical_loss, pretty, recommend, rolling, summarize, total_loss)
        self.assertEqual(SEEDS, (11, 12, 13, 14, 15))
        self.assertEqual(len(candidate_actions()), 10)
        self.assertEqual(pretty("DO_NOTHING"), "Hold Current Staffing")
        for fn in (clinical_loss, recommend, rolling, summarize, total_loss):
            self.assertTrue(callable(fn))
        self.assertTrue(hasattr(OldRecommendation, "headline"))


class Determinism(unittest.TestCase):
    def test_same_inputs_give_identical_output(self):
        one = plan_search("heatwave", quick=True, api=StubAPI()).to_dict()
        two = plan_search("heatwave", quick=True, api=StubAPI()).to_dict()
        self.assertEqual(one, two)
        self.assertEqual(json.dumps(one, allow_nan=False), json.dumps(two, allow_nan=False))

    def test_workers_give_the_same_answer_as_serial(self):
        api = StubAPI()
        jobs = [(api.generate("heatwave", api.truth_seed(i))[0], [DO_NOTHING, Plan(720.0, 1, True, 0.5)])
                for i in range(3)]
        self.assertEqual(run_batches(jobs, workers=2), run_batches(jobs, workers=1))


class RollingLoop(unittest.TestCase):
    def setUp(self):
        self.api = StubAPI()

    def _live(self, name):
        truth, info = self.api.generate(name, self.api.truth_seed(0))
        return LiveSpy(truth), info

    def test_rolling_guard_prevents_stacking(self):
        live, info = self._live("heatwave")
        base_docs = live.sim.cap["d"]
        # force the "no false alarm" gate open, so only the stacking guard can stop repeated actions
        with mock.patch.object(opt, "confidence_range", lambda xs: (5.0, 1.0, 9.0)):
            rows = rolling_plan(live, info, self.api, start=9 * 60, until=16 * 60, n_seeds=2)
        applied = [r["t"] for r in rows if r["applied"]]
        self.assertGreaterEqual(len(applied), 1)
        self.assertTrue(all(b - a >= opt.ROLLING_WINDOW_MIN for a, b in zip(applied, applied[1:])), applied)
        self.assertLessEqual(len(applied), 2)                           # 7 hours, 4-hour windows
        held = [r for r in rows if not r["evaluated"]]
        self.assertTrue(held and all(r["action"] == "hold" for r in held))
        self.assertTrue(applied[0] < held[0]["t"] < applied[0] + opt.ROLLING_WINDOW_MIN)
        self.assertLessEqual(live.max_docs, base_docs + 1)              # never two extra clinicians at once

    def test_rolling_acts_once_on_a_real_surge_and_stays_quiet_when_calm(self):
        live, info = self._live("heatwave")
        rows = rolling_plan(live, info, self.api, start=9 * 60, until=16 * 60, n_seeds=3)
        applied = [r for r in rows if r["applied"]]
        self.assertEqual(len(applied), 1)
        self.assertGreater(applied[0]["gain_lo"], 0.0)                  # only acts when the range clears zero
        calm, calm_info = self._live("calm")
        quiet = rolling_plan(calm, calm_info, self.api, start=9 * 60, until=14 * 60, n_seeds=3)
        self.assertFalse(any(r["applied"] for r in quiet))
        self.assertEqual(calm.sim.cap["d"], 5)

    def test_rolling_is_deterministic_and_steps_every_30_minutes(self):
        runs = []
        for _ in range(2):
            live, info = self._live("heatwave")
            runs.append(rolling_plan(live, info, self.api, start=10 * 60, until=12 * 60, n_seeds=2))
        self.assertEqual(runs[0], runs[1])
        self.assertEqual([r["t"] for r in runs[0]], [600, 630, 660, 690, 720])

    def test_rolling_never_sees_the_future(self):
        live, info = self._live("heatwave")
        seen = []
        original = live.snapshot

        def spy():
            snap = original()
            seen.append((snap.now, snap.cnt["future"], sum(p.t > snap.now for p in snap.pat.values())))
            return snap

        live.snapshot = spy
        n_before = len(self.api.calls["generate"])
        rolling_plan(live, info, self.api, start=10 * 60, until=11 * 60, n_seeds=2)
        self.assertTrue(seen and all(f == 0 and later == 0 for _now, f, later in seen))
        self.assertEqual(len(self.api.calls["generate"]), n_before)     # the rolling loop never calls generate

    def test_snapshot_with_a_stale_departure_can_be_run(self):
        # Simulator quirk: after an expedite the old DEPART event stays in the heap; if the patient
        # has left and snapshot(prune=True) deleted them, running the copy raises KeyError.
        sim = EDSim()
        sim.add_arrivals([Arrival(0, 0.0, 2, 2.0, 10.0, True, 200.0)])
        sim.run_until(50.0)
        with boarding_cut_shim(0.5):
            sim.apply(Action(expedite=True, duration=360.0))
            sim.run_until(150.0)                       # the patient leaves at 131; an old event is due at 212
        snap = sim.snapshot(strip_future=True, prune=True)
        self.assertEqual(len(snap.pat), 0)
        # CHANGED 2026-10-17: the simulator quirk is FIXED (ed_simulator.py, DEPART handler now uses
        # self.pat.get). This used to assert that a KeyError was raised; now it must run without error.
        # clean_snapshot() is no longer needed for this bug but is harmless, so it stays as a safety net.
        EDSim.restore(snap).run_until(300.0)
        opt.clean_snapshot(snap)
        EDSim.restore(snap).run_until(300.0)           # fine with the work-around
        self.assertEqual(opt.run_from_snapshot(DO_NOTHING, snap, [], 300.0).urgent_n, 0)

    def test_big_stability_penalty_keeps_the_loop_calm(self):
        live, info = self._live("heatwave")
        rows = rolling_plan(live, info, self.api, start=9 * 60, until=16 * 60, n_seeds=3,
                            settings=Settings(stability_penalty_pp=50.0))   # changing course is very costly
        self.assertFalse(any(r["applied"] for r in rows))
        self.assertEqual(live.sim.cap["d"], 5)

    def test_stability_penalty(self):
        a, b = (1, False, 0.0), (0, True, 0.5)
        self.assertEqual(stability_penalty(a, a), 0.0)
        self.assertEqual(stability_penalty(a, b), opt.STABILITY_PENALTY_PP)
        self.assertEqual(stability_penalty(a, b, Settings(stability_penalty_pp=2.0)), 2.0)

    def test_rolling_candidates(self):
        cands = opt.rolling_candidates(600.0)
        self.assertEqual(len(cands), 12)
        self.assertEqual(cands[0], DO_NOTHING)
        self.assertTrue(all(c.start == 600.0 and c.duration == opt.ROLLING_WINDOW_MIN
                            for c in cands if not c.is_nothing))


class CommandLine(unittest.TestCase):
    def _run(self, argv, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(argv, api=StubAPI(), **kw)
        return code, out.getvalue()

    def test_quick_mode_prints_a_table_and_writes_nothing(self):
        repo_results = Path(opt.__file__).resolve().parents[2] / "results"
        existed = repo_results.exists()
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with mock.patch.object(opt, "write_results", side_effect=AssertionError("must not write")):
                    code, text = self._run(["--quick"])
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(old_cwd)
        self.assertEqual(code, 0)
        self.assertEqual(repo_results.exists(), existed)
        self.assertIn("Scenario: normal", text)
        self.assertIn("Scenario: heatwave", text)
        self.assertNotIn("Scenario: hurricane", text)                   # quick = 2 scenarios
        self.assertIn("RECOMMENDED", text)
        self.assertIn("Do nothing", text)
        self.assertIn("Mode: quick", text)
        self.assertEqual(text.count("Look-ahead seeds: 5"), 2)

    def test_full_mode_writes_csv_and_json_to_the_chosen_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = self._run(["--scenario", "calm", "--seeds", "2", "--out", tmp, "--top", "0"])
            self.assertEqual(code, 0)
            self.assertEqual(sorted(os.listdir(tmp)), ["plan_search_calm.csv", "plan_search_calm.json"])
            with open(Path(tmp) / "plan_search_calm.json", encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["scenario"], "calm")
            self.assertEqual(len(data["rows"]), 78)
            self.assertEqual(data["recommended"]["label"], "Do nothing")
            with open(Path(tmp) / "plan_search_calm.csv", encoding="utf-8") as fh:
                lines = fh.read().strip().splitlines()
            self.assertEqual(len(lines), 79)
            self.assertIn("lookahead_gain_lo", lines[0])
            self.assertNotIn("plans shown", text)                       # --top 0 shows every plan

    def test_bad_input_gives_a_clear_message_and_exit_code_2(self):
        for argv in (["--scenario", "no_such_scenario"], ["--seeds", "0", "--scenario", "calm"]):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = main(argv + ["--out", tempfile.gettempdir() + "/never_written"], api=StubAPI())
            self.assertEqual(code, 2)
            self.assertIn("Cannot run", err.getvalue())
        self.assertFalse(Path(tempfile.gettempdir(), "never_written").exists())

    def test_missing_scenario_api_gives_a_clear_message(self):
        err = io.StringIO()
        with mock.patch.object(opt, "load_scenario_api", side_effect=RuntimeError("no scenarios yet")):
            with contextlib.redirect_stderr(err):
                code = main(["--quick"])
        self.assertEqual(code, 2)
        self.assertIn("no scenarios yet", err.getvalue())


if __name__ == "__main__":
    unittest.main()
