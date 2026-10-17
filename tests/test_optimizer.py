import unittest

from src.dashboard.app import compute_view
from src.explainer.briefing import build_briefing
from src.optimization.optimizer import (SEEDS, candidate_actions, clinical_loss, recommend, rolling,
                                        total_loss)
from src.simulation.ed_simulator import EDSim, LiveED
from src.simulation.scenarios import SCENARIOS

COMPOUND, NORMAL = SCENARIOS["compound"], SCENARIOS["normal"]


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live = LiveED(COMPOUND, 42); live.advance_to(14 * 60)
        cls.rec = recommend(live.snapshot(), COMPOUND)

    def test_candidates(self):
        names = [a.name for a in candidate_actions()]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("DO_NOTHING", names)
        self.assertIn("ADD_PHYSICIAN+FAST_TRACK", names)

    def test_recommendation_structure(self):
        r = self.rec
        self.assertEqual(len(r.table), len(candidate_actions()))
        self.assertTrue(all(len(v) == len(SEEDS) for v in r.results.values()))
        self.assertIn(r.action, r.pareto)
        self.assertLessEqual(r.loss_best, r.loss_base + 1e-9)
        for ci in (r.wait_reduction_ci, r.breach_reduction_ci, r.loss_reduction_ci):
            self.assertLessEqual(ci[1], ci[0] + 1e-9)
            self.assertLessEqual(ci[0], ci[2] + 1e-9)
        self.assertTrue(r.headline().startswith("RECOMMENDED"))

    def test_loss_matches_formula(self):
        r = self.rec
        res = r.results["DO_NOTHING"][0]
        manual = (res.breaches[1] * 100 + res.breaches[2] * 25 + res.breaches[3] * 10 + res.breaches[4] * 3
                  + res.breaches[5] * 1 + 5 * res.median_wait + 50 * res.lwbs)
        self.assertAlmostEqual(clinical_loss(res), manual)
        self.assertAlmostEqual(total_loss(res, 600.0), manual + 60.0)

    def test_gridlock_triggers_action_and_deterministic(self):
        self.assertNotEqual(self.rec.action, "DO_NOTHING")
        live = LiveED(COMPOUND, 42); live.advance_to(14 * 60)
        again = recommend(live.snapshot(), COMPOUND)
        self.assertEqual((again.action, again.loss_best), (self.rec.action, self.rec.loss_best))

    def test_empty_ed_does_nothing(self):
        rec = recommend(EDSim(), NORMAL, hours=1)   # t=0, quiet night: nothing is worth paying for
        self.assertEqual(rec.action, "DO_NOTHING")
        self.assertEqual(rec.breach_reduction_pct, 0.0)

    def test_optimizer_never_sees_live_future(self):
        live = LiveED(COMPOUND, 42); live.advance_to(600)
        snap = live.snapshot()
        self.assertEqual(snap.cnt["future"], 0)
        self.assertEqual(sum(p.t > snap.now for p in snap.pat.values()), 0)

    def test_rolling_30_minute_cadence(self):
        live = LiveED(NORMAL, 1)
        recs = list(rolling(live, NORMAL, start=0, until=120, step=30))
        self.assertEqual([r.t for r in recs], [0, 30, 60, 90, 120])


class BriefingTests(unittest.TestCase):
    def test_three_paragraphs(self):
        live = LiveED(COMPOUND, 42); live.advance_to(14 * 60)
        text = build_briefing(COMPOUND, recommend(live.snapshot(), COMPOUND))
        paras = text.split("\n\n")
        self.assertEqual(len(paras), 3)
        self.assertTrue(paras[0].startswith("SITUATION") and paras[1].startswith("BOTTLENECKS")
                        and paras[2].startswith("INTERVENTIONS"))
        self.assertIn("Compound Calamity", paras[0])

    def test_dashboard_view(self):
        v = compute_view("hurricane", 15 * 60)
        self.assertEqual(len(v["series"]), 16)
        self.assertTrue(0 <= v["metrics"]["bed_pct"] <= 100)
        self.assertEqual(len(v["briefing"].split("\n\n")), 3)


if __name__ == "__main__":
    unittest.main()
