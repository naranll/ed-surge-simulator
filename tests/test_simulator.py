import unittest

from src.simulation.ed_simulator import Action, EDSim, LiveED, simulate
from src.simulation.scenarios import HOURLY, SCENARIOS, Arrival, generate_arrivals

NORMAL, COMPOUND = SCENARIOS["normal"], SCENARIOS["compound"]


class ScenarioTests(unittest.TestCase):
    def test_hourly_normalised_and_volume_ordering(self):
        self.assertAlmostEqual(sum(HOURLY), 1.0)
        v = {k: s.expected_daily() for k, s in SCENARIOS.items()}
        self.assertAlmostEqual(v["normal"], 200.0)
        self.assertGreater(v["heatwave"], v["normal"])
        self.assertGreater(v["hurricane"], v["heatwave"])
        self.assertGreater(v["compound"], v["hurricane"])
        for s in SCENARIOS.values():
            self.assertAlmostEqual(sum(s.acuity), 1.0)

    def test_crn_arrivals_reproducible(self):
        a, b = generate_arrivals(NORMAL, 7), generate_arrivals(NORMAL, 7)
        self.assertEqual(a, b)
        self.assertNotEqual(a, generate_arrivals(NORMAL, 8))
        self.assertTrue(all(x.t <= y.t for x, y in zip(a, a[1:])))
        self.assertTrue(150 < len(a) < 260)

    def test_action_parse_and_cost(self):
        self.assertEqual(Action.parse("DO_NOTHING").cost, 0)
        a = Action.parse("ADD_PHYSICIAN+FAST_TRACK")
        self.assertEqual((a.cost, a.name), (750.0, "ADD_PHYSICIAN+FAST_TRACK"))
        self.assertEqual(Action.parse("ADD_NURSE").cost, 200)
        with self.assertRaises(ValueError):
            Action.parse("TELEPORT")


class EngineTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(simulate(seed=3, hours=24), simulate(seed=3, hours=24))

    def test_snapshot_resume_is_exact(self):
        arr = generate_arrivals(COMPOUND, 5, 0, 600)
        full = EDSim(); full.add_arrivals(arr); full.run_until(600)
        a = EDSim(); a.add_arrivals(arr); a.run_until(240)
        snap = a.snapshot()
        a.run_until(600)                      # original keeps running; snapshot must be isolated
        b = EDSim.restore(snap)
        self.assertEqual(b.now, 240)
        b.run_until(600)
        self.assertEqual(full.result(0), b.result(0))
        self.assertEqual({k: (p.st, p.t_doc, p.t_dep) for k, p in full.pat.items()},
                         {k: (p.st, p.t_doc, p.t_dep) for k, p in b.pat.items()})

    def test_simulate_does_not_mutate_snapshot(self):
        live = LiveED(COMPOUND, 1); live.advance_to(300)
        snap = live.snapshot()
        before = (snap.now, dict(snap.cnt), len(snap.heap))
        simulate(snap, action="ADD_PHYSICIAN", seed=1)
        self.assertEqual(before, (snap.now, dict(snap.cnt), len(snap.heap)))

    def test_forecast_snapshot_hides_future(self):
        live = LiveED(NORMAL, 1); live.advance_to(300)
        snap = live.snapshot()
        self.assertEqual(snap.cnt["future"], 0)
        self.assertFalse(snap.heap and any(e[2] == 0 for e in snap.heap))
        self.assertGreater(live.sim.cnt["future"], 0)

    def test_invariants(self):
        s = EDSim(); s.add_arrivals(generate_arrivals(COMPOUND, 2, 0, 720))
        for t in range(30, 721, 30):
            s.run_until(t)
            self.assertTrue(0 <= s.busy["b"] <= s.cap["b"] and 0 <= s.busy["d"] <= s.cap["d"]
                            and 0 <= s.busy["n"] <= s.cap["n"])
            self.assertEqual(sum(s.cnt.values()), len(s.pat))
            held = sum(p.st in ("doc_q", "consult", "board") for p in s.pat.values())
            self.assertEqual(held, s.busy["b"])

    def test_lwbs_after_180_minutes(self):
        s = EDSim(nurses=1, docs=1, beds=1)
        s.add_arrivals([Arrival(i, float(i), 3, 2.0, 60.0, False, 0.0) for i in range(8)])
        s.run_until(600)
        r = s.result(0)
        self.assertGreater(r.lwbs, 0)
        for p in s.pat.values():
            if p.st == "lwbs":
                self.assertAlmostEqual(p.t_lwbs - p.t, 180.0)

    def test_ats1_skips_triage_and_doctor_not_locked_for_stay(self):
        s = EDSim(nurses=1, docs=1, beds=5)
        s.add_arrivals([Arrival(0, 0.0, 1, 3.0, 10.0, True, 200.0), Arrival(1, 1.0, 3, 3.0, 10.0, False, 0.0)])
        s.run_until(30)
        self.assertEqual(s.pat[0].t_doc, 0.0)            # no triage delay for ATS 1
        self.assertEqual(s.pat[1].t_doc, 10.0)           # doctor free after 10-min consult...
        self.assertEqual(s.pat[0].st, "board")           # ...while ATS1 patient still boards

    def test_fast_track_and_expedite_effects(self):
        arr = [Arrival(0, 0.0, 5, 2.0, 20.0, False, 0.0)]
        s = EDSim(); s.add_arrivals(arr); s.apply("FAST_TRACK"); s.run_until(200)
        self.assertAlmostEqual(s.pat[0].t_dep, 2.0 + 20.0 * 0.6 + 45.0, places=6)
        arr = [Arrival(0, 0.0, 2, 2.0, 10.0, True, 200.0)]
        s = EDSim(); s.add_arrivals(arr); s.apply("INPATIENT_BUFFER_EXPEDITE"); s.run_until(400)
        self.assertAlmostEqual(s.pat[0].t_dep, 2.0 + 10.0 + 140.0, places=6)

    def test_added_physician_never_hurts_on_crn(self):
        live = LiveED(COMPOUND, 4); live.advance_to(14 * 60)
        snap = live.snapshot()
        for seed in (1, 2, 3):
            arr = generate_arrivals(COMPOUND, seed, snap.now, snap.now + 240, base_pid=10**6)
            a = simulate(snap, arr, None, hours=4)
            b = simulate(snap, arr, "ADD_PHYSICIAN+ADD_NURSE", hours=4)
            self.assertLessEqual(b.mean_d2d, a.mean_d2d + 1e-9)
            self.assertEqual(a.n, b.n)

    def test_compound_is_harder_than_normal(self):
        n, c = simulate(scenario=NORMAL, seed=1, hours=24), simulate(scenario=COMPOUND, seed=1, hours=24)
        self.assertGreater(c.lwbs, n.lwbs)
        self.assertGreater(c.total_breaches, n.total_breaches)


if __name__ == "__main__":
    unittest.main()
