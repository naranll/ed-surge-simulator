"""Fix #7: 'boarders' = ADMITTED patients holding a bed while they wait for a ward (not discharged patients)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.ed_simulator import EDSim  # noqa: E402
from src.simulation.hospital import PRESETS, binding_constraint, build_sim, day_metrics, prepare, run_day  # noqa: E402
from src.simulation.scenarios import generate, truth_seed  # noqa: E402


def _admitted_in_bed(sim):
    return sum(1 for p in sim.pat.values() if p.st == "board" and p.admit)


def test_counter_equals_a_direct_count_at_every_stop_and_after_a_snapshot():
    cfg = PRESETS["Large"]
    truth, _ = generate("heatwave", truth_seed(3))
    sim = build_sim(cfg, prepare(cfg, truth))
    for t in range(60, 1441, 60):
        sim.run_until(t)
        assert sim.board_admitted == _admitted_in_bed(sim)
        snap = sim.snapshot(strip_future=True, prune=True)
        assert snap.board_admitted == _admitted_in_bed(snap)
    assert sim.board_admitted <= sim.cnt["board"]


def test_old_counter_still_includes_discharged_patients_in_flow_mode():
    cfg = PRESETS["Large"]
    truth, _ = generate("normal", truth_seed(1))
    sim = build_sim(cfg, prepare(cfg, truth))
    sim.run_until(14 * 60)
    assert sim.cnt["board"] > sim.board_admitted          # this gap is why the old number was mislabelled


def test_day_metrics_timeline_uses_admitted_boarders_and_default_sim_is_unchanged():
    cfg = PRESETS["Large"]
    truth, _ = generate("normal", truth_seed(2))
    m = day_metrics(run_day(cfg, truth), cfg)
    assert all(a <= b for a, b in zip(m["timeline"]["boarders"], m["timeline"]["in_bed_after_consult"]))
    old = EDSim()
    old.add_arrivals(truth)
    old.run_until(1440)
    assert old.board_admitted == _admitted_in_bed(old)


def test_binding_constraint_names_a_resource():
    cfg = PRESETS["Small"]
    truth, _ = generate("compound", truth_seed(0))
    text = binding_constraint(run_day(cfg, truth))
    assert text.split(" ")[0] in ("boarding", "beds", "staff")
