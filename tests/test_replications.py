"""--replications: configurable forecast runs, worker pool gives the same answer as a serial run."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization import optimizer as opt  # noqa: E402
from src.simulation import registry  # noqa: E402
from src.simulation.hospital import PRESETS, runner_for  # noqa: E402
from src.simulation.scenarios import USE_LEGACY_PARAMS  # noqa: E402


def test_ci_uses_the_t_table_beyond_9_degrees_of_freedom():
    xs = [float(i % 7) for i in range(30)]
    m, lo, hi = opt._ci(xs)
    m2, lo2, hi2 = opt.confidence_range(xs)
    assert (round(lo, 9), round(hi, 9)) == (round(lo2, 9), round(hi2, 9))
    assert opt.t_critical(4) == 2.776                      # unchanged for the old 5-run default


def test_plan_search_replications_override_quick():
    res = opt.plan_search("normal", quick=True, api=registry.api(), runner=runner_for(PRESETS["Small"]), replications=7)
    assert res.n_seeds == 7 and res.truth["n_seeds"] == 7


def test_hospital_runner_is_picklable_and_parallel_gives_the_same_result():
    import pickle
    r = runner_for(PRESETS["Small"])
    assert pickle.loads(pickle.dumps(r)) == r and r.parallel_safe
    a = opt.plan_search("heatwave", quick=True, api=registry.api(), runner=r, replications=4, workers=1)
    b = opt.plan_search("heatwave", quick=True, api=registry.api(), runner=r, replications=4, workers=2)
    assert a.rows == b.rows and a.recommended == b.recommended


@pytest.mark.skipif(USE_LEGACY_PARAMS, reason="legacy route keeps the old 5 runs")
def test_demo_uses_the_requested_number_of_replications():
    from src.demo.live_demo import DEFAULT_REPLICATIONS, LiveDay, recommend_now
    assert DEFAULT_REPLICATIONS == 30
    live = LiveDay("heatwave", 3)
    live.advance_to(600)
    rec = recommend_now(live, replications=6)
    assert all(len(v) == 6 for v in rec.results.values())
