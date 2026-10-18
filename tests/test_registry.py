"""The scenario registry is the one definition of every scenario, and it does not change the old days."""
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation import forecaster, registry  # noqa: E402
from src.simulation import scenarios as S  # noqa: E402

legacy = pytest.mark.skipif(S.USE_LEGACY_PARAMS, reason="legacy switch uses the original hand-typed profiles")


def test_all_five_scenarios_are_registered():
    assert registry.names() == ["normal", "heatwave", "hurricane", "hurricane_burst", "compound"]
    assert registry.COMPOUND_ASSUMPTION in registry.description("compound")
    assert "UNSOURCED" in registry.description("hurricane_burst")


@pytest.mark.parametrize("seed", [S.truth_seed(0), S.truth_seed(7), 42])
def test_old_scenarios_give_exactly_the_same_days(seed):
    for key in ("normal", "heatwave", "hurricane"):
        assert registry.generate(key, seed)[0] == S.generate(key, seed)[0]
        assert registry.generate(key, seed, burst=True)[0] == S.generate(key, seed, burst=True)[0]
    assert registry.generate("hurricane_burst", seed)[0] == S.generate("hurricane", seed, burst=True)[0]


def test_forecast_matches_the_old_forecaster_and_never_contains_the_burst():
    for key in ("normal", "heatwave", "hurricane"):
        assert registry.forecast(registry.info(key), 0, 24) == forecaster.forecast(S.scenario_info(key), 0, 24)
    assert registry.forecast(registry.info("hurricane_burst"), 0, 24) == registry.forecast(registry.info("hurricane"), 0, 24)


def test_info_depends_on_the_key_only():
    for key in registry.names():
        assert registry.info(key) == registry.info(key)
        assert not {"seed", "multiplier", "burst_hour", "sizes"} & set(registry.info(key))
    assert registry.info("compound")["window"] == (0, 24) and registry.info("compound")["burst_possible"] is True


@legacy
def test_compound_is_heatwave_times_hurricane_within_the_cited_ranges():
    lo = S.SURGE_RANGES["heatwave"]["low"] * S.SURGE_RANGES["hurricane"]["low"]
    hi = S.SURGE_RANGES["heatwave"]["high"] * S.SURGE_RANGES["hurricane"]["high"]
    assert lo <= registry.expected_daily("compound") / registry.expected_daily("normal") <= hi
    ratios = [len(registry.generate("compound", S.truth_seed(n))[0]) / len(registry.generate("normal", S.truth_seed(n))[0])
              for n in range(50)]
    assert lo - 0.03 <= statistics.mean(ratios) <= hi + 0.03          # sampling noise allowance on 50 days


@legacy
def test_fixed_intensity_is_limited_to_the_published_range():
    a, _ = registry.generate("compound", 5, intensity={"heatwave": 0.06, "hurricane": 0.09})
    b, _ = registry.generate("compound", 5, intensity={"heatwave": 0.13, "hurricane": 0.13})
    assert len(b) > len(a)
    with pytest.raises(ValueError):
        registry.generate("compound", 5, intensity={"heatwave": 0.30, "hurricane": 0.10})
    with pytest.raises(ValueError):
        registry.generate("compound", 5, intensity=0.10)       # two components need a dict


def test_every_entry_point_uses_the_registry():
    for rel in ("run.py", "scripts/run_scenarios.py", "scripts/generate_days.py", "src/demo/live_demo.py",
                "src/dashboard/app_v2.py", "models/predict_risk.py"):
        assert "registry" in (ROOT / rel).read_text(encoding="utf-8"), rel
    from src.optimization.optimizer import load_scenario_api
    assert load_scenario_api().list_scenarios() == registry.names()
