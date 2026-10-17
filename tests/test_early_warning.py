"""Tests for A2 (early-warning model on simulated days)."""
import builtins
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
import generate_days  # noqa: E402
import train_early_warning as a2  # noqa: E402

from models import predict_risk as pr  # noqa: E402
from src.simulation.ed_simulator import EDSim  # noqa: E402
from src.simulation.hospital import PRESETS, prepare, run_day  # noqa: E402
from src.simulation.scenarios import generate, lookahead_seed, scenario_info, truth_seed  # noqa: E402

TRUTH_WORDS = ("truth", "true", "future", "seed", "multiplier", "burst_hour", "label")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    p = tmp_path_factory.mktemp("ew") / "snap.csv.gz"
    generate_days.generate_days(15, 1, p, quiet=True)
    return pd.read_csv(p)


def test_feature_list_has_no_truth_or_label_names():
    for f in pr.FEATURES:
        assert not any(w in f for w in TRUTH_WORDS), f
        assert not f.startswith(("y_", "first_breach")), f          # label columns are never features
    assert "burst_possible" in pr.FEATURES      # allowed: it comes from the advance info, not from the true day


def test_dataset_columns_use_only_the_declared_features(tiny):
    feature_cols = [c for c in tiny.columns if c not in ("scenario", "day", "hospital", "t")
                    and not c.startswith(("y_", "first_breach_"))]
    assert feature_cols == list(pr.FEATURES)


def test_features_do_not_see_the_future():
    cfg = PRESETS["Large"]
    truth, info = generate("heatwave", truth_seed(4))
    sim = EDSim(nurses=cfg.triage_capacity, docs=cfg.doctors, beds=cfg.effective_beds, per_patient_stay=True)
    sim.add_arrivals(prepare(cfg, truth))
    sim.run_until(13 * 60)
    full = pr.build_features(sim, info, cfg.volume_scale)
    stripped = pr.build_features(sim.snapshot(strip_future=True), info, cfg.volume_scale)
    assert full == stripped                       # removing every not-yet-arrived patient changes nothing


def test_same_info_gives_identical_forecast_features():
    assert pr.forecast_next4h(scenario_info("heatwave"), 600.0) == pr.forecast_next4h(scenario_info("heatwave"), 600.0)
    assert pr.forecast_next4h(scenario_info("heatwave"), 720.0)[1] > pr.forecast_next4h(scenario_info("normal"), 720.0)[1]


def test_no_day_is_in_two_splits_and_seed_ranges_are_disjoint(tiny):
    sp = a2.make_split(tiny)
    a2.check_split(sp)
    assert len(set(sp["train"]) | set(sp["val"]) | set(sp["test"])) == tiny["day"].nunique()
    assert {truth_seed(n) for n in range(100000)}.isdisjoint({lookahead_seed(n) for n in range(100000)})


def test_labels_are_binary_and_the_label_window_fits_inside_the_day(tiny):
    for c in ("y_urgent_late", "y_beds_full", "y_lwbs"):
        assert set(tiny[c].unique()) <= {0, 1}
    assert tiny["t"].min() >= 30 and tiny["t"].max() <= 1200


def test_predict_risk_graceful_when_no_model(tmp_path):
    cfg = PRESETS["Large"]
    truth, info = generate("normal", truth_seed(0))
    sim = run_day(cfg, truth)
    r = pr.predict_risk(sim, info, model_dir=tmp_path)
    assert r["available"] is False and "not trained" in r["message"]


@pytest.mark.skipif(not (ROOT / "models" / "early_warning" / "early_warning_meta.json").exists(),
                    reason="early-warning model not trained yet")
def test_predict_risk_runs_on_a_real_snapshot():
    cfg = PRESETS["Large"]
    truth, info = generate("hurricane", truth_seed(7))
    out = {}

    def grab(sim):
        if sim.now == 14 * 60:
            out["r"] = pr.predict_risk(sim, info, cfg.volume_scale)

    run_day(cfg, truth, stop_every=30.0, on_stop=grab)
    r = out["r"]
    assert r["available"] is True
    for k in ("risk_urgent_late", "risk_beds_full", "risk_lwbs"):
        assert 0.0 <= r[k] <= 1.0
    assert isinstance(r["breaks_first"], str) and isinstance(r["message"], str)


def test_graceful_message_when_ml_library_is_missing(monkeypatch, capsys):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("xgboost"):
            raise ImportError("no xgboost here")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert a2.main(["--quick"]) == 2
    assert "SKIPPED A2" in capsys.readouterr().out
