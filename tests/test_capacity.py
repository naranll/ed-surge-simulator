"""One capacity source of truth: configs/capacity.json feeds the presets, run_all and run.py."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization.optimizer import SEEDS, recommend  # noqa: E402
from src.simulation import capacity  # noqa: E402
from src.simulation.ed_simulator import LiveED  # noqa: E402
from src.simulation.hospital import PRESETS  # noqa: E402
from src.simulation.scenarios import SCENARIOS, USE_LEGACY_PARAMS, generate_arrivals  # noqa: E402


def test_capacity_file_has_a_source_and_valid_reference():
    data = json.loads(capacity.path().read_text(encoding="utf-8"))
    assert "NHAMCS" in data["source"] and "calibrate.py" in data["source"]
    ref = capacity.reference()
    assert all(isinstance(v, int) and v >= 1 for v in ref.values())
    assert "UNSOURCED" in data["preset_volume_shares"]["source"]


def test_presets_come_from_the_capacity_file():
    ref = capacity.reference()
    large = PRESETS[capacity.load()["reference_hospital"]["preset_name"]]
    assert (large.beds, large.doctors, large.triage_nurses) == (ref["beds"], ref["doctors"], ref["triage_nurses"])
    for name, share in capacity.load()["preset_volume_shares"]["values"].items():
        c = PRESETS[name]
        assert c.volume_scale == share and c.beds == max(1, math.ceil(ref["beds"] * share - 1e-9))


@pytest.mark.skipif(USE_LEGACY_PARAMS, reason="the legacy switch keeps run.py on the original 20-bed route")
def test_run_py_demo_uses_the_calibrated_hospital():
    from src.demo.live_demo import LiveDay
    live = LiveDay("normal", 42)
    ref = capacity.reference()
    assert (live.sim.cap["b"], live.sim.cap["d"], live.sim.cap["n"]) == (ref["beds"], ref["doctors"], ref["triage_nurses"])
    assert live.sim.per_patient_stay is True


def test_recommend_with_given_forecasts_equals_the_old_default():
    live = LiveED(SCENARIOS["heatwave"], 7)
    live.advance_to(600)
    snap = live.snapshot()
    own = {s: generate_arrivals(SCENARIOS["heatwave"], s, 600, 840, base_pid=10**6) for s in SEEDS}
    a = recommend(snap, SCENARIOS["heatwave"])
    b = recommend(snap, SCENARIOS["heatwave"], forecasts=own)
    assert (a.action, a.loss_best, a.breach_reduction_ci) == (b.action, b.loss_best, b.breach_reduction_ci)


def test_missing_capacity_file_gives_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setenv("ED_CAPACITY_FILE", str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError, match="calibrate.py"):
        capacity.load()
