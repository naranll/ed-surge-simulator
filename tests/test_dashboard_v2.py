"""Smoke test for the planner dashboard: it must load for every scenario without an error."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "src" / "dashboard" / "app_v2.py")


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP, default_timeout=180).run()


def test_loads_without_error_and_shows_key_results(app):
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Urgent seen late" in labels and "Left without being seen" in labels


@pytest.mark.parametrize("scenario", ["Heatwave", "Hurricane (receiving hospital)"])
def test_every_scenario_runs(app, scenario):
    app.selectbox[2].select(scenario).run()
    assert not app.exception


def test_page_states_the_honesty_lines(app):
    text = " ".join(c.value for c in app.caption)
    assert "Simulation only" in text and "ASSUMPTION" in text
    assert any("LinuxONE" in (i.value or "") for i in app.info) or (ROOT / "results" / "linuxone_run.json").exists()
