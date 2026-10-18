"""Nothing may be labelled LinuxONE unless the run record says the machine was s390x."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("matplotlib")
import make_demo_results  # noqa: E402
import make_plots  # noqa: E402


def _fake_run(tmp_path, machine):
    (tmp_path / "linuxone_run.json").write_text(json.dumps(
        {"uname_m": machine, "nproc": 3, "python": "3.12", "total_seconds": 1.0,
         "parallel_timing": [{"workers": 1, "seconds": 1.0, "days": 4, "speedup_vs_1_worker": 1.0}]}), encoding="utf-8")


@pytest.mark.parametrize("machine, accepted", [("AMD64", False), ("x86_64", False), ("s390x", True)])
def test_demo_results_only_trust_s390x(tmp_path, monkeypatch, machine, accepted):
    _fake_run(tmp_path, machine)
    monkeypatch.setattr(make_demo_results, "RESULTS", tmp_path)
    monkeypatch.setattr(make_demo_results, "MODELS", tmp_path)
    make_demo_results.main()
    out = json.loads((tmp_path / "demo_results.json").read_text(encoding="utf-8"))
    if accepted:
        assert out["linuxone"]["uname_m"] == "s390x"
    else:
        assert out["linuxone"] == "No LinuxONE run recorded yet" and "linuxone_file_rejected" in out


@pytest.mark.parametrize("machine, accepted", [("AMD64", False), ("s390x", True)])
def test_key_numbers_only_trust_s390x(tmp_path, monkeypatch, machine, accepted):
    _fake_run(tmp_path, machine)
    monkeypatch.setattr(make_plots, "RESULTS", tmp_path)
    monkeypatch.setattr(make_plots, "MODELS", tmp_path)
    monkeypatch.setattr(make_plots, "OUT", tmp_path / "for_slides")
    make_plots.key_numbers()
    text = (tmp_path / "for_slides" / "key_numbers.md").read_text(encoding="utf-8")
    assert ("Recorded run: s390x" in text) is accepted
    assert ("No LinuxONE run recorded yet" in text) is not accepted


def test_dashboard_checks_the_machine():
    src = (ROOT / "src" / "dashboard" / "app_v2.py").read_text(encoding="utf-8")
    assert 'lx.get("uname_m") != "s390x"' in src
