"""Tests for A1 (patient-flow model on real NHAMCS data)."""
import builtins
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
import train_patient_flow as a1  # noqa: E402

DATA = ROOT / "data" / "processed" / "nhamcs_2022_cleaned.csv"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="cleaned NHAMCS file not present")

FORBIDDEN = ("wait", "lov", "admit", "adisp", "diag", "los", "board", "lwbs", "proc", "med", "image", "xray", "bnp", "dispos")


@needs_data
def test_features_contain_no_outcome_or_after_arrival_variable():
    d, _ = a1.load_table(DATA)
    X = a1.build_X(d)
    for col in X.columns:
        assert not any(tok in col.lower() for tok in FORBIDDEN), f"suspicious feature {col}"
    assert set(a1.INCLUDED) == {"age_years", "triage_level", "arrival_hour", "weekday", "payment_type", "sex",
                                "ambulance_arrival", "injury_flag"}


@needs_data
def test_probabilities_are_between_0_and_1():
    d, _ = a1.load_table(DATA)
    X = a1.build_X(d)
    m = a1.fit(a1.make_model("logistic", {"C": 1.0}, "clf"), X, d["admitted"].to_numpy(), np.ones(len(d)))
    p = a1.predict(m, X, "clf")
    assert p.min() >= 0.0 and p.max() <= 1.0


@needs_data
def test_lookup_is_well_formed():
    d, _ = a1.load_table(DATA)
    lk = a1.build_lookup(d)
    for lvl in "12345":
        assert abs(sum(lk["age_share_given_triage"][lvl].values()) - 1.0) < 1e-3
        for cell in lk["admission"][lvl]["bands"].values():
            assert 0.0 <= cell["rate"] <= 1.0 and cell["n"] >= 0
    assert lk["tag"] == "DATA"


@needs_data
def test_same_seed_gives_same_split_and_metrics(tmp_path):
    out = []
    for i in range(2):
        assert a1.main(["--quick", "--out", str(tmp_path / f"run{i}")]) == 0
        m = json.loads((tmp_path / f"run{i}" / "metrics.json").read_text())["patient_flow"]
        out.append((m["split"], m["admission"]["test"]["chosen_model"]["auc_roc"], m["admission"]["chosen"]["params"]))
    assert out[0] == out[1]
    assert "HOSPCODE" in out[0][0]["grouped_by"] and out[0][0]["test_hospitals"] > 0   # whole hospitals are held out


def test_graceful_message_when_ml_library_is_missing(monkeypatch, capsys):
    real = builtins.__import__

    def fake(name, *args, **kw):
        if name.startswith("xgboost"):
            raise ImportError("no xgboost here")
        return real(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert a1.main(["--quick"]) == 2
    assert "SKIPPED A1" in capsys.readouterr().out
