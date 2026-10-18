"""Rule (c): when urgent breaches are above a normal day, never 'do nothing' unless every tested plan's 95% range
includes 0, and then say which resource is the constraint. Also: the demo prints both views."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization import optimizer as opt  # noqa: E402
from src.simulation import registry  # noqa: E402
from src.simulation.scenarios import USE_LEGACY_PARAMS  # noqa: E402

pytestmark = pytest.mark.skipif(USE_LEGACY_PARAMS, reason="rule (c) is off under the legacy switch (old decisions kept)")


class FakeRunner:
    """More patients -> fewer urgent patients on time (so a heatwave day has more breaches than a normal day).
    A clinician adds ``doc_points`` percentage points; a little day-to-day noise keeps the 95% range finite."""

    def __init__(self, doc_points):
        self.doc_points = doc_points

    def __call__(self, plan, patients):
        n = len(patients)
        share = 0.95 - 0.001 * n + (self.doc_points / 100.0 if plan.clinician else 0.0) + 0.002 * (n % 3)
        return opt.RunMetrics(1000, int(round(1000 * min(share, 1.0))), 100, 30.0, 0)

    def diagnose(self, patients):
        return "beds (99% busy)"


def _search(name, doc_points):
    return opt.plan_search(name, quick=True, api=registry.api(), runner=FakeRunner(doc_points), replications=8)


def test_small_but_real_gain_is_recommended_when_breaches_rise():
    # +8 points minus the cost penalty of a 6-hour clinician (6.3 points) = about +1.7: above 0, below the 2-point bar
    res = _search("heatwave", 8.0)
    assert not res.recommended.is_nothing
    assert res.reason.startswith("Rule (c)")
    assert res.truth["rule_c"]["breaches_rise"] is True and res.truth["gain_lo"] > 0


def test_nothing_helps_is_said_explicitly_with_the_constraint():
    res = _search("heatwave", 0.0)
    assert res.recommended.is_nothing
    assert res.reason.startswith("No tested plan helps; constraint is beds")


def test_normal_day_is_unaffected():
    res = _search("normal", 8.0)
    assert res.recommended.is_nothing and "rule_c" not in res.truth


def test_rule_c_can_be_switched_off():
    res = opt.plan_search("heatwave", quick=True, api=registry.api(), runner=FakeRunner(8.0), replications=8, rule_c=False)
    assert res.recommended.is_nothing


def test_demo_prints_both_views_and_uses_the_plan_search_rules():
    from src.demo.live_demo import LiveDay, decide
    live = LiveDay("compound", 42)
    live.advance_to(14 * 60)
    dec = decide(live, replications=8)
    text = "\n".join(dec.lines())
    assert "Next 4h:" in text and "Whole-day plan for this scenario:" in text
    assert dec.rule in ("bar", "rule_c", "nothing_needed", "no_plan_helps")
    if dec.plan.is_nothing and dec.normal_late_pct is not None and 100 - dec.base["urgent_on_time_pct"] > dec.normal_late_pct:
        assert dec.reason.startswith("No tested plan helps; constraint is")      # rule (c) in the demo
    if not dec.plan.is_nothing:
        assert "staff-hours" in text and "UNSOURCED" in text
