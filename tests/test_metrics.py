"""Tests for fairness metrics using textbook examples."""

from dataclasses import dataclass
from typing import List

from gerrymander.districting import DistrictPlan
from gerrymander.metrics import efficiency_gap, mean_median, partisan_bias
from gerrymander.state_model import Precinct, StateGrid


def _make_plan(district_votes: List[tuple]) -> DistrictPlan:
    """Build a tiny synthetic plan: one precinct per district."""
    precincts = []
    assignment = []
    for d, (d_votes, r_votes) in enumerate(district_votes):
        total = d_votes + r_votes
        d_share = d_votes / total
        precincts.append(Precinct(
            idx=d, row=0, col=d, population=total,
            d_share=d_share,
            demographics={"white": 1.0, "black": 0.0, "hispanic": 0.0, "asian": 0.0, "other": 0.0},
        ))
        assignment.append(d)
    grid = StateGrid(
        state="XX", rows=1, cols=len(district_votes),
        num_districts=len(district_votes),
        precincts=precincts, d_lean=0.5, seed=0,
    )
    return DistrictPlan(grid=grid, assignment=assignment, label="test")


def test_efficiency_gap_textbook():
    # Stephanopoulos & McGhee example: party A wins 75/25 in three districts,
    # party B wins 60/40 in two districts (10 total dist, scaled small).
    # Simpler classic: 5 districts.
    # District votes (D, R): A wins big in 1, B wins narrow in 4.
    plan = _make_plan([(75, 25), (40, 60), (40, 60), (40, 60), (40, 60)])
    eg = efficiency_gap(plan)
    # D wasted: 24+40+40+40+40 = 184; R wasted: 25+9+9+9+9 = 61
    # Total = 500; EG = (61-184)/500 = -0.246
    assert abs(eg - (61 - 184) / 500) < 0.01


def test_mean_median_symmetric_zero():
    # Symmetric distribution -> mean == median -> 0.
    plan = _make_plan([(40, 60), (50, 50), (60, 40)])
    assert abs(mean_median(plan, "D")) < 1e-6


def test_mean_median_packed_positive_for_d():
    # D packed into one stronghold, loses the rest narrowly.
    plan = _make_plan([(95, 5), (45, 55), (45, 55), (45, 55), (45, 55)])
    mm = mean_median(plan, "D")
    # mean > median -> positive
    assert mm > 0.05


def test_partisan_bias_neutral():
    plan = _make_plan([(50, 50), (50, 50), (50, 50), (50, 50), (50, 50)])
    assert abs(partisan_bias(plan, "D")) < 0.6  # tie districts -> flexible


def test_efficiency_gap_balanced_plan():
    # Symmetric outcomes -> EG near 0.
    plan = _make_plan([(60, 40), (40, 60), (60, 40), (40, 60)])
    assert abs(efficiency_gap(plan)) < 0.05
