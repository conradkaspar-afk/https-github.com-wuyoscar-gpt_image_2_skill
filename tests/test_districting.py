"""Invariants for districting algorithms."""

import pytest

from gerrymander.state_model import build_state
from gerrymander.districting import (
    neutral_districts, pack_and_crack, random_districts, _is_contiguous,
)
from gerrymander.metrics import efficiency_gap


@pytest.mark.parametrize("state", ["PA", "NC", "OH"])
def test_neutral_contiguity_and_balance(state):
    grid = build_state(state, seed=1)
    plan = neutral_districts(grid, seed=1)
    assert plan.num_districts == grid.num_districts
    for d in range(plan.num_districts):
        assert _is_contiguous(grid, plan.assignment, d), f"district {d} not contiguous"
    pops = plan.district_population()
    ideal = grid.total_population() / grid.num_districts
    spread = (max(pops) - min(pops)) / ideal
    assert spread < 0.30, f"population imbalance too large: {spread:.2%}"


@pytest.mark.parametrize("state,party", [("PA", "R"), ("NC", "D"), ("WI", "R")])
def test_gerrymander_skews_seats(state, party):
    grid = build_state(state, seed=2)
    neutral = neutral_districts(grid, seed=2)
    gerry = pack_and_crack(grid, target_party=party, intensity=0.9, seed=2)
    # Gerrymander should grant target party >= neutral seats.
    neutral_seats = sum(1 for d, r in neutral.district_d_votes()
                       if (d > r if party == "D" else r > d))
    gerry_seats = sum(1 for d, r in gerry.district_d_votes()
                     if (d > r if party == "D" else r > d))
    assert gerry_seats >= neutral_seats


def test_random_runs():
    grid = build_state("OH", seed=3)
    plan = random_districts(grid, seed=3)
    assert plan.num_districts == grid.num_districts


def test_efficiency_gap_moves_right_direction():
    grid = build_state("NC", seed=7)
    neutral = neutral_districts(grid, seed=7)
    gerry_r = pack_and_crack(grid, target_party="R", intensity=0.9, seed=7)
    # EG positive favors R; gerrymander for R should increase EG.
    assert efficiency_gap(gerry_r) >= efficiency_gap(neutral) - 0.01
