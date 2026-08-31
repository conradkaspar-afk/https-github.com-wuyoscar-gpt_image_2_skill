"""Smoke tests for real-county geography loader."""

import pytest

from gerrymander.geo import build_state_real, list_real_states
from gerrymander.districting import neutral_districts, pack_and_crack, _is_contiguous


def test_lists_50_states():
    assert len(list_real_states()) == 50


@pytest.mark.parametrize("state,expected_counties_min", [
    ("PA", 60),  # PA has 67
    ("NC", 90),  # NC has 100
    ("TX", 240),  # TX has 254
])
def test_real_geo_loads(state, expected_counties_min):
    grid = build_state_real(state, seed=1)
    assert grid.n >= expected_counties_min
    assert grid.total_population() > 1_000_000
    # Adjacency graph is connected.
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nb in grid.neighbors(cur):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    assert len(seen) == grid.n, "county adjacency graph is not connected"


def test_real_geo_districting_contiguous():
    grid = build_state_real("PA", seed=1)
    plan = neutral_districts(grid, seed=1)
    assert plan.num_districts == grid.num_districts
    for d in range(plan.num_districts):
        assert _is_contiguous(grid, plan.assignment, d), f"district {d} not contiguous"


@pytest.mark.parametrize("state,seats,party", [
    # Cases that previously produced discontiguous or empty districts on real
    # county graphs (islands, single huge counties, seats far above default).
    ("HI", 5, "R"), ("DE", 2, "D"), ("AZ", 9, "D"), ("CT", 8, "R"),
    ("MA", 12, "R"), ("FL", 31, "D"), ("NY", 29, "D"), ("CO", 8, "D"),
    ("MD", 4, "R"), ("VA", 11, "R"), ("WV", 5, "D"),
])
def test_real_geo_plans_are_contiguous_and_non_empty(state, seats, party):
    grid = build_state_real(state, seed=1)
    seats = min(seats, grid.n)
    plan = pack_and_crack(grid, n=seats, target_party=party, intensity=0.9, seed=1)
    assert plan.num_districts == seats
    used = set(plan.assignment)
    assert used == set(range(seats)), f"empty district(s): {set(range(seats)) - used}"
    for d in range(seats):
        assert _is_contiguous(grid, plan.assignment, d), f"district {d} not contiguous"


def test_real_geo_precinct_has_geometry():
    grid = build_state_real("OH", seed=1)
    p = grid.precincts[0]
    assert hasattr(p, "geometry")
    assert hasattr(p, "centroid")
    assert len(p.geometry) >= 1  # at least one polygon
