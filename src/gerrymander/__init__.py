"""Educational gerrymandering visualizer.

This package generates synthetic state precinct grids, draws districts using
neutral and gerrymandered (pack-and-crack) algorithms, computes standard
fairness metrics, and renders side-by-side maps. It is for education and
detection only — not for drawing real districts.
"""

from .state_model import build_state, list_states, StateGrid, Precinct
from .districting import neutral_districts, pack_and_crack, random_districts, DistrictPlan
from .metrics import efficiency_gap, mean_median, partisan_bias, seats_votes, summarize

__all__ = [
    "build_state", "list_states", "StateGrid", "Precinct",
    "neutral_districts", "pack_and_crack", "random_districts", "DistrictPlan",
    "efficiency_gap", "mean_median", "partisan_bias", "seats_votes", "summarize",
]
