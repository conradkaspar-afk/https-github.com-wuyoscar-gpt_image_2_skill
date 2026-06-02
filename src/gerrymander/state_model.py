"""Synthetic-but-realistic state precinct model.

A `StateGrid` is an `rows x cols` lattice of `Precinct` cells. Each precinct
has population, two-party vote share, and a demographic breakdown. Spatial
correlation is produced by smoothing a 2D Gaussian random field so that
urban clusters lean Democratic and minority-heavy while rural cells lean
Republican and more white — matching observed US geographic polarization
without requiring shapefiles or external data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


# Per-state presets: (grid_rows, grid_cols, num_districts, D_lean, urban_frac)
# D_lean is the statewide two-party Democratic share (approx 2020 presidential).
# urban_frac is the fraction of population in dense clusters.
STATE_PRESETS: Dict[str, Tuple[int, int, int, float, float]] = {
    "AL": (12, 10, 7, 0.37, 0.30), "AK": (10, 14, 1, 0.44, 0.35),
    "AZ": (14, 16, 9, 0.50, 0.55), "AR": (10, 12, 4, 0.36, 0.30),
    "CA": (20, 16, 52, 0.64, 0.70), "CO": (12, 14, 8, 0.56, 0.55),
    "CT": (8, 10, 5, 0.59, 0.55), "DE": (10, 6, 1, 0.59, 0.50),
    "FL": (12, 22, 28, 0.48, 0.60), "GA": (14, 14, 14, 0.50, 0.50),
    "HI": (8, 12, 2, 0.64, 0.55), "ID": (14, 10, 2, 0.34, 0.30),
    "IL": (16, 12, 17, 0.57, 0.65), "IN": (14, 12, 9, 0.41, 0.45),
    "IA": (10, 14, 4, 0.45, 0.35), "KS": (10, 14, 4, 0.42, 0.40),
    "KY": (10, 16, 6, 0.36, 0.35), "LA": (10, 14, 6, 0.40, 0.45),
    "ME": (14, 8, 2, 0.53, 0.30), "MD": (10, 12, 8, 0.65, 0.60),
    "MA": (10, 12, 9, 0.65, 0.65), "MI": (14, 14, 13, 0.51, 0.50),
    "MN": (14, 12, 8, 0.52, 0.50), "MS": (12, 10, 4, 0.41, 0.30),
    "MO": (12, 14, 8, 0.42, 0.45), "MT": (10, 18, 2, 0.41, 0.30),
    "NE": (10, 16, 3, 0.39, 0.40), "NV": (12, 12, 4, 0.51, 0.65),
    "NH": (10, 8, 2, 0.53, 0.35), "NJ": (14, 8, 12, 0.58, 0.70),
    "NM": (12, 14, 3, 0.55, 0.45), "NY": (16, 16, 26, 0.61, 0.70),
    "NC": (12, 18, 14, 0.49, 0.45), "ND": (10, 14, 1, 0.34, 0.30),
    "OH": (14, 14, 15, 0.45, 0.50), "OK": (12, 14, 5, 0.34, 0.40),
    "OR": (14, 12, 6, 0.57, 0.50), "PA": (12, 16, 17, 0.50, 0.55),
    "RI": (8, 6, 2, 0.60, 0.55), "SC": (12, 12, 7, 0.43, 0.40),
    "SD": (10, 14, 1, 0.36, 0.30), "TN": (10, 18, 9, 0.38, 0.40),
    "TX": (18, 20, 38, 0.47, 0.60), "UT": (12, 12, 4, 0.39, 0.50),
    "VT": (12, 6, 1, 0.66, 0.30), "VA": (12, 16, 11, 0.54, 0.50),
    "WA": (14, 12, 10, 0.58, 0.60), "WV": (10, 12, 2, 0.30, 0.30),
    "WI": (12, 14, 8, 0.50, 0.45), "WY": (10, 12, 1, 0.27, 0.30),
}


DEMOGRAPHIC_KEYS = ("white", "black", "hispanic", "asian", "other")


@dataclass
class Precinct:
    idx: int
    row: int
    col: int
    population: int
    d_share: float  # two-party Democratic vote share, 0..1
    demographics: Dict[str, float]  # shares summing to 1.0

    @property
    def coords(self) -> Tuple[int, int]:
        return (self.row, self.col)


@dataclass
class StateGrid:
    state: str
    rows: int
    cols: int
    num_districts: int
    precincts: List[Precinct]
    d_lean: float
    seed: int
    _index: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._index:
            self._index = {(p.row, p.col): p.idx for p in self.precincts}

    def neighbors(self, idx: int) -> List[int]:
        p = self.precincts[idx]
        out: List[int] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            key = (p.row + dr, p.col + dc)
            if key in self._index:
                out.append(self._index[key])
        return out

    @property
    def n(self) -> int:
        return len(self.precincts)

    def total_population(self) -> int:
        return sum(p.population for p in self.precincts)


def list_states() -> List[str]:
    return sorted(STATE_PRESETS.keys())


def _smooth_field(rng: np.random.Generator, rows: int, cols: int, passes: int = 6) -> np.ndarray:
    """Generate a smoothed 2D Gaussian random field on a grid."""
    f = rng.standard_normal((rows, cols))
    for _ in range(passes):
        f = (
            f
            + np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1)
        ) / 5.0
    f = (f - f.mean()) / (f.std() + 1e-9)
    return f


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_state(state: str, seed: int = 0) -> StateGrid:
    """Construct a synthetic precinct grid for a US state."""
    state = state.upper()
    if state not in STATE_PRESETS:
        raise ValueError(f"Unknown state {state!r}. Try: {', '.join(list_states())}")
    rows, cols, n_districts, d_lean, urban_frac = STATE_PRESETS[state]
    rng = np.random.default_rng(seed)

    # Urban density field: high values = dense urban areas.
    density_field = _smooth_field(rng, rows, cols, passes=4)
    # Threshold so roughly `urban_frac` of cells are "urban".
    cutoff = np.quantile(density_field, 1.0 - urban_frac)
    urbanness = _logistic((density_field - cutoff) * 3.0)  # 0..1 smooth

    # Population: urban cells ~3-4x more people than rural (coarse grid).
    base_pop = 4000
    population = (base_pop * (1.0 + 3.0 * urbanness) * (1.0 + 0.15 * rng.standard_normal((rows, cols)))).clip(min=500)
    population = population.astype(int)

    # Party lean: urban -> D, rural -> R. Add local noise and a separate field.
    party_field = _smooth_field(rng, rows, cols, passes=5)
    # Calibrate intercept so population-weighted mean matches d_lean.
    raw = 1.2 * urbanness - 0.7 * (1.0 - urbanness) + 0.6 * party_field
    for _ in range(30):
        d_share = _logistic(raw)
        weighted = float((d_share * population).sum() / population.sum())
        raw += (d_lean - weighted) * 2.0
    d_share = _logistic(raw)

    # Demographics: urban -> more diverse; rural -> more white.
    minority_field = _smooth_field(rng, rows, cols, passes=4)
    minority_intensity = _logistic(1.5 * urbanness + 0.7 * minority_field - 0.5)
    # Sub-fields select which minority group dominates locally.
    black_f = _logistic(_smooth_field(rng, rows, cols, passes=3))
    hisp_f = _logistic(_smooth_field(rng, rows, cols, passes=3))
    asian_f = _logistic(_smooth_field(rng, rows, cols, passes=3))
    sub_sum = black_f + hisp_f + asian_f + 1e-9
    b_share_min = (black_f / sub_sum) * minority_intensity * 0.85
    h_share_min = (hisp_f / sub_sum) * minority_intensity * 0.85
    a_share_min = (asian_f / sub_sum) * minority_intensity * 0.85
    other_share = (1.0 - black_f - hisp_f - asian_f).clip(0, 1) * minority_intensity * 0.15
    white_share = (1.0 - (b_share_min + h_share_min + a_share_min + other_share)).clip(0.05, 1.0)
    # Re-normalize to sum to 1.
    total = white_share + b_share_min + h_share_min + a_share_min + other_share
    white_share /= total
    b_share = b_share_min / total
    h_share = h_share_min / total
    a_share = a_share_min / total
    o_share = other_share / total

    precincts: List[Precinct] = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            demo = {
                "white": float(white_share[r, c]),
                "black": float(b_share[r, c]),
                "hispanic": float(h_share[r, c]),
                "asian": float(a_share[r, c]),
                "other": float(o_share[r, c]),
            }
            precincts.append(Precinct(
                idx=idx, row=r, col=c,
                population=int(population[r, c]),
                d_share=float(d_share[r, c]),
                demographics=demo,
            ))
            idx += 1

    return StateGrid(
        state=state, rows=rows, cols=cols, num_districts=n_districts,
        precincts=precincts, d_lean=d_lean, seed=seed,
    )
