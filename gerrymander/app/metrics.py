"""Partisan and shape metrics computed on a gerrychain Partition.

Functions here are also unit-tested directly on dicts of district -> (dem, rep)
so they don't require gerrychain to be importable.
"""
from __future__ import annotations

import math
import statistics
from typing import Mapping, Tuple


def seats_won(votes: Mapping[int, Tuple[float, float]]) -> Tuple[int, int]:
    dem = sum(1 for d, r in votes.values() if d > r)
    rep = sum(1 for d, r in votes.values() if r > d)
    return dem, rep


def efficiency_gap(votes: Mapping[int, Tuple[float, float]]) -> float:
    """Stephanopoulos–McGhee efficiency gap. Positive = pro-D, negative = pro-R."""
    wasted_d = 0.0
    wasted_r = 0.0
    total = 0.0
    for d, r in votes.values():
        n = d + r
        if n == 0:
            continue
        total += n
        threshold = n / 2.0
        if d > r:
            wasted_d += d - threshold
            wasted_r += r
        else:
            wasted_r += r - threshold
            wasted_d += d
    if total == 0:
        return 0.0
    return (wasted_r - wasted_d) / total


def mean_median(votes: Mapping[int, Tuple[float, float]]) -> float:
    """Mean minus median of the Democratic vote share across districts.

    Positive = pro-D, negative = pro-R.
    """
    shares = []
    for d, r in votes.values():
        n = d + r
        if n > 0:
            shares.append(d / n)
    if not shares:
        return 0.0
    return statistics.mean(shares) - statistics.median(shares)


def polsby_popper(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return (4.0 * math.pi * area) / (perimeter * perimeter)
