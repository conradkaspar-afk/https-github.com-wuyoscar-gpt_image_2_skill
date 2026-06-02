"""Partisan fairness metrics on a `DistrictPlan`.

Definitions follow the standard literature:
- Efficiency Gap (Stephanopoulos & McGhee 2015): (wasted_R - wasted_D) / total_votes.
  Wasted votes = losing-party votes + winning-party votes above 50%+1.
  Positive value -> Republican advantage; negative -> Democratic advantage.
- Mean-Median (party D): mean district D-share minus median district D-share.
  Positive for D -> D has an advantage (mean pulled up by packed safe seats).
- Partisan bias: seat share at 50% statewide vote under uniform partisan swing,
  minus 0.5. Positive -> bias to D.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .districting import DistrictPlan


def _vote_totals(plan: DistrictPlan) -> List[Tuple[int, int]]:
    return plan.district_d_votes()


def efficiency_gap(plan: DistrictPlan) -> float:
    totals = _vote_totals(plan)
    wasted_d = 0
    wasted_r = 0
    total = 0
    for d_votes, r_votes in totals:
        votes = d_votes + r_votes
        total += votes
        threshold = votes // 2 + 1
        if d_votes >= r_votes:
            wasted_d += d_votes - threshold
            wasted_r += r_votes
        else:
            wasted_r += r_votes - threshold
            wasted_d += d_votes
    if total == 0:
        return 0.0
    return (wasted_r - wasted_d) / total


def mean_median(plan: DistrictPlan, party: str = "D") -> float:
    totals = _vote_totals(plan)
    shares: List[float] = []
    for d_votes, r_votes in totals:
        tot = d_votes + r_votes
        if tot == 0:
            continue
        s = d_votes / tot if party == "D" else r_votes / tot
        shares.append(s)
    if not shares:
        return 0.0
    mean = sum(shares) / len(shares)
    s_sorted = sorted(shares)
    mid = len(s_sorted) // 2
    median = s_sorted[mid] if len(s_sorted) % 2 == 1 else 0.5 * (s_sorted[mid - 1] + s_sorted[mid])
    return mean - median


def partisan_bias(plan: DistrictPlan, party: str = "D") -> float:
    """Seat share for `party` if statewide vote were exactly 50%, minus 0.5.

    Uses uniform partisan swing.
    """
    totals = _vote_totals(plan)
    shares: List[float] = []
    total_d = 0
    total_v = 0
    for d_votes, r_votes in totals:
        tot = d_votes + r_votes
        if tot == 0:
            continue
        shares.append(d_votes / tot)
        total_d += d_votes
        total_v += tot
    if not shares or total_v == 0:
        return 0.0
    statewide_d = total_d / total_v
    swing = 0.5 - statewide_d
    swung = [s + swing for s in shares]
    if party == "D":
        wins = sum(1 for s in swung if s > 0.5)
    else:
        wins = sum(1 for s in swung if s < 0.5)
    return wins / len(swung) - 0.5


def seats_votes(plan: DistrictPlan, party: str = "D") -> Dict[str, float]:
    totals = _vote_totals(plan)
    total_d = sum(d for d, _ in totals)
    total_r = sum(r for _, r in totals)
    total = total_d + total_r
    if total == 0:
        return {"vote_share": 0.0, "seat_share": 0.0, "proportional": 0.0, "seats": 0.0, "n": 0.0}
    if party == "D":
        vote_share = total_d / total
        wins = sum(1 for d, r in totals if d > r)
    else:
        vote_share = total_r / total
        wins = sum(1 for d, r in totals if r > d)
    n = len(totals)
    return {
        "vote_share": vote_share,
        "seat_share": wins / n,
        "proportional": vote_share,
        "seats": float(wins),
        "n": float(n),
    }


def summarize(plan: DistrictPlan) -> Dict[str, float]:
    sv_d = seats_votes(plan, "D")
    sv_r = seats_votes(plan, "R")
    return {
        "efficiency_gap": efficiency_gap(plan),
        "mean_median_D": mean_median(plan, "D"),
        "partisan_bias_D": partisan_bias(plan, "D"),
        "D_vote_share": sv_d["vote_share"],
        "D_seat_share": sv_d["seat_share"],
        "R_vote_share": sv_r["vote_share"],
        "R_seat_share": sv_r["seat_share"],
        "D_seats": sv_d["seats"],
        "R_seats": sv_r["seats"],
        "n_districts": sv_d["n"],
    }
