"""Matplotlib-based renderer for `DistrictPlan`s.

Cells are colored by either party lean or demographic majority; thick black
lines mark district boundaries. Each district is labeled with its number,
projected D/R vote share, and top two demographic groups.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from .districting import DistrictPlan
from .metrics import summarize


DEMO_COLORS = {
    "white": "#cccccc",
    "black": "#8470b0",
    "hispanic": "#f0a070",
    "asian": "#70b0a0",
    "other": "#a0a0c0",
}


def _party_color(d_share: float) -> Tuple[float, float, float]:
    # Blue for D, red for R, white at 0.5.
    if d_share >= 0.5:
        t = (d_share - 0.5) / 0.5
        # white -> blue
        return (1.0 - 0.8 * t, 1.0 - 0.7 * t, 1.0 - 0.2 * t)
    else:
        t = (0.5 - d_share) / 0.5
        # white -> red
        return (1.0 - 0.2 * t, 1.0 - 0.7 * t, 1.0 - 0.8 * t)


def _is_real_geo(plan: DistrictPlan) -> bool:
    return bool(plan.grid.precincts) and hasattr(plan.grid.precincts[0], "geometry") \
        and getattr(plan.grid.precincts[0], "geometry", None) is not None


def _draw_plan_grid(ax: plt.Axes, plan: DistrictPlan, view: str, title: str) -> None:
    grid = plan.grid
    ax.set_xlim(-0.5, grid.cols - 0.5)
    ax.set_ylim(grid.rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)

    for p in grid.precincts:
        if view == "party":
            color = _party_color(p.d_share)
        elif view == "demographic":
            top = max(p.demographics, key=lambda k: p.demographics[k])
            color = DEMO_COLORS.get(top, "#cccccc")
        else:
            color = "#dddddd"
        rect = Rectangle((p.col - 0.5, p.row - 0.5), 1, 1, facecolor=color, edgecolor="#999999", linewidth=0.3)
        ax.add_patch(rect)

    a = plan.assignment
    for p in grid.precincts:
        for nb in grid.neighbors(p.idx):
            if a[nb] == a[p.idx]:
                continue
            np_ = grid.precincts[nb]
            if np_.row == p.row + 1:
                ax.plot([p.col - 0.5, p.col + 0.5], [p.row + 0.5, p.row + 0.5], color="black", linewidth=1.6)
            elif np_.col == p.col + 1:
                ax.plot([p.col + 0.5, p.col + 0.5], [p.row - 0.5, p.row + 0.5], color="black", linewidth=1.6)
    ax.add_patch(Rectangle((-0.5, -0.5), grid.cols, grid.rows, fill=False, edgecolor="black", linewidth=1.6))

    districts = plan.districts()
    votes = plan.district_d_votes()
    demos = plan.district_demographics()
    for d, cells in enumerate(districts):
        if not cells:
            continue
        rs = [grid.precincts[i].row for i in cells]
        cs = [grid.precincts[i].col for i in cells]
        cy = sum(rs) / len(rs)
        cx = sum(cs) / len(cs)
        d_votes, r_votes = votes[d]
        tot = d_votes + r_votes
        d_pct = 100.0 * d_votes / tot if tot else 0.0
        top_demo = sorted(demos[d].items(), key=lambda kv: -kv[1])[:2]
        demo_str = ", ".join(f"{k[:1].upper()}{int(v*100)}" for k, v in top_demo)
        label = f"D{d+1}\nD{d_pct:.0f}/R{100-d_pct:.0f}\n{demo_str}"
        ax.text(cx, cy, label, ha="center", va="center", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"))


def _draw_plan_real(ax: plt.Axes, plan: DistrictPlan, view: str, title: str) -> None:
    grid = plan.grid
    bbox = getattr(grid, "_bbox", None)
    if bbox is None:
        xs = [c for p in grid.precincts for poly in p.geometry for ring in poly for c in [pt[0] for pt in ring]]  # type: ignore[attr-defined]
        ys = [c for p in grid.precincts for poly in p.geometry for ring in poly for c in [pt[1] for pt in ring]]  # type: ignore[attr-defined]
        bbox = (min(xs), min(ys), max(xs), max(ys))
    minx, miny, maxx, maxy = bbox
    pad = max((maxx - minx), (maxy - miny)) * 0.04
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect(1.0 / math.cos(math.radians((miny + maxy) / 2.0)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)

    # Fill each county polygon.
    for p in grid.precincts:
        if view == "party":
            color = _party_color(p.d_share)
        elif view == "demographic":
            top = max(p.demographics, key=lambda k: p.demographics[k])
            color = DEMO_COLORS.get(top, "#cccccc")
        else:
            color = "#dddddd"
        for poly in p.geometry:  # type: ignore[attr-defined]
            if not poly:
                continue
            outer = poly[0]
            ax.add_patch(MplPolygon(outer, closed=True, facecolor=color,
                                    edgecolor="#777777", linewidth=0.3))
            # Holes drawn in white (rare for counties; gives a visual cue).
            for ring in poly[1:]:
                ax.add_patch(MplPolygon(ring, closed=True, facecolor="white",
                                        edgecolor="#777777", linewidth=0.3))

    # District boundaries: draw the segments between counties in different
    # districts as thick lines. We approximate by drawing lines between the
    # centroids of neighboring counties that live in different districts,
    # AND by overlaying thick borders around districts using shared-segment detection.
    a = plan.assignment
    # Edge detection: for each pair of neighboring counties in different
    # districts, find their shared polygon segments and draw them thick black.
    for p in grid.precincts:
        for nb in grid.neighbors(p.idx):
            if nb <= p.idx:
                continue
            if a[nb] == a[p.idx]:
                continue
            other = grid.precincts[nb]
            shared_segments = _shared_segments(p.geometry, other.geometry)  # type: ignore[attr-defined]
            for seg in shared_segments:
                xs = [pt[0] for pt in seg]
                ys = [pt[1] for pt in seg]
                ax.plot(xs, ys, color="black", linewidth=1.8, solid_capstyle="round")

    # State outline: draw the union of all counties' outer rings minus internal edges.
    # Simpler: outline counties on the state boundary by detecting edges not shared with any neighbor.
    for p in grid.precincts:
        nb_geoms = [grid.precincts[nb].geometry for nb in grid.neighbors(p.idx)]  # type: ignore[attr-defined]
        for poly in p.geometry:  # type: ignore[attr-defined]
            outer = poly[0]
            for i in range(len(outer) - 1):
                seg = (outer[i], outer[i + 1])
                if not _segment_shared_with_any(seg, nb_geoms):
                    ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                            color="black", linewidth=1.4)

    # District labels at centroid (population-weighted).
    districts = plan.districts()
    votes = plan.district_d_votes()
    demos = plan.district_demographics()
    for d, cells in enumerate(districts):
        if not cells:
            continue
        wx = wy = wp = 0.0
        for i in cells:
            pp = grid.precincts[i]
            wx += pp.centroid[0] * pp.population  # type: ignore[attr-defined]
            wy += pp.centroid[1] * pp.population  # type: ignore[attr-defined]
            wp += pp.population
        cx = wx / wp
        cy = wy / wp
        d_votes, r_votes = votes[d]
        tot = d_votes + r_votes
        d_pct = 100.0 * d_votes / tot if tot else 0.0
        top_demo = sorted(demos[d].items(), key=lambda kv: -kv[1])[:2]
        demo_str = ", ".join(f"{k[:1].upper()}{int(v*100)}" for k, v in top_demo)
        label = f"D{d+1}\nD{d_pct:.0f}/R{100-d_pct:.0f}\n{demo_str}"
        ax.text(cx, cy, label, ha="center", va="center", fontsize=6.0,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.78, edgecolor="none"))


import math


def _segments_of(geom) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    segs = []
    for poly in geom:
        for ring in poly:
            for i in range(len(ring) - 1):
                a = (round(ring[i][0], 5), round(ring[i][1], 5))
                b = (round(ring[i + 1][0], 5), round(ring[i + 1][1], 5))
                if a == b:
                    continue
                segs.append((a, b) if a < b else (b, a))
    return segs


def _shared_segments(geom_a, geom_b) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    sa = set(_segments_of(geom_a))
    sb = set(_segments_of(geom_b))
    return [s for s in sa if s in sb]


def _segment_shared_with_any(seg, nb_geoms) -> bool:
    a = (round(seg[0][0], 5), round(seg[0][1], 5))
    b = (round(seg[1][0], 5), round(seg[1][1], 5))
    key = (a, b) if a < b else (b, a)
    for g in nb_geoms:
        if key in set(_segments_of(g)):
            return True
    return False


def _draw_plan(ax: plt.Axes, plan: DistrictPlan, view: str = "party", title: str = "") -> None:
    if _is_real_geo(plan):
        _draw_plan_real(ax, plan, view, title)
    else:
        _draw_plan_grid(ax, plan, view, title)


def _legend_for(view: str) -> List[Line2D]:
    if view == "party":
        return [
            Line2D([0], [0], marker="s", color="w", markerfacecolor=(0.2, 0.3, 0.8), markersize=10, label="Democratic lean"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=(0.8, 0.3, 0.2), markersize=10, label="Republican lean"),
        ]
    return [Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=k.capitalize())
            for k, c in DEMO_COLORS.items()]


def render_comparison(
    neutral_plan: DistrictPlan,
    gerrymander_plan: DistrictPlan,
    out_path: str,
    view: str = "party",
    state: str = "",
    target_party: str = "R",
) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.2], hspace=0.3, wspace=0.15)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")

    _draw_plan(ax_left, neutral_plan, view=view, title=f"Neutral baseline — {neutral_plan.label}")
    _draw_plan(ax_right, gerrymander_plan, view=view, title=f"Gerrymandered ({target_party}) — {gerrymander_plan.label}")

    fig.suptitle(
        f"Educational gerrymandering demo — {state}  (view: {view})",
        fontsize=14, fontweight="bold",
    )
    fig.text(0.5, 0.94,
             "For education and detection only. Synthetic precinct data.",
             ha="center", fontsize=9, style="italic", color="#555555")

    leg = _legend_for(view)
    ax_left.legend(handles=leg, loc="upper right", fontsize=7, framealpha=0.8)

    # Metrics table.
    s_n = summarize(neutral_plan)
    s_g = summarize(gerrymander_plan)
    rows = [
        ("Statewide D vote share", f"{s_n['D_vote_share']*100:.1f}%", f"{s_g['D_vote_share']*100:.1f}%"),
        ("D seats / total", f"{int(s_n['D_seats'])} / {int(s_n['n_districts'])}", f"{int(s_g['D_seats'])} / {int(s_g['n_districts'])}"),
        ("D seat share", f"{s_n['D_seat_share']*100:.1f}%", f"{s_g['D_seat_share']*100:.1f}%"),
        ("Efficiency gap (+R / -D)", f"{s_n['efficiency_gap']*100:+.2f}%", f"{s_g['efficiency_gap']*100:+.2f}%"),
        ("Mean - Median (D)", f"{s_n['mean_median_D']*100:+.2f}%", f"{s_g['mean_median_D']*100:+.2f}%"),
        ("Partisan bias (D)", f"{s_n['partisan_bias_D']*100:+.2f}%", f"{s_g['partisan_bias_D']*100:+.2f}%"),
    ]
    table = ax_table.table(
        cellText=[list(r) for r in rows],
        colLabels=["Metric", "Neutral", "Gerrymandered"],
        loc="center",
        cellLoc="center",
        colWidths=[0.4, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_single(plan: DistrictPlan, out_path: str, view: str = "party", title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_plan(ax, plan, view=view, title=title or plan.label)
    ax.legend(handles=_legend_for(view), loc="upper right", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
