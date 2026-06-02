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
from matplotlib.patches import Rectangle
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


def _draw_plan(
    ax: plt.Axes,
    plan: DistrictPlan,
    view: str = "party",
    title: str = "",
) -> None:
    grid = plan.grid
    ax.set_xlim(-0.5, grid.cols - 0.5)
    ax.set_ylim(grid.rows - 0.5, -0.5)  # invert so row 0 is at top
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

    # District boundary lines: thick black between cells of different districts.
    a = plan.assignment
    for p in grid.precincts:
        for nb in grid.neighbors(p.idx):
            if a[nb] == a[p.idx]:
                continue
            np_ = grid.precincts[nb]
            # Draw line between p and np_.
            if np_.row == p.row + 1:  # neighbor below
                ax.plot([p.col - 0.5, p.col + 0.5], [p.row + 0.5, p.row + 0.5], color="black", linewidth=1.6)
            elif np_.col == p.col + 1:  # neighbor right
                ax.plot([p.col + 0.5, p.col + 0.5], [p.row - 0.5, p.row + 0.5], color="black", linewidth=1.6)
    # Outer border.
    ax.add_patch(Rectangle((-0.5, -0.5), grid.cols, grid.rows, fill=False, edgecolor="black", linewidth=1.6))

    # District labels at centroid.
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
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"))


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
