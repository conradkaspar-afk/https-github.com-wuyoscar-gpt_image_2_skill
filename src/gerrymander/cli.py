"""Command-line entry point for the gerrymandering visualizer."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .state_model import build_state, list_states
from .districting import neutral_districts, pack_and_crack
from .metrics import summarize
from .visualize import render_comparison, render_single


BANNER = (
    "gerrymander — educational visualizer\n"
    "  Synthetic precinct data. For demonstration & detection only.\n"
)


def _print_metrics(name: str, plan) -> None:
    s = summarize(plan)
    print(f"\n[{name}]  {plan.label}")
    print(f"  D vote share:    {s['D_vote_share']*100:6.2f}%")
    print(f"  D seats:         {int(s['D_seats'])}/{int(s['n_districts'])}  "
          f"({s['D_seat_share']*100:.1f}%)")
    print(f"  efficiency gap:  {s['efficiency_gap']*100:+6.2f}%  (+R / -D)")
    print(f"  mean - median D: {s['mean_median_D']*100:+6.2f}%")
    print(f"  partisan bias D: {s['partisan_bias_D']*100:+6.2f}%")


def _explain(neutral, gerry, target_party: str) -> str:
    s_n = summarize(neutral)
    s_g = summarize(gerry)
    delta_seats = int(s_g[f"{target_party}_seats"] - s_n[f"{target_party}_seats"])
    eg_shift = (s_g["efficiency_gap"] - s_n["efficiency_gap"]) * 100
    return (
        "Explanation\n"
        "-----------\n"
        f"Target party: {target_party}. The pack-and-crack algorithm reserved a few\n"
        "districts to absorb opponent strongholds (packing), then distributed the\n"
        "remaining opponent voters thinly across the other districts so that the\n"
        f"target party wins each with a comfortable margin (cracking).\n\n"
        f"Net effect vs neutral baseline: {delta_seats:+d} extra {target_party} seats; "
        f"efficiency gap shifted by {eg_shift:+.2f} pp.\n"
        "A value above ~7% on efficiency gap is widely cited as evidence of\n"
        "an aggressive partisan gerrymander.\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gerrymander",
        description="Generate and visualize an educational gerrymandered district map.",
    )
    parser.add_argument("--state", help="Two-letter US state code (e.g. TX, CA, PA).")
    parser.add_argument("--party", choices=["D", "R"], default="R",
                        help="Party to favor with gerrymandering (default: R).")
    parser.add_argument("--intensity", type=float, default=0.7,
                        help="Gerrymandering intensity, 0.0 - 1.0 (default 0.7).")
    parser.add_argument("--view", choices=["party", "demographic"], default="party",
                        help="Coloring scheme for precincts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", default=None, help="Output PNG path.")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Render only the gerrymandered map (skip baseline).")
    parser.add_argument("--explain", action="store_true",
                        help="Print a short explanation of the result.")
    parser.add_argument("--list-states", action="store_true",
                        help="Print supported state codes and exit.")
    args = parser.parse_args(argv)

    if args.list_states:
        print(" ".join(list_states()))
        return 0

    if not args.state:
        parser.error("--state is required (or use --list-states)")

    print(BANNER)
    grid = build_state(args.state, seed=args.seed)
    print(f"State: {grid.state}  precincts={grid.n}  districts={grid.num_districts}  "
          f"pop={grid.total_population():,}  baseline D lean={grid.d_lean:.2f}")

    neutral = neutral_districts(grid, seed=args.seed)
    gerry = pack_and_crack(grid, target_party=args.party, intensity=args.intensity, seed=args.seed)

    _print_metrics("Neutral baseline", neutral)
    _print_metrics("Gerrymandered", gerry)

    if args.explain:
        print()
        print(_explain(neutral, gerry, args.party))

    out = args.out or f"out/{grid.state.lower()}_{args.party.lower()}_{args.view}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if args.no_comparison:
        render_single(gerry, out, view=args.view,
                      title=f"{grid.state} — gerrymandered ({args.party})")
    else:
        render_comparison(neutral, gerry, out, view=args.view,
                          state=grid.state, target_party=args.party)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
