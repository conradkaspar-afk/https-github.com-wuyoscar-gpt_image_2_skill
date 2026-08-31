"""District assignment algorithms.

All functions return a `DistrictPlan`: a list of districts where each
district is a list of precinct indices. Plans are contiguous and
population-balanced within `pop_tolerance` of the ideal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import heapq

import numpy as np

from .state_model import StateGrid


DistrictAssignment = List[int]  # assignment[precinct_idx] = district_id


@dataclass
class DistrictPlan:
    grid: StateGrid
    assignment: DistrictAssignment
    label: str = "plan"

    @property
    def num_districts(self) -> int:
        return max(self.assignment) + 1

    def districts(self) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in range(self.num_districts)]
        for i, d in enumerate(self.assignment):
            out[d].append(i)
        return out

    def district_population(self) -> List[int]:
        pops = [0] * self.num_districts
        for i, d in enumerate(self.assignment):
            pops[d] += self.grid.precincts[i].population
        return pops

    def district_d_votes(self) -> List[Tuple[int, int]]:
        """For each district, returns (D_votes, R_votes) as integers."""
        out = [(0, 0)] * self.num_districts
        agg: List[List[float]] = [[0.0, 0.0] for _ in range(self.num_districts)]
        for i, d in enumerate(self.assignment):
            p = self.grid.precincts[i]
            agg[d][0] += p.d_share * p.population
            agg[d][1] += (1.0 - p.d_share) * p.population
        return [(int(a[0]), int(a[1])) for a in agg]

    def district_demographics(self) -> List[Dict[str, float]]:
        keys = list(self.grid.precincts[0].demographics.keys())
        agg = [{k: 0.0 for k in keys} for _ in range(self.num_districts)]
        pops = [0.0] * self.num_districts
        for i, d in enumerate(self.assignment):
            p = self.grid.precincts[i]
            for k in keys:
                agg[d][k] += p.demographics[k] * p.population
            pops[d] += p.population
        for d in range(self.num_districts):
            if pops[d] > 0:
                for k in keys:
                    agg[d][k] /= pops[d]
        return agg


def _precinct_coords(grid: StateGrid) -> "np.ndarray":
    """Use centroid if precincts carry one (real-geo), otherwise grid (row, col)."""
    if grid.precincts and hasattr(grid.precincts[0], "centroid"):
        return np.array([p.centroid for p in grid.precincts], dtype=float)  # type: ignore[attr-defined]
    return np.array([(p.row, p.col) for p in grid.precincts], dtype=float)


def _seed_kmeans_pp(grid: StateGrid, n: int, rng: np.random.Generator) -> List[int]:
    coords = _precinct_coords(grid)
    first = int(rng.integers(0, grid.n))
    seeds = [first]
    dists = ((coords - coords[first]) ** 2).sum(axis=1)
    for _ in range(n - 1):
        probs = dists / dists.sum()
        choice = int(rng.choice(grid.n, p=probs))
        seeds.append(choice)
        new_d = ((coords - coords[choice]) ** 2).sum(axis=1)
        dists = np.minimum(dists, new_d)
    return seeds


def _grow_regions(
    grid: StateGrid,
    seeds: Sequence[int],
    forbidden: Optional[Dict[int, set]] = None,
) -> DistrictAssignment:
    """Multi-source BFS growth to a complete contiguous assignment.

    `forbidden[district_id]` is a set of precincts that district may not absorb.
    Priority weights by current district population to keep things balanced.
    """
    n = grid.n
    assignment: List[int] = [-1] * n
    pops = [0] * len(seeds)
    target = grid.total_population() / len(seeds)
    forbidden = forbidden or {}

    heap: List[Tuple[float, int, int, int]] = []
    counter = 0
    for d, s in enumerate(seeds):
        assignment[s] = d
        pops[d] = grid.precincts[s].population
        for nb in grid.neighbors(s):
            if assignment[nb] == -1 and nb not in forbidden.get(d, set()):
                heapq.heappush(heap, (pops[d] / target, counter, d, nb))
                counter += 1

    while heap:
        _, _, d, nb = heapq.heappop(heap)
        if assignment[nb] != -1:
            continue
        if nb in forbidden.get(d, set()):
            continue
        assignment[nb] = d
        pops[d] += grid.precincts[nb].population
        for nn in grid.neighbors(nb):
            if assignment[nn] == -1 and nn not in forbidden.get(d, set()):
                heapq.heappush(heap, (pops[d] / target, counter, d, nn))
                counter += 1

    # Repair any unassigned cells (can happen when forbidden masks isolate them).
    for i in range(n):
        if assignment[i] == -1:
            # Attach to any neighboring district with smallest pop.
            cand = [assignment[nb] for nb in grid.neighbors(i) if assignment[nb] != -1]
            if cand:
                pick = min(cand, key=lambda d: pops[d])
                assignment[i] = pick
                pops[pick] += grid.precincts[i].population
            else:
                # Isolated — assign to district 0.
                assignment[i] = 0
                pops[0] += grid.precincts[i].population
    return assignment


def _is_contiguous(grid: StateGrid, assignment: DistrictAssignment, d: int) -> bool:
    cells = [i for i, a in enumerate(assignment) if a == d]
    if not cells:
        return True
    seen = {cells[0]}
    stack = [cells[0]]
    while stack:
        cur = stack.pop()
        for nb in grid.neighbors(cur):
            if assignment[nb] == d and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _components(grid: StateGrid, cells: List[int], assignment: DistrictAssignment,
                d: int) -> List[List[int]]:
    """Connected components of district `d` within the precinct graph."""
    remaining = set(cells)
    out: List[List[int]] = []
    while remaining:
        start = next(iter(remaining))
        comp = [start]
        remaining.discard(start)
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in grid.neighbors(cur):
                if nb in remaining and assignment[nb] == d:
                    remaining.discard(nb)
                    comp.append(nb)
                    stack.append(nb)
        out.append(comp)
    return out


def _repair_plan(grid: StateGrid, assignment: DistrictAssignment,
                 n: int) -> DistrictAssignment:
    """Enforce the plan invariants: every district non-empty and contiguous.

    Construction on real county graphs can strand fragments (islands joined by
    synthetic adjacency edges, very uneven county sizes) or leave a district
    with no cells. Rather than special-case every construction path, this runs
    last and repairs whatever it is given:

      1. Each district keeps only its largest connected component; cells in
         smaller fragments are re-attached to an adjacent district.
      2. Any empty district is refilled by splitting a contiguous chunk off the
         most-populous district, choosing only cells whose removal leaves that
         district connected.
    """
    # --- 1. Strand-free: keep the largest component of each district. --------
    by_district: Dict[int, List[int]] = {}
    for i, d in enumerate(assignment):
        by_district.setdefault(d, []).append(i)

    orphans: List[int] = []
    for d, cells in by_district.items():
        comps = _components(grid, cells, assignment, d)
        if len(comps) <= 1:
            continue
        comps.sort(key=lambda c: sum(grid.precincts[i].population for i in c), reverse=True)
        for comp in comps[1:]:
            orphans.extend(comp)
    for i in orphans:
        assignment[i] = -1

    # Re-attach orphans to an adjacent district, smallest population first.
    pops = [0] * n
    for i, d in enumerate(assignment):
        if d >= 0:
            pops[d] += grid.precincts[i].population
    pending = list(orphans)
    while pending:
        progressed = False
        still: List[int] = []
        for i in pending:
            cand = {assignment[nb] for nb in grid.neighbors(i) if assignment[nb] >= 0}
            if not cand:
                still.append(i)
                continue
            pick = min(cand, key=lambda d: pops[d])
            assignment[i] = pick
            pops[pick] += grid.precincts[i].population
            progressed = True
        if not progressed:
            # Fully disconnected leftovers: attach to the smallest district.
            fallback = min(range(n), key=lambda d: pops[d])
            for i in still:
                assignment[i] = fallback
                pops[fallback] += grid.precincts[i].population
            break
        pending = still

    # --- 2. Refill empty districts by splitting the largest one. -------------
    for d in range(n):
        if any(a == d for a in assignment):
            continue
        # Donate from a district that actually has cells to spare. Ranking by
        # population alone can pick a single huge county, which cannot be split.
        cells_by_district: Dict[int, List[int]] = {}
        for i, a in enumerate(assignment):
            if a >= 0:
                cells_by_district.setdefault(a, []).append(i)
        splittable = [x for x, cs in cells_by_district.items() if len(cs) >= 2]
        if not splittable:
            continue  # every district is a single unit; leave as-is
        donor = max(splittable, key=lambda x: (
            len(cells_by_district[x]),
            sum(grid.precincts[i].population for i in cells_by_district[x]),
        ))
        donor_cells = cells_by_district[donor]
        # Start from the donor cell furthest from the donor's centre of mass so
        # the piece we peel off sits on the district's edge.
        cx = sum(grid.precincts[i].row for i in donor_cells) / len(donor_cells)
        cy = sum(grid.precincts[i].col for i in donor_cells) / len(donor_cells)
        start = max(donor_cells, key=lambda i:
                    (grid.precincts[i].row - cx) ** 2 + (grid.precincts[i].col - cy) ** 2)
        target = max(1, len(donor_cells) // 2)
        accepted: set = set()
        frontier = [start]
        seen = {start}
        while frontier and len(accepted) < target:
            cell = frontier.pop(0)
            # Enqueue neighbours before attempting the move: a rejected move
            # must not end the search, or a district whose first candidate
            # fails the contiguity check would never be filled at all.
            for nb in grid.neighbors(cell):
                if nb not in seen and assignment[nb] == donor:
                    seen.add(nb)
                    frontier.append(nb)
            if assignment[cell] != donor:
                continue
            # The new district must itself stay in one piece, so only take a
            # cell touching what we have already taken.
            if accepted and not any(nb in accepted for nb in grid.neighbors(cell)):
                continue
            if sum(1 for a in assignment if a == donor) <= 1:
                break
            assignment[cell] = d
            # Only keep the move if the donor stays in one piece too.
            if not _is_contiguous(grid, assignment, donor):
                assignment[cell] = donor
                continue
            accepted.add(cell)
    return assignment


def _rebalance(
    grid: StateGrid,
    assignment: DistrictAssignment,
    n_districts: int,
    tolerance: float = 0.05,
    max_passes: int = 2000,
) -> DistrictAssignment:
    """Iteratively move border precincts to reduce population variance.

    At each pass: pick any boundary precinct whose move from its current
    over-populated district to an under-populated neighbor reduces total
    variance, and apply it if it preserves contiguity.
    """
    pops = [0] * n_districts
    for i, d in enumerate(assignment):
        pops[d] += grid.precincts[i].population
    ideal = grid.total_population() / n_districts

    for _ in range(max_passes):
        spread = (max(pops) - min(pops)) / ideal
        if spread < tolerance:
            break
        # Collect all candidate moves: (improvement, cell, src, dst).
        candidates: List[Tuple[float, int, int, int]] = []
        for i, src in enumerate(assignment):
            pop_i = grid.precincts[i].population
            if pops[src] - pop_i <= 0:
                continue  # don't empty a district
            for nb in grid.neighbors(i):
                dst = assignment[nb]
                if dst == src:
                    continue
                if pops[src] <= pops[dst]:
                    continue
                # Variance reduction proxy: square error before vs after.
                before = (pops[src] - ideal) ** 2 + (pops[dst] - ideal) ** 2
                after = (pops[src] - pop_i - ideal) ** 2 + (pops[dst] + pop_i - ideal) ** 2
                gain = before - after
                if gain > 0:
                    candidates.append((-gain, i, src, dst))
        if not candidates:
            break
        candidates.sort()
        moved = False
        for _, i, src, dst in candidates[:50]:
            if assignment[i] != src:
                continue
            assignment[i] = dst
            if _is_contiguous(grid, assignment, src):
                pop_i = grid.precincts[i].population
                pops[src] -= pop_i
                pops[dst] += pop_i
                moved = True
                break
            assignment[i] = src
        if not moved:
            break
    return assignment


def neutral_districts(grid: StateGrid, n: Optional[int] = None, seed: int = 0) -> DistrictPlan:
    """Compact baseline: k-means++ seeds + region growth + population rebalancing."""
    if n is None:
        n = grid.num_districts
    rng = np.random.default_rng(seed)
    seeds = _seed_kmeans_pp(grid, n, rng)
    assignment = _grow_regions(grid, seeds)
    assignment = _rebalance(grid, assignment, n)
    assignment = _repair_plan(grid, assignment, n)
    return DistrictPlan(grid=grid, assignment=assignment, label="neutral")


def random_districts(grid: StateGrid, n: Optional[int] = None, seed: int = 0) -> DistrictPlan:
    """Control: random seeds + growth (no compactness preference)."""
    if n is None:
        n = grid.num_districts
    rng = np.random.default_rng(seed + 999)
    seeds = list(rng.choice(grid.n, size=n, replace=False))
    assignment = _grow_regions(grid, seeds)
    assignment = _rebalance(grid, assignment, n)
    assignment = _repair_plan(grid, assignment, n)
    return DistrictPlan(grid=grid, assignment=assignment, label="random")


def pack_and_crack(
    grid: StateGrid,
    n: Optional[int] = None,
    target_party: str = "R",
    intensity: float = 0.7,
    seed: int = 0,
) -> DistrictPlan:
    """Gerrymander in favor of `target_party` using packing and cracking.

    Two-phase greedy:
      1. **Pack**: starting from the highest-opponent precincts, grow `k_pack`
         contiguous districts that gobble adjacent opponent-leaning precincts
         until each reaches ~target population. This concentrates opponent
         voters into very few "wasted" wins.
      2. **Crack**: grow the remaining `n - k_pack` districts via standard
         region growth over the leftover precincts. Because the leftover pool
         is now skewed toward the target party, each crack district wins by
         a comfortable margin.

    `intensity` (0..1) scales how many districts are packed and how
    aggressively pack districts pursue opponent cells.
    """
    if n is None:
        n = grid.num_districts
    if target_party not in ("D", "R"):
        raise ValueError("target_party must be 'D' or 'R'")
    if n <= 1:
        # Single-district states (e.g. AK, DE, ND, SD, VT, WY): nothing to gerrymander.
        return DistrictPlan(
            grid=grid, assignment=[0] * grid.n,
            label=f"gerrymander({target_party}, intensity={intensity:.2f}) [single district]",
        )
    rng = np.random.default_rng(seed + 12345)

    if target_party == "R":
        opp = np.array([p.d_share for p in grid.precincts])
    else:
        opp = np.array([1.0 - p.d_share for p in grid.precincts])

    opp_share = float(((opp * np.array([p.population for p in grid.precincts])).sum())
                      / sum(p.population for p in grid.precincts))
    proportional = int(round(opp_share * n))
    # Pack count: at intensity=1, halve proportional opponent seats (everyone packed in fewer districts).
    k_pack = max(1, int(round(proportional * (1.0 - 0.5 * intensity))))
    if proportional <= 1:
        k_pack = max(1, proportional)
    k_pack = min(k_pack, max(1, n - 2))

    target_pop = grid.total_population() / n

    # --- Pack phase ---
    # Seeds: the highest-opp cells, spatially separated.
    coords = _precinct_coords(grid)
    order = np.argsort(-opp)  # descending opp
    pack_seeds: List[int] = []
    extent = float(coords.max(axis=0).sum() - coords.min(axis=0).sum())
    min_sep = max(0.5, extent / (2 * k_pack + 2))
    for idx in order:
        if all(((coords[idx] - coords[s]) ** 2).sum() >= min_sep ** 2 for s in pack_seeds):
            pack_seeds.append(int(idx))
            if len(pack_seeds) == k_pack:
                break
    while len(pack_seeds) < k_pack:
        # Fallback: top-opp regardless of separation.
        for idx in order:
            if int(idx) not in pack_seeds:
                pack_seeds.append(int(idx))
                break

    assignment: List[int] = [-1] * grid.n
    pops = [0] * n

    # Claim every pack seed BEFORE any growth. Growing districts one at a time
    # while seeds were still unclaimed let an earlier district absorb a later
    # district's seed cell; re-seeding then punched a hole in the earlier
    # district and stranded the later one as an island, so both ended up
    # discontiguous.
    for d, s in enumerate(pack_seeds):
        assignment[s] = d
        pops[d] = grid.precincts[s].population

    # Grow the pack districts round-robin so no single district monopolizes the
    # strongest opponent cells, each taking its highest-opponent adjacent cell.
    frontiers: List[List[Tuple[float, int, int]]] = [[] for _ in range(k_pack)]
    counter = 0
    for d, s in enumerate(pack_seeds):
        for nb in grid.neighbors(s):
            if assignment[nb] == -1:
                heapq.heappush(frontiers[d], (-float(opp[nb]), counter, nb))
                counter += 1

    active = set(range(k_pack))
    while active:
        for d in sorted(active):
            frontier = frontiers[d]
            if pops[d] >= target_pop:
                active.discard(d)
                continue
            grew = False
            while frontier:
                _, _, nb = heapq.heappop(frontier)
                if assignment[nb] != -1:
                    continue
                # At low intensity, become picky and stop absorbing weak cells.
                if opp[nb] < 0.5 and pops[d] > 0.5 * target_pop and intensity < 0.95:
                    # Only keep absorbing weak cells if district isn't yet ~target.
                    if pops[d] > target_pop * (1.0 - 0.2 * intensity):
                        continue
                assignment[nb] = d
                pops[d] += grid.precincts[nb].population
                for nn in grid.neighbors(nb):
                    if assignment[nn] == -1:
                        heapq.heappush(frontier, (-float(opp[nn]), counter, nn))
                        counter += 1
                grew = True
                break
            if not grew:
                active.discard(d)

    # --- Crack phase ---
    # Remaining cells: grow n - k_pack districts via k-means++ on remaining cells,
    # then region-grow respecting "may not enter pack districts" forbidden mask.
    remaining = [i for i in range(grid.n) if assignment[i] == -1]
    if remaining:
        # Pick crack seeds among remaining cells weighted toward target-friendly territory.
        friendly = (1.0 - opp[remaining])
        fweights = (friendly ** 2) + 1e-6
        fweights /= fweights.sum()
        first = int(rng.choice(remaining, p=fweights))
        crack_seeds = [first]
        taken = {first}
        rem_coords = coords[remaining]
        dists = ((rem_coords - coords[first]) ** 2).sum(axis=1)
        n_crack = n - k_pack
        for _ in range(n_crack - 1):
            probs = dists * fweights
            # Zero out cells already used as a seed: rng.choice samples with
            # replacement, and a duplicate seed would leave one district with no
            # cells of its own while stranding the other.
            for j, cell in enumerate(remaining):
                if cell in taken:
                    probs[j] = 0.0
            if probs.sum() <= 0:
                # Fall back to any unused remaining cell.
                free = [c for c in remaining if c not in taken]
                if not free:
                    break
                nxt = int(free[0])
            else:
                probs = probs / probs.sum()
                nxt = int(rng.choice(remaining, p=probs))
            crack_seeds.append(nxt)
            taken.add(nxt)
            nd = ((rem_coords - coords[nxt]) ** 2).sum(axis=1)
            dists = np.minimum(dists, nd)

        # Region-grow crack districts (ids k_pack..n-1), forbidden from pack territory.
        for ci, s in enumerate(crack_seeds):
            d = k_pack + ci
            assignment[s] = d
            pops[d] = grid.precincts[s].population

        heap: List[Tuple[float, int, int, int]] = []
        counter = 0
        for ci, s in enumerate(crack_seeds):
            d = k_pack + ci
            for nb in grid.neighbors(s):
                if assignment[nb] == -1:
                    heapq.heappush(heap, (pops[d] / target_pop, counter, d, nb))
                    counter += 1
        while heap:
            _, _, d, nb = heapq.heappop(heap)
            if assignment[nb] != -1:
                continue
            assignment[nb] = d
            pops[d] += grid.precincts[nb].population
            for nn in grid.neighbors(nb):
                if assignment[nn] == -1:
                    heapq.heappush(heap, (pops[d] / target_pop, counter, d, nn))
                    counter += 1

    # Repair: any cell still unassigned (rare) joins smallest neighboring district.
    for i in range(grid.n):
        if assignment[i] == -1:
            cand = [assignment[nb] for nb in grid.neighbors(i) if assignment[nb] != -1]
            if cand:
                pick = min(cand, key=lambda d: pops[d])
                assignment[i] = pick
                pops[pick] += grid.precincts[i].population
            else:
                assignment[i] = 0
                pops[0] += grid.precincts[i].population

    # Rebalance with looser tolerance to preserve the gerrymander signal.
    assignment = _rebalance(grid, assignment, n, tolerance=0.08)

    # Local-search seat optimizer on the redrawn plan.
    redraw_assn = _optimize_seats(grid, list(assignment), n, target_party, max_passes=400)

    # Also try starting from a compact neutral plan and optimizing — guarantees
    # the result is never worse than the neutral baseline for the target party.
    # `n` must be passed through: without it this builds the state's default
    # district count, whose ids then overflow the n-sized arrays in
    # _optimize_seats whenever the caller asked for a different seat count.
    neutral_assn = neutral_districts(grid, n=n, seed=seed).assignment
    from_neutral = _optimize_seats(grid, list(neutral_assn), n, target_party, max_passes=400)

    redraw_seats = _count_target_seats(grid, redraw_assn, n, target_party)
    neutral_seats = _count_target_seats(grid, from_neutral, n, target_party)
    chosen = redraw_assn if redraw_seats >= neutral_seats else from_neutral

    label = f"gerrymander({target_party}, intensity={intensity:.2f})"
    chosen = _repair_plan(grid, list(chosen), n)
    return DistrictPlan(grid=grid, assignment=chosen, label=label)


def _count_target_seats(grid: StateGrid, assignment: DistrictAssignment, n: int, target_party: str) -> int:
    d_votes = [0.0] * n
    pop_d = [0.0] * n
    for i, a in enumerate(assignment):
        p = grid.precincts[i]
        d_votes[a] += p.d_share * p.population
        pop_d[a] += p.population
    count = 0
    for d in range(n):
        if pop_d[d] == 0:
            continue
        share = d_votes[d] / pop_d[d]
        if target_party == "D" and share > 0.5:
            count += 1
        elif target_party == "R" and share < 0.5:
            count += 1
    return count


def _district_d_share(grid: StateGrid, assignment: DistrictAssignment, d: int) -> float:
    dv = 0.0
    pv = 0.0
    for i, a in enumerate(assignment):
        if a != d:
            continue
        p = grid.precincts[i]
        dv += p.d_share * p.population
        pv += p.population
    return dv / pv if pv > 0 else 0.0


def _optimize_seats(
    grid: StateGrid,
    assignment: DistrictAssignment,
    n: int,
    target_party: str,
    max_passes: int = 400,
    pop_tol: float = 0.15,
) -> DistrictAssignment:
    """Hill-climb on border swaps to increase target-party seats won.

    Constraints: contiguity preserved, population spread <= pop_tol.
    """
    pops = [0] * n
    for i, d in enumerate(assignment):
        pops[d] += grid.precincts[i].population
    ideal = grid.total_population() / n

    # D-share per district (cached).
    d_votes = [0.0] * n
    pop_d = [0.0] * n
    for i, d in enumerate(assignment):
        p = grid.precincts[i]
        d_votes[d] += p.d_share * p.population
        pop_d[d] += p.population

    def seat_value(d: int) -> int:
        share = d_votes[d] / pop_d[d] if pop_d[d] > 0 else 0.5
        if target_party == "R":
            return 1 if share < 0.5 else 0
        return 1 if share > 0.5 else 0

    cur_seats = sum(seat_value(d) for d in range(n))

    for _ in range(max_passes):
        # Collect all positive-gain moves; try them in order, accept the first
        # that preserves contiguity. Also allow gain==0 moves that improve a
        # district's margin toward 50% (so we set up future flips).
        candidates: List[Tuple[float, int, int, int]] = []
        for i in range(grid.n):
            src = assignment[i]
            p = grid.precincts[i]
            for nb in grid.neighbors(i):
                dst = assignment[nb]
                if dst == src:
                    continue
                new_src = pops[src] - p.population
                new_dst = pops[dst] + p.population
                if new_src <= 0:
                    continue
                other = [pops[k] for k in range(n) if k != src and k != dst]
                hi = max([new_src, new_dst] + other)
                lo = min([new_src, new_dst] + other)
                if (hi - lo) / ideal > pop_tol:
                    continue
                new_d_src = d_votes[src] - p.d_share * p.population
                new_pd_src = pop_d[src] - p.population
                new_d_dst = d_votes[dst] + p.d_share * p.population
                new_pd_dst = pop_d[dst] + p.population
                share_src = new_d_src / new_pd_src if new_pd_src > 0 else 0.5
                share_dst = new_d_dst / new_pd_dst if new_pd_dst > 0 else 0.5
                if target_party == "R":
                    new_seat_src = 1 if share_src < 0.5 else 0
                    new_seat_dst = 1 if share_dst < 0.5 else 0
                else:
                    new_seat_src = 1 if share_src > 0.5 else 0
                    new_seat_dst = 1 if share_dst > 0.5 else 0
                gain = (new_seat_src + new_seat_dst) - (seat_value(src) + seat_value(dst))
                if gain <= 0:
                    continue
                candidates.append((-gain, i, src, dst))
        if not candidates:
            break
        candidates.sort()
        applied = False
        for neg_gain, i, src, dst in candidates:
            if assignment[i] != src:
                continue
            p = grid.precincts[i]
            assignment[i] = dst
            if not _is_contiguous(grid, assignment, src):
                assignment[i] = src
                continue
            pops[src] -= p.population
            pops[dst] += p.population
            d_votes[src] -= p.d_share * p.population
            pop_d[src] -= p.population
            d_votes[dst] += p.d_share * p.population
            pop_d[dst] += p.population
            cur_seats += -neg_gain
            applied = True
            break
        if not applied:
            break
    return assignment
