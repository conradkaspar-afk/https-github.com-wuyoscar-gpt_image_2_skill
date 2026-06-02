"""ReCom-based gerrymander search over a precinct adjacency graph."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import geopandas as gpd
from gerrychain import Graph, Partition, MarkovChain, constraints
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import Tally, cut_edges
from gerrychain.accept import always_accept

from . import metrics

log = logging.getLogger(__name__)

POP_TOLERANCE = 0.02  # ±2 % of ideal district population


@dataclass
class BestPlan:
    assignment: dict[int, int]
    metrics: dict
    step: int


def build_graph(gdf: gpd.GeoDataFrame) -> Graph:
    return Graph.from_geodataframe(gdf, ignore_errors=True)


def _district_votes(partition: Partition) -> dict[int, tuple[float, float]]:
    return {d: (partition["dem"][d], partition["rep"][d]) for d in partition.parts}


def _polsby_popper_mean(partition: Partition) -> float:
    # Sum geometry per part.
    geoms = {}
    for node, part in partition.assignment.items():
        g = partition.graph.nodes[node].get("geometry")
        if g is None:
            continue
        geoms.setdefault(part, []).append(g)
    if not geoms:
        return 0.0
    from shapely.ops import unary_union
    vals = []
    for part, gs in geoms.items():
        u = unary_union(gs)
        vals.append(metrics.polsby_popper(u.area, u.length))
    return sum(vals) / len(vals) if vals else 0.0


def _score(partition: Partition, objective: str) -> float:
    votes = _district_votes(partition)
    dem, rep = metrics.seats_won(votes)
    eg = metrics.efficiency_gap(votes)
    if objective == "pro_dem":
        return dem + max(eg, 0.0) * 0.1  # tiebreak on positive EG
    if objective == "pro_rep":
        return rep + max(-eg, 0.0) * 0.1
    if objective == "compact":
        return _polsby_popper_mean(partition)
    raise ValueError(f"unknown objective: {objective}")


def seed_partition(graph: Graph, num_districts: int) -> Partition:
    total_pop = sum(graph.nodes[n]["pop"] for n in graph.nodes)
    ideal_pop = total_pop / num_districts
    assignment = recursive_tree_part(
        graph,
        parts=range(num_districts),
        pop_target=ideal_pop,
        pop_col="pop",
        epsilon=POP_TOLERANCE,
        node_repeats=2,
    )
    return Partition(
        graph,
        assignment=assignment,
        updaters={
            "population": Tally("pop", alias="population"),
            "dem": Tally("dem", alias="dem"),
            "rep": Tally("rep", alias="rep"),
            "cut_edges": cut_edges,
        },
    )


def run_chain(
    seed: Partition,
    objective: str,
    steps: int,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> BestPlan:
    total_pop = sum(seed["population"].values())
    ideal_pop = total_pop / len(seed.parts)

    proposal = partial(
        recom,
        pop_col="pop",
        pop_target=ideal_pop,
        epsilon=POP_TOLERANCE,
        node_repeats=2,
    )

    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            constraints.within_percent_of_ideal_population(seed, POP_TOLERANCE),
            constraints.contiguous,
        ],
        accept=always_accept,
        initial_state=seed,
        total_steps=steps,
    )

    best_score = -float("inf")
    best: Optional[Partition] = seed
    for i, partition in enumerate(chain):
        s = _score(partition, objective)
        if s > best_score:
            best_score = s
            best = partition
        if progress_cb and i % max(1, steps // 50) == 0:
            progress_cb(i, steps)

    votes = _district_votes(best)
    dem, rep = metrics.seats_won(votes)
    pops = list(best["population"].values())
    pop_dev = (max(pops) - min(pops)) / ideal_pop
    summary = {
        "dem_seats": dem,
        "rep_seats": rep,
        "efficiency_gap": metrics.efficiency_gap(votes),
        "mean_median": metrics.mean_median(votes),
        "mean_polsby_popper": _polsby_popper_mean(best),
        "population_deviation": pop_dev,
    }
    return BestPlan(assignment=dict(best.assignment), metrics=summary, step=steps)


def plan_to_geojson(gdf: gpd.GeoDataFrame, plan: BestPlan) -> dict:
    """Dissolve precincts by district id and emit a GeoJSON FeatureCollection in WGS84."""
    gdf = gdf.copy()
    # gerrychain assignment keys are node ids that match gdf index by default.
    gdf["district"] = gdf.index.map(plan.assignment)
    # Drop precincts not assigned (shouldn't happen).
    gdf = gdf[gdf["district"].notna()]
    dissolved = gdf.dissolve(by="district", aggfunc={"pop": "sum", "dem": "sum", "rep": "sum"})
    dissolved = dissolved.to_crs(epsg=4326).reset_index()
    dissolved["district"] = dissolved["district"].astype(int)
    dissolved["party"] = dissolved.apply(
        lambda row: "D" if row["dem"] > row["rep"] else "R", axis=1
    )
    return dissolved.__geo_interface__
