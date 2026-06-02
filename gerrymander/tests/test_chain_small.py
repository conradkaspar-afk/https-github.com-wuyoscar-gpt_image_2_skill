"""Smoke test of the ReCom chain on a synthetic 5x5 grid.

Skipped if gerrychain is not installed in the test environment.
"""
from __future__ import annotations

import pytest

pytest.importorskip("gerrychain")
pytest.importorskip("networkx")

import networkx as nx
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally, cut_edges

from app import redistrict


def _grid_graph():
    g = nx.grid_2d_graph(5, 5)
    # Relabel to integer node ids; gerrychain expects hashable ids and node attrs.
    g = nx.convert_node_labels_to_integers(g, label_attribute="coord")
    for n in g.nodes:
        x, y = g.nodes[n]["coord"]
        g.nodes[n]["pop"] = 100
        # Left half leans D, right half leans R.
        if x < 2:
            g.nodes[n]["dem"], g.nodes[n]["rep"] = 70, 30
        elif x > 2:
            g.nodes[n]["dem"], g.nodes[n]["rep"] = 30, 70
        else:
            g.nodes[n]["dem"], g.nodes[n]["rep"] = 50, 50
    return Graph(g)


def test_chain_improves_pro_dem_objective():
    graph = _grid_graph()
    seed = redistrict.seed_partition(graph, num_districts=5)
    seed_votes = {d: (seed["dem"][d], seed["rep"][d]) for d in seed.parts}
    seed_d_seats = sum(1 for d, r in seed_votes.values() if d > r)

    best = redistrict.run_chain(seed, objective="pro_dem", steps=50)
    assert best.metrics["dem_seats"] >= seed_d_seats
    assert best.metrics["population_deviation"] <= 0.05
