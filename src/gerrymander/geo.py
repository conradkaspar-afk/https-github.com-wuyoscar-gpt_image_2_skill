"""Real US county geography + demographics + 2016 vote data.

Loads bundled datasets:
  - `data/us_counties.json` — Census cartographic boundary GeoJSON
    (FIPS-coded county polygons).
  - `data/county_context.csv` — MIT Election Data + Science Lab
    "2018 election context" file with per-county population, race breakdown,
    rural fraction, and 2016 presidential vote totals.

Produces a `StateGrid` whose precincts are real counties and whose
"neighbors" relation is computed from shared polygon edges. Districting
and rendering modules then operate on real geometry.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .state_model import Precinct, StateGrid, STATE_PRESETS


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GEOJSON_PATH = os.path.join(DATA_DIR, "us_counties.json")
CONTEXT_CSV = os.path.join(DATA_DIR, "county_context.csv")


# State postal code -> two-digit FIPS.
STATE_FIPS: Dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16",
    "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
    "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
    "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


# Cached singletons (loaded lazily, kept for the process lifetime).
_GEOJSON: Optional[dict] = None
_CONTEXT: Optional[Dict[str, dict]] = None


def _load_geojson() -> dict:
    global _GEOJSON
    if _GEOJSON is None:
        with open(GEOJSON_PATH, "r") as fh:
            _GEOJSON = json.load(fh)
    return _GEOJSON


def _load_context() -> Dict[str, dict]:
    """FIPS string -> context row dict."""
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    out: Dict[str, dict] = {}
    with open(CONTEXT_CSV, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fips = row["fips"].zfill(5)
            out[fips] = row
    _CONTEXT = out
    return out


def _ring_centroid_area(ring: List[List[float]]) -> Tuple[float, float, float]:
    """Shoelace centroid + signed area for a single ring (lon, lat)."""
    cx = cy = a2 = 0.0
    n = len(ring)
    for i in range(n - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        cross = x0 * y1 - x1 * y0
        a2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if a2 == 0:
        # Degenerate ring: average vertices.
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (sum(xs) / max(1, len(xs)), sum(ys) / max(1, len(ys)), 0.0)
    area = a2 / 2.0
    cx /= 3.0 * a2
    cy /= 3.0 * a2
    return (cx, cy, abs(area))


def _polygons_from_geom(geom: dict) -> List[List[List[List[float]]]]:
    """Return list of polygons; each polygon = list of rings; each ring = list of [lon, lat]."""
    t = geom["type"]
    if t == "Polygon":
        return [geom["coordinates"]]
    if t == "MultiPolygon":
        return geom["coordinates"]
    return []


def _county_centroid(polys: List[List[List[List[float]]]]) -> Tuple[float, float]:
    cx = cy = total_a = 0.0
    for poly in polys:
        if not poly:
            continue
        x, y, a = _ring_centroid_area(poly[0])  # outer ring
        cx += x * a
        cy += y * a
        total_a += a
    if total_a == 0:
        # Fallback
        for poly in polys:
            for ring in poly:
                for p in ring:
                    cx += p[0]
                    cy += p[1]
                    total_a += 1
        return (cx / max(1, total_a), cy / max(1, total_a))
    return (cx / total_a, cy / total_a)


def _bbox(polys: List[List[List[List[float]]]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for poly in polys for ring in poly for p in ring]
    ys = [p[1] for poly in polys for ring in poly for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _build_adjacency(polys_list: List[List[List[List[List[float]]]]]) -> List[List[int]]:
    """Edge-adjacency via shared vertices.

    Two counties share an edge if they share >= 2 vertices (rounded to 5
    decimals, ~1 m). Pure point-touch (single shared vertex) doesn't count.
    """
    n = len(polys_list)
    vmap: Dict[Tuple[int, int], List[int]] = {}
    for i, polys in enumerate(polys_list):
        seen = set()
        for poly in polys:
            for ring in poly:
                for p in ring:
                    key = (int(round(p[0] * 1e5)), int(round(p[1] * 1e5)))
                    if key in seen:
                        continue
                    seen.add(key)
                    vmap.setdefault(key, []).append(i)
    shared: Dict[Tuple[int, int], int] = {}
    for owners in vmap.values():
        for a in range(len(owners)):
            for b in range(a + 1, len(owners)):
                u, v = owners[a], owners[b]
                if u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                shared[key] = shared.get(key, 0) + 1
    adj: List[List[int]] = [[] for _ in range(n)]
    for (u, v), c in shared.items():
        if c >= 2:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def _connect_islands(centroids: List[Tuple[float, float]], adj: List[List[int]]) -> List[List[int]]:
    """Ensure the adjacency graph is connected by linking each disconnected
    component to its geographically nearest neighbor in the rest of the state
    (handles islands and disjoint coastal counties).
    """
    n = len(centroids)
    if n == 0:
        return adj
    # Find components.
    comp = [-1] * n
    cid = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        stack = [s]
        comp[s] = cid
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if comp[nb] == -1:
                    comp[nb] = cid
                    stack.append(nb)
        cid += 1
    if cid == 1:
        return adj
    # Link each non-zero component to component 0 via the nearest pair.
    for c in range(1, cid):
        in_c = [i for i in range(n) if comp[i] == c]
        in_main = [i for i in range(n) if comp[i] < c]  # already linked to 0
        best = None
        for i in in_c:
            for j in in_main:
                d = (centroids[i][0] - centroids[j][0]) ** 2 + (centroids[i][1] - centroids[j][1]) ** 2
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is not None:
            _, i, j = best
            adj[i].append(j)
            adj[j].append(i)
            # Merge component labels.
            for k in range(n):
                if comp[k] == c:
                    comp[k] = comp[j]
    return adj


def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_state_real(state: str, seed: int = 0):
    """Construct a `StateGrid` using real county geometry and real 2016 vote /
    demographic data per county. Counties with no context-CSV row are filled
    with state-average values.
    """
    import numpy as np
    state = state.upper()
    if state not in STATE_FIPS:
        raise ValueError(f"Unknown state {state!r}")
    if state not in STATE_PRESETS:
        raise ValueError(f"No district count preset for {state!r}")

    fips = STATE_FIPS[state]
    gj = _load_geojson()
    ctx = _load_context()
    rng = np.random.default_rng(seed)

    # Filter features by state FIPS prefix.
    features = [f for f in gj["features"] if (f.get("id") or "").startswith(fips)]
    features.sort(key=lambda f: f["id"])
    n = len(features)
    if n == 0:
        raise RuntimeError(f"No counties found for {state} (FIPS {fips}) in bundled GeoJSON")

    polys_list: List[List[List[List[List[float]]]]] = []
    centroids: List[Tuple[float, float]] = []
    populations: List[int] = []
    d_shares: List[float] = []
    demos: List[Dict[str, float]] = []
    names: List[str] = []

    rows, cols, n_districts, state_d_lean, _ = STATE_PRESETS[state]

    for feat in features:
        polys = _polygons_from_geom(feat["geometry"])
        polys_list.append(polys)
        centroid = _county_centroid(polys)
        centroids.append(centroid)
        county_fips = feat["id"]
        names.append(feat.get("properties", {}).get("NAME", county_fips))
        row = ctx.get(county_fips)
        if row is not None:
            pop = int(_safe_float(row["total_population"], 10000))
            t = _safe_float(row["trump16"], 0)
            c = _safe_float(row["clinton16"], 0)
            d_share = c / (c + t) if (c + t) > 0 else state_d_lean
            white = _safe_float(row["white_pct"], 80) / 100.0
            black = _safe_float(row["black_pct"], 5) / 100.0
            hispanic = _safe_float(row["hispanic_pct"], 5) / 100.0
            nonwhite = _safe_float(row["nonwhite_pct"], 20) / 100.0
            # Use leftover non-W/B/H share, split into Asian (60%) and other (40%).
            leftover = max(0.0, nonwhite - black - hispanic)
            asian = leftover * 0.6
            other = leftover * 0.4
        else:
            pop = 10000
            d_share = state_d_lean
            white, black, hispanic, asian, other = 0.7, 0.1, 0.1, 0.05, 0.05
        total_d = white + black + hispanic + asian + other
        if total_d > 0:
            white /= total_d; black /= total_d; hispanic /= total_d
            asian /= total_d; other /= total_d
        else:
            white, black, hispanic, asian, other = 0.7, 0.1, 0.1, 0.05, 0.05
        populations.append(max(500, pop))
        d_shares.append(min(0.999, max(0.001, d_share)))
        demos.append({"white": white, "black": black, "hispanic": hispanic,
                      "asian": asian, "other": other})

    adj = _build_adjacency(polys_list)
    adj = _connect_islands(centroids, adj)

    precincts: List[Precinct] = []
    for i, feat in enumerate(features):
        p = Precinct(
            idx=i, row=0, col=i,
            population=populations[i],
            d_share=d_shares[i],
            demographics=demos[i],
        )
        # Attach geo data as dynamic attributes.
        p.centroid = centroids[i]  # type: ignore[attr-defined]
        p.geometry = polys_list[i]  # type: ignore[attr-defined]
        p.name = names[i]  # type: ignore[attr-defined]
        precincts.append(p)

    minx = min(c[0] for c in centroids)
    miny = min(c[1] for c in centroids)
    maxx = max(c[0] for c in centroids)
    maxy = max(c[1] for c in centroids)

    grid = StateGrid(
        state=state, rows=int(round(maxy - miny)) or 1, cols=int(round(maxx - minx)) or 1,
        num_districts=n_districts, precincts=precincts, d_lean=state_d_lean, seed=seed,
    )
    # Override neighbors with adjacency list (since grid relation no longer applies).
    grid._adjacency = adj  # type: ignore[attr-defined]
    grid._is_real = True  # type: ignore[attr-defined]
    grid._bbox = (minx, miny, maxx, maxy)  # type: ignore[attr-defined]
    grid._names = names  # type: ignore[attr-defined]

    # Monkey-patch `neighbors` to use the adjacency list when real.
    orig_neighbors = grid.neighbors

    def _neighbors(idx: int, _adj=adj) -> List[int]:
        return list(_adj[idx])

    grid.neighbors = _neighbors  # type: ignore[assignment]
    return grid


def list_real_states() -> List[str]:
    """States supported with real geography (all 48 contiguous + HI; AK omitted
    because the Plotly GeoJSON excludes Alaska from the standard projection;
    we still allow it via centroid fallback)."""
    return sorted(STATE_FIPS.keys())
