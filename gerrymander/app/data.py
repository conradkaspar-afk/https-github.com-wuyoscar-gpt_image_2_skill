"""Lazy state precinct-data loader.

Downloads VEST (Voting and Election Science Team) precinct shapefiles from the
Harvard Dataverse on first request, normalizes columns to (pop, dem, rep,
geometry), and caches as a GeoPackage under data/cache/{ST}.gpkg.

The STATE_SOURCES URLs point to VEST's 2020 precinct boundaries with 2020
presidential results, which is the most broadly available cycle. Some states
are absent from VEST and will return None; the API surfaces that as 404.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# VEST 2020 precinct + 2020 presidential results, Harvard Dataverse.
# Persistent IDs of the form doi:10.7910/DVN/K7760H (per state). The URLs below
# resolve to the shapefile zip download endpoint. A subset is listed; states
# not present here will 404 until added.
VEST_BASE = "https://dataverse.harvard.edu/api/access/datafile/:persistentId"
STATE_SOURCES: dict[str, dict[str, str]] = {
    # USPS: {url, dem_col, rep_col, pop_col}
    "RI": {
        "pid": "doi:10.7910/DVN/K7760H/RHODE_ISLAND_2020",
        "dem_col": "G20PREDBID",
        "rep_col": "G20PRERTRU",
        "pop_col": "TOTPOP",
    },
    "NC": {
        "pid": "doi:10.7910/DVN/K7760H/NORTH_CAROLINA_2020",
        "dem_col": "G20PREDBID",
        "rep_col": "G20PRERTRU",
        "pop_col": "TOTPOP",
    },
    "PA": {
        "pid": "doi:10.7910/DVN/K7760H/PENNSYLVANIA_2020",
        "dem_col": "G20PREDBID",
        "rep_col": "G20PRERTRU",
        "pop_col": "TOTPOP",
    },
    "WI": {
        "pid": "doi:10.7910/DVN/K7760H/WISCONSIN_2020",
        "dem_col": "G20PREDBID",
        "rep_col": "G20PRERTRU",
        "pop_col": "TOTPOP",
    },
    "MD": {
        "pid": "doi:10.7910/DVN/K7760H/MARYLAND_2020",
        "dem_col": "G20PREDBID",
        "rep_col": "G20PRERTRU",
        "pop_col": "TOTPOP",
    },
}


def available_states() -> list[str]:
    return sorted(STATE_SOURCES.keys())


def _cache_path(state: str) -> Path:
    return CACHE_DIR / f"{state.upper()}.gpkg"


def load_state(state: str) -> Optional[gpd.GeoDataFrame]:
    """Return GeoDataFrame with columns: pop, dem, rep, geometry. None if unsupported."""
    state = state.upper()
    if state not in STATE_SOURCES:
        return None

    cache = _cache_path(state)
    if cache.exists():
        log.info("cache hit: %s", state)
        return gpd.read_file(cache)

    src = STATE_SOURCES[state]
    log.info("downloading VEST data for %s", state)
    resp = requests.get(VEST_BASE, params={"persistentId": src["pid"]}, timeout=120)
    resp.raise_for_status()

    # Read shapefile straight out of the zip in memory.
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        shp_names = [n for n in zf.namelist() if n.lower().endswith(".shp")]
        if not shp_names:
            raise RuntimeError(f"no .shp inside VEST archive for {state}")
        # geopandas can read directly from a zip via /vsizip/ on the path-like
        # URL when given a single file URI; simpler to extract to cache dir.
        extract_dir = CACHE_DIR / f"_{state}_raw"
        extract_dir.mkdir(exist_ok=True)
        zf.extractall(extract_dir)
        gdf = gpd.read_file(extract_dir / shp_names[0])

    gdf = gdf.rename(
        columns={src["pop_col"]: "pop", src["dem_col"]: "dem", src["rep_col"]: "rep"}
    )
    keep = ["pop", "dem", "rep", "geometry"]
    missing = [c for c in keep if c not in gdf.columns]
    if missing:
        raise RuntimeError(f"VEST data for {state} missing expected columns: {missing}")
    gdf = gdf[keep].copy()
    # Ensure numeric.
    for c in ("pop", "dem", "rep"):
        gdf[c] = gdf[c].fillna(0).astype(float)
    # Project to an equal-area CRS so contiguity / area metrics behave.
    gdf = gdf.to_crs(epsg=5070)
    gdf.to_file(cache, driver="GPKG")
    return gdf
