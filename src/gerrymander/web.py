"""Self-contained HTTP server that exposes the districting engine to a
browser-based UI.

Usage:
    python -m gerrymander.web
or:
    gerrymander-web

Opens http://127.0.0.1:8765 . Serves a vanilla-JS + Leaflet front-end from
`static/` and exposes a small JSON API powered by the existing Python
districting engine in `geo.py` + `districting.py`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .geo import build_state_real, STATE_FIPS
from .state_model import STATE_PRESETS, list_states as _list_synth_states
from .districting import neutral_districts, pack_and_crack
from .metrics import summarize


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# State postal -> full name (for the UI dropdown).
STATE_NAMES: Dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}


# Cache of built `StateGrid` objects keyed by (state, seed). The grid is
# expensive to construct (parses bundled GeoJSON + CSV); districting itself is
# fast, so this cache keeps API latency low.
_GRID_CACHE: Dict[Tuple[str, int], Any] = {}
_GRID_LOCK = threading.Lock()


def _get_grid(state: str, seed: int):
    key = (state.upper(), int(seed))
    with _GRID_LOCK:
        if key not in _GRID_CACHE:
            _GRID_CACHE[key] = build_state_real(key[0], seed=key[1])
        return _GRID_CACHE[key]


def _plan_to_geojson(plan, grid) -> Dict[str, Any]:
    """Translate a DistrictPlan over real counties into a GeoJSON
    FeatureCollection. Each county feature carries its district id and the
    district's aggregate vote share so the front-end can color quickly."""
    # Pre-compute district vote shares so each feature can show them.
    d_votes = plan.district_d_votes()
    n = plan.num_districts
    d_pct = [0.0] * n
    for i, (d, r) in enumerate(d_votes):
        tot = d + r
        d_pct[i] = (100.0 * d / tot) if tot > 0 else 50.0

    features = []
    for i, p in enumerate(grid.precincts):
        geom = getattr(p, "geometry", None)
        if not geom:
            continue
        if len(geom) == 1:
            gj_geom = {"type": "Polygon", "coordinates": geom[0]}
        else:
            gj_geom = {"type": "MultiPolygon", "coordinates": geom}
        district_id = plan.assignment[i]
        features.append({
            "type": "Feature",
            "geometry": gj_geom,
            "properties": {
                "county": getattr(p, "name", str(i)),
                "fips_index": i,
                "district_id": int(district_id),
                "district_d_pct": round(d_pct[district_id], 2),
                "population": int(p.population),
                "d_share": round(p.d_share, 4),
                "demographics": {k: round(v, 4) for k, v in p.demographics.items()},
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _district_summaries(plan) -> list:
    votes = plan.district_d_votes()
    demos = plan.district_demographics()
    pops = plan.district_population()
    out = []
    for d in range(plan.num_districts):
        dv, rv = votes[d]
        tot = dv + rv
        d_pct = (100.0 * dv / tot) if tot else 50.0
        out.append({
            "id": d,
            "d_pct": round(d_pct, 2),
            "r_pct": round(100.0 - d_pct, 2),
            "winner": "D" if d_pct > 50 else "R",
            "population": pops[d],
            "demographics": {k: round(v, 4) for k, v in demos[d].items()},
        })
    return out


def _build_plan_response(state: str, party: str, seats: Optional[int],
                        intensity: float, seed: int) -> Dict[str, Any]:
    grid = _get_grid(state, seed)
    if seats is None:
        seats = grid.num_districts
    seats = max(1, min(int(seats), max(1, grid.n)))
    party_u = party.upper()
    if party_u in ("NEUTRAL", "N", ""):
        plan = neutral_districts(grid, n=seats, seed=seed)
        party_label = "neutral"
    elif party_u in ("D", "DEM", "DEMOCRAT"):
        plan = pack_and_crack(grid, n=seats, target_party="D", intensity=intensity, seed=seed)
        party_label = "D"
    elif party_u in ("R", "REP", "REPUBLICAN"):
        plan = pack_and_crack(grid, n=seats, target_party="R", intensity=intensity, seed=seed)
        party_label = "R"
    else:
        raise ValueError(f"Unknown party {party!r}")

    metrics = summarize(plan)
    geojson = _plan_to_geojson(plan, grid)
    districts = _district_summaries(plan)
    minx, miny, maxx, maxy = grid._bbox  # type: ignore[attr-defined]
    return {
        "state": grid.state,
        "state_name": STATE_NAMES.get(grid.state, grid.state),
        "party": party_label,
        "seats": seats,
        "intensity": intensity,
        "seed": seed,
        "bbox": [minx, miny, maxx, maxy],
        "geojson": geojson,
        "districts": districts,
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
    }


def _states_response() -> list:
    out = []
    for code in sorted(STATE_FIPS.keys()):
        if code not in STATE_PRESETS:
            continue
        _, _, n_districts, d_lean, _ = STATE_PRESETS[code]
        out.append({
            "code": code,
            "name": STATE_NAMES.get(code, code),
            "default_seats": n_districts,
            "min_seats": 1,
            "max_seats": max(n_districts * 2, 8),
            "d_lean": round(d_lean, 3),
        })
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "GerrymanderWeb/1.0"

    def _send_json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, content_type: str) -> None:
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except FileNotFoundError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        # Quieter logs.
        sys.stderr.write("[gerrymander-web] " + format % args + "\n")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/" or path == "/index.html":
                self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                rel = path[len("/static/"):].replace("..", "")
                full = os.path.join(STATIC_DIR, rel)
                ext = os.path.splitext(full)[1].lower()
                ctypes = {
                    ".html": "text/html; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".png": "image/png",
                    ".svg": "image/svg+xml",
                    ".json": "application/json; charset=utf-8",
                }
                self._send_file(full, ctypes.get(ext, "application/octet-stream"))
            elif path == "/api/states":
                self._send_json(200, _states_response())
            elif path == "/api/plan":
                q = parse_qs(parsed.query)
                state = (q.get("state") or ["PA"])[0]
                party = (q.get("party") or ["neutral"])[0]
                seats = q.get("seats", [None])[0]
                intensity = float((q.get("intensity") or ["0.9"])[0])
                seed = int((q.get("seed") or ["7"])[0])
                payload = _build_plan_response(
                    state=state,
                    party=party,
                    seats=int(seats) if seats is not None else None,
                    intensity=intensity,
                    seed=seed,
                )
                self._send_json(200, payload)
            else:
                self.send_error(404, "Not found")
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": str(exc)})


class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(host: str = "127.0.0.1", port: int = 8765) -> ThreadingServer:
    """Build and return the server (not started). Useful for tests."""
    return ThreadingServer((host, port), Handler)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="gerrymander-web",
                                     description="Run the gerrymandering visualization web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    httpd = serve(args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"gerrymander-web running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
