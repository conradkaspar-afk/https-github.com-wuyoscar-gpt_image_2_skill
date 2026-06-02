# Gerrymander Map Web App

A research/education web app that generates **realistic gerrymandered US congressional district maps** from real precinct geometry and partisan vote data. The user picks a state, a number of districts, and a political objective; the backend runs a ReCom Markov chain over real precinct adjacency graphs and returns the most-gerrymandered plan it found that still satisfies contiguity and ±2% population balance.

> ⚠️ For education and research on gerrymandering only. Outputs are not intended for any official redistricting use.

## Architecture

```
gerrymander/
├── app/                FastAPI backend
│   ├── main.py         routes: GET /states, POST /generate, GET /job/{id}
│   ├── data.py         lazy VEST shapefile loader + on-disk cache
│   ├── redistrict.py   gerrychain graph build, seed plan, ReCom search
│   ├── metrics.py      efficiency gap, mean-median, Polsby–Popper, seat shares
│   ├── jobs.py         in-process async job registry
│   └── schemas.py      pydantic models
├── web/                static frontend (Leaflet, no build step)
│   ├── index.html
│   └── app.js
├── tests/
└── data/cache/         downloaded state precinct files (gitignored)
```

## Data source

Precinct geometry + two-party vote totals come from the **VEST (Voting and Election Science Team)** Harvard Dataverse collection. The `STATE_SOURCES` table in `app/data.py` maps each USPS code to a downloadable archive. On first request for a state, the archive is downloaded, normalized to columns `pop / dem / rep`, and cached as a GeoPackage under `data/cache/{ST}.gpkg`.

VEST coverage is incomplete: some state-cycle combinations don't exist. The API returns a clean 404 in that case.

## Algorithm

1. Build a precinct adjacency graph with `gerrychain.Graph.from_geodataframe`.
2. Seed an initial plan via `recursive_tree_part` (contiguous, ±2% of ideal population).
3. Run a `MarkovChain` with `ReCom` proposals, `contiguous` + `within_percent_of_ideal_population(0.02)` constraints, `always_accept`.
4. Score every visited plan with the user-selected objective and keep the argmax:
   - `pro_dem` / `pro_rep` — maximize seats for that party, tiebreak on efficiency gap in that direction.
   - `compact` — minimize total Polsby–Popper (control / non-partisan baseline).

The chain is single-threaded; runtimes scale with state size and `steps`. Rhode Island / 500 steps finishes in well under a minute; North Carolina / 5000 steps can take several minutes.

## Run locally

```bash
cd gerrymander
pip install -e .
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/
```

`GET /` serves the Leaflet UI from `web/`. `POST /generate` accepts JSON:

```json
{"state": "RI", "num_districts": 2, "objective": "pro_dem", "steps": 500}
```

It returns `{"job_id": "..."}`. Poll `GET /job/{job_id}` until `status == "done"`; the `result` field then contains `{geojson, metrics}`.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

`test_metrics.py` checks hand-computed metric values. `test_chain_small.py` runs the chain on a 5×5 synthetic grid and verifies the `pro_dem` objective is at least as good as the seed plan.
