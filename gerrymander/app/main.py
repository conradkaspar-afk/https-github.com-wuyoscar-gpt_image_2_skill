"""FastAPI entrypoint."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import data, jobs, redistrict
from .schemas import GenerateRequest, JobStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Gerrymander Map Generator")


@app.get("/states")
def list_states() -> dict:
    return {"states": data.available_states()}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    if req.state.upper() not in data.STATE_SOURCES:
        raise HTTPException(404, f"state {req.state} not in supported list")

    async def work(job) -> dict:
        loop = asyncio.get_running_loop()

        def heavy() -> dict:
            job.message = "loading precinct data"
            gdf = data.load_state(req.state)
            if gdf is None:
                raise RuntimeError(f"no data for {req.state}")

            job.message = "building precinct graph"
            graph = redistrict.build_graph(gdf)

            job.message = "seeding plan"
            seed = redistrict.seed_partition(graph, req.num_districts)

            def cb(i, total):
                job.progress = i / total
                job.message = f"step {i}/{total}"

            job.message = "running ReCom chain"
            best = redistrict.run_chain(seed, req.objective, req.steps, progress_cb=cb)

            job.message = "rendering geojson"
            geojson = redistrict.plan_to_geojson(gdf, best)
            return {"geojson": geojson, "metrics": best.metrics}

        return await loop.run_in_executor(None, heavy)

    job = jobs.submit(work)
    return {"job_id": job.id}


@app.get("/job/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return JobStatus(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result=job.result,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


def run() -> None:
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
