"""In-process async job registry. Single-worker; fine for v1."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    task: Optional[asyncio.Task] = field(default=None, repr=False)


_JOBS: dict[str, Job] = {}


def get(job_id: str) -> Optional[Job]:
    return _JOBS.get(job_id)


def submit(coro_factory: Callable[[Job], Awaitable[dict]]) -> Job:
    job = Job(id=uuid.uuid4().hex[:12])
    _JOBS[job.id] = job

    async def runner():
        job.status = "running"
        try:
            job.result = await coro_factory(job)
            job.status = "done"
            job.progress = 1.0
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job.id)
            job.status = "error"
            job.message = f"{type(e).__name__}: {e}"

    job.task = asyncio.create_task(runner())
    return job
