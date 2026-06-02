from typing import Literal, Optional
from pydantic import BaseModel, Field

Objective = Literal["pro_dem", "pro_rep", "compact"]


class GenerateRequest(BaseModel):
    state: str = Field(..., min_length=2, max_length=2, description="USPS state code, e.g. 'RI'")
    num_districts: int = Field(..., ge=1, le=60)
    objective: Objective = "pro_dem"
    steps: int = Field(500, ge=10, le=20000)


class Metrics(BaseModel):
    dem_seats: int
    rep_seats: int
    efficiency_gap: float
    mean_median: float
    mean_polsby_popper: float
    population_deviation: float


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
