"""HTTP messages shared by TORCH schedulers and workers."""

from typing import Literal

from pydantic import BaseModel, Field

JobState = Literal["reserved", "running", "completed", "failed"]


class JobRequest(BaseModel):
    """A job needs whole, homogeneous GPUs on a single worker."""

    job_id: str = Field(min_length=1)
    gpu_count: int = Field(ge=1)
    vram_gb: float = Field(gt=0, allow_inf_nan=False)
    compatible_gpu_types: list[str] = Field(default_factory=list)
    work_units: float = Field(gt=0, allow_inf_nan=False)


class JobStatus(BaseModel):
    """Stable reservation identity and the latest lifecycle state of a job."""

    job_id: str
    worker_id: str
    reservation_id: str
    state: JobState
    gpu_count: int
