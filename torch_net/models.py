"""HTTP messages shared by TORCH schedulers and workers."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

JobState = Literal["reserved", "running", "completed", "failed"]


class JobRequest(BaseModel):
    """A job needs whole, homogeneous GPUs on a single worker."""

    # IDs travel in /jobs/{job_id} paths; reject URL delimiters and traversal.
    job_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}$",
    )
    gpu_count: int = Field(ge=1)
    vram_gb: float = Field(gt=0, allow_inf_nan=False)
    compatible_gpu_types: list[str] = Field(default_factory=list)
    work_units: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("gpu_count", "vram_gb", "work_units", mode="before")
    @classmethod
    def reject_boolean_quantities(cls, value: object) -> object:
        if isinstance(value, bool):
            # Pydantic converts ValueError into HTTP 422; TypeError would escape.
            raise ValueError("GPU and workload quantities must be numeric")  # noqa: TRY004
        return value


class JobStatus(BaseModel):
    """Stable reservation identity and the latest lifecycle state of a job."""

    job_id: str
    worker_id: str
    reservation_id: str
    state: JobState
    gpu_count: int
    error: str | None = None
