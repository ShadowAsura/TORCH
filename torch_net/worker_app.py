"""FastAPI process serving one worker's local GPU reservations.

Start with: uvicorn torch_net.worker_app:app --host 0.0.0.0 --port 8001
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from .models import JobRequest, JobStatus
from .worker import JobNotFound, ReservationConflict, Worker, WorkerSettings


def create_app(settings: WorkerSettings | None = None) -> FastAPI:
    worker = Worker(settings if settings is not None else WorkerSettings.from_env())

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await worker.close()

    app = FastAPI(title="TORCH Worker", lifespan=lifespan)
    app.state.worker = worker

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "worker_id": worker.settings.worker_id}

    @app.get("/resources")
    async def resources() -> dict:
        return await worker.resources()

    @app.post("/reservations", response_model=JobStatus, status_code=201)
    async def reserve(request: JobRequest, response: Response) -> JobStatus:
        try:
            job, created = await worker.reserve(request)
        except ReservationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        response.status_code = 201 if created else 200
        return job




    return app


app = create_app()
