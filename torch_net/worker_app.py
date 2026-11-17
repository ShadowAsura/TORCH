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






    return app


app = create_app()
