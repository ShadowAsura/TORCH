"""FastAPI entrypoint for TORCH's networked scheduling coordinator."""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from torch_net.scheduler import (
    Scheduler,
    SchedulerSettings,
    SchedulingConfigurationError,
    SchedulingRequest,
)


def create_app(
    settings: SchedulerSettings | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    settings = settings or SchedulerSettings.from_env()
    scheduler = Scheduler(settings, client=client)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await scheduler.close()

    app = FastAPI(title="TORCH scheduler", version="0.1.0", lifespan=lifespan)
    app.state.scheduler = scheduler

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "scheduler",
            "configured_workers": len(settings.workers),
        }

    @app.get("/workers")
    async def workers():
        # These are configured network endpoints, not cached resource snapshots.
        return {
            "workers": [
                {"worker_id": worker.name, "url": worker.url}
                for worker in settings.workers
            ]
        }

    @app.post("/jobs")
    async def submit_job(job: SchedulingRequest):
        try:
            record, created = await scheduler.submit(job)
        except SchedulingConfigurationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(status_code=202 if created else 200, content=record)



    return app


app = create_app()
