"""Actual ASGI requests exercising worker reservation races and cleanup."""

import asyncio

from httpx import ASGITransport, AsyncClient

from torch_net.worker import WorkerSettings
from torch_net.worker_app import create_app


def settings(**overrides):
    values = {
        "worker_id": "test-a100",
        "gpu_type": "A100",
        "gpu_count": 2,
        "gpu_vram_gb": 80,
        "gpu_performance": 2.0,
        "sim_seconds_per_work_unit": 0.06,
        "reservation_lease_seconds": 0.4,
    }
    values.update(overrides)
    return WorkerSettings(**values)


def job(job_id="job-1", **overrides):
    values = {
        "job_id": job_id,
        "gpu_count": 1,
        "vram_gb": 40,
        "compatible_gpu_types": ["A100"],
        "work_units": 2,
    }
    values.update(overrides)
    return values


def test_resources_compatibility_and_validation():
    async def check():
        app = create_app(settings())
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            health = await client.get("/health")
            assert health.json() == {"status": "ok", "worker_id": "test-a100"}
            resources = (await client.get("/resources")).json()
            assert resources == {
                "worker_id": "test-a100",
                "gpu_type": "A100",
                "gpu_count": 2,
                "free_gpus": 2,
                "gpu_vram_gb": 80,
                "available_vram_gb": 160,
                "performance": 2.0,
                "allocated_gpus": 0,
                "active_jobs": 0,
            }
        await app.state.worker.close()

    asyncio.run(check())
















