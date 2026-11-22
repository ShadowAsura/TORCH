"""Scheduler/worker integration over actual HTTP request and ASGI boundaries.

The Docker smoke test exercises TCP sockets; these tests run the same apps
through HTTPX's in-memory ASGI transport for deterministic fault injection.
"""

import asyncio

import httpx

from torch_net.policies import WorkerEndpoint
from torch_net.scheduler import SchedulerSettings
from torch_net.scheduler_app import create_app as make_scheduler
from torch_net.worker import WorkerSettings
from torch_net.worker_app import create_app as make_worker


class WorkerRouter(httpx.AsyncBaseTransport):
    def __init__(self, apps: dict[str, object]):
        self.transports = {
            hostname: httpx.ASGITransport(app=app) for hostname, app in apps.items()
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self.transports.get(request.url.host)
        if transport is None:
            raise httpx.ConnectError(f"worker {request.url.host} is unreachable")
        return await transport.handle_async_request(request)

    async def aclose(self):
        await asyncio.gather(
            *(transport.aclose() for transport in self.transports.values())
        )


def job(job_id: str, **overrides) -> dict:
    request = {
        "job_id": job_id,
        "gpu_count": 1,
        "vram_gb": 16,
        "compatible_gpu_types": [],
        "work_units": 4,
    }
    request.update(overrides)
    return request


async def await_terminal(client, job_id: str):
    for _ in range(300):
        response = await client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        result = response.json()
        if result["state"] in ("completed", "failed"):
            return result
        await asyncio.sleep(0.005)
    raise AssertionError(f"job {job_id} did not complete or fail within test budget")


def worker(worker_id: str, kind: str, vram: int, performance: float, count: int = 1):
    return make_worker(
        WorkerSettings(
            worker_id=worker_id,
            gpu_type=kind,
            gpu_count=count,
            gpu_vram_gb=vram,
            gpu_performance=performance,
            sim_seconds_per_work_unit=0.008,
            reservation_lease_seconds=1.0,
        )
    )












