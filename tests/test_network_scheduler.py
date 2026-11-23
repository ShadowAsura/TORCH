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


async def run_with_network(
    configured, live_apps, scenario, *, sample_size=2, max_retries=20
):
    settings = SchedulerSettings(
        workers=tuple(
            WorkerEndpoint(name, f"http://{name}:8000") for name in configured
        ),
        sample_size=sample_size,
        max_retries=max_retries,
        retry_delay_seconds=0.01,
        timeout_seconds=0.1,
        poll_seconds=0.005,
    )
    try:
        async with httpx.AsyncClient(transport=WorkerRouter(live_apps)) as worker_http:
            app = make_scheduler(settings, client=worker_http)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://scheduler",
                ) as scheduler_http:
                    await scenario(scheduler_http, live_apps)
            finally:
                await app.state.scheduler.close()
    finally:
        for worker_app in live_apps.values():
            await worker_app.state.worker.close()


def test_centralized_queries_all_workers_sampling_queries_subset_and_releases():
    async def scenario(client, apps):
        centralized = await client.post(
            "/jobs", json=job("central", policy="centralized")
        )
        assert centralized.status_code == 202
        result = await await_terminal(client, "central")
        assert result["state"] == "completed"
        assert result["selected_worker"] == "a100"  # shortest simulated runtime
        assert result["query_count"] == 3
        assert set(result["queried_workers"]) == {"a100", "l40", "t4"}
        assert result["reservation_rejections"] == 0
        assert result["scheduling_latency_ms"] >= 0
        assert result["execution_seconds"] > 0

        sampling = await client.post("/jobs", json=job("sampled", policy="sampling"))
        assert sampling.status_code == 202
        sampled = await await_terminal(client, "sampled")
        assert sampled["state"] == "completed"
        assert sampled["query_count"] == 2
        assert len(sampled["queried_workers"]) == 2
        assert sampled["selected_worker"] in sampled["queried_workers"]
        assert (
            await client.post("/jobs", json=job("sampled", policy="sampling"))
        ).status_code == 200
        assert (
            await client.post("/jobs", json=job("sampled", work_units=5))
        ).status_code == 409

        for app in apps.values():
            state = await app.state.worker.resources()
            assert state["allocated_gpus"] == 0
        metrics = (await client.get("/metrics")).json()
        assert metrics["jobs_completed"] == 2
        assert metrics["worker_state_queries"] == 5
        assert metrics["reservation_successes"] == 2
        assert metrics["mean_execution_seconds"] > 0

    asyncio.run(
        run_with_network(
            ["a100", "l40", "t4"],
            {
                "a100": worker("a100", "A100", 80, 3),
                "l40": worker("l40", "L40", 48, 2),
                "t4": worker("t4", "T4", 16, 1),
            },
            scenario,
        )
    )








