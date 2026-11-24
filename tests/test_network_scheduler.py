"""Scheduler/worker integration over actual HTTP request and ASGI boundaries.

The Docker smoke test exercises TCP sockets; these tests run the same apps
through HTTPX's in-memory ASGI transport for deterministic fault injection.
"""

import asyncio

import httpx
import pytest

from torch_net.policies import WorkerEndpoint
from torch_net.scheduler import (
    Scheduler,
    SchedulerClosedError,
    SchedulerSettings,
    SchedulingConfigurationError,
    SchedulingRequest,
)
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


class FaultRouter(WorkerRouter):
    """Intercept selected HTTP calls while other requests reach the real worker."""

    def __init__(self, apps, fault):
        super().__init__(apps)
        self.fault = fault

    async def handle_async_request(self, request):
        return await self.fault(request, super().handle_async_request)


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
    configured,
    live_apps,
    scenario,
    *,
    sample_size=2,
    max_retries=20,
    fault=None,
    max_monitor_errors=5,
    monitor_timeout_seconds=3600,
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
        max_monitor_errors=max_monitor_errors,
        monitor_timeout_seconds=monitor_timeout_seconds,
    )
    try:
        router = FaultRouter(live_apps, fault) if fault else WorkerRouter(live_apps)
        async with httpx.AsyncClient(transport=router) as worker_http:
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


def test_heterogeneous_constraints_reject_incompatible_jobs():
    async def scenario(client, _):
        submitted = await client.post(
            "/jobs",
            json=job("a100-only", vram_gb=64, compatible_gpu_types=["A100"]),
        )
        assert submitted.status_code == 202
        result = await await_terminal(client, "a100-only")
        assert result["state"] == "completed"
        assert result["selected_worker"] == "a100"

        rejected = await client.post("/jobs", json=job("impossible", vram_gb=128))
        assert rejected.status_code == 202
        failure = await await_terminal(client, "impossible")
        assert failure["state"] == "failed"
        assert failure["reason"] == "no worker has compatible GPU capacity"
        assert failure["query_count"] == 2
        assert failure["retries"] == 0
        # Failed admission is not a successful job's scheduling latency.
        metrics = (await client.get("/metrics")).json()
        assert metrics["mean_scheduling_latency_ms"] == result["scheduling_latency_ms"]

    asyncio.run(
        run_with_network(
            ["l40", "a100"],
            {"l40": worker("l40", "L40", 48, 1), "a100": worker("a100", "A100", 80, 2)},
            scenario,
        )
    )


def test_unreachable_worker_does_not_prevent_healthy_worker_placement():
    async def scenario(client, _):
        assert (await client.get("/health")).json()["configured_workers"] == 2
        assert (await client.post("/jobs", json=job("available"))).status_code == 202
        status = await await_terminal(client, "available")
        assert status["state"] == "completed"
        assert status["selected_worker"] == "a100"
        assert status["query_count"] == 2
        assert (await client.get("/metrics")).json()["worker_network_errors"] >= 1

    asyncio.run(
        run_with_network(
            ["unavailable", "a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
        )
    )


def test_competing_scheduler_requests_retry_after_worker_reservation_rejection():
    async def scenario(client, apps):
        first, second = await asyncio.gather(
            client.post("/jobs", json=job("first", work_units=15)),
            client.post("/jobs", json=job("second", work_units=15)),
        )
        assert first.status_code == second.status_code == 202
        a, b = await asyncio.gather(
            await_terminal(client, "first"),
            await_terminal(client, "second"),
        )
        assert a["state"] == b["state"] == "completed"
        assert (await apps["a100"].state.worker.resources())["free_gpus"] == 1
        metrics = (await client.get("/metrics")).json()
        assert metrics["reservation_successes"] == 2
        assert metrics["reservation_rejections"] >= 1
        assert metrics["scheduling_retries"] >= 1

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 1)},
            scenario,
            max_retries=100,
        )
    )


def test_missing_workers_fail_after_bounded_retries():
    async def scenario(client, _):
        assert (await client.post("/jobs", json=job("offline"))).status_code == 202
        result = await await_terminal(client, "offline")
        assert result["state"] == "failed"
        assert result["reason"] == "no capacity after bounded retries"
        assert result["attempts"] == 3
        assert result["retries"] == 2
        assert result["query_count"] == 3

    asyncio.run(
        run_with_network(
            ["missing"],
            {},
            scenario,
            max_retries=2,
        )
    )


def test_unsupported_policy_does_not_register_or_dispatch_job():
    async def scenario(client, apps):
        response = await client.post("/jobs", json=job("unsupported", policy="gossip"))
        assert response.status_code == 422
        assert "unknown policy" in response.json()["detail"]
        assert (await client.get("/jobs/unsupported")).status_code == 404

        metrics = (await client.get("/metrics")).json()
        assert metrics["jobs_submitted"] == 0
        assert metrics["worker_state_queries"] == 0
        assert (await apps["a100"].state.worker.resources())["allocated_gpus"] == 0

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
        )
    )


def test_explicit_empty_policy_is_rejected_instead_of_using_default():
    async def scenario(client, _):
        response = await client.post("/jobs", json=job("empty-policy", policy=""))
        assert response.status_code == 422
        assert (await client.get("/jobs/empty-policy")).status_code == 404
        assert (await client.get("/metrics")).json()["jobs_submitted"] == 0

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"workers": (WorkerEndpoint("", "http://worker:8000"),)},
        {"workers": (WorkerEndpoint("bad", "ftp://worker:8000"),)},
        {"workers": (WorkerEndpoint("bad", "no-host"),)},
        {"workers": (WorkerEndpoint("bad", "http://"),)},
        {"workers": (WorkerEndpoint("bad", "http://worker:8000/"),)},
        {"workers": (WorkerEndpoint("bad", "http://worker:8000?debug=1"),)},
        {"workers": (WorkerEndpoint("bad", "http://worker:8000#frag"),)},
        {"sample_size": 1.5},
        {"sample_size": True},
        {"max_retries": 2.5},
        {"max_retries": False},
        {"max_monitor_errors": 1.5},
        {"max_monitor_errors": True},
        {"poll_seconds": float("nan")},
        {"poll_seconds": False},
        {"poll_seconds": 10**400},
        {"timeout_seconds": float("inf")},
        {"retry_delay_seconds": float("nan")},
        {"monitor_timeout_seconds": 0},
        {"max_monitor_errors": 0},
        {"sample_size": 0},
    ],
)
def test_scheduler_settings_reject_invalid_inputs(overrides):
    settings = {"workers": (WorkerEndpoint("a100", "http://a100:8000"),)}
    settings.update(overrides)
    with pytest.raises(SchedulingConfigurationError):
        SchedulerSettings(**settings).validate()


def test_malformed_resource_reply_does_not_break_healthy_placement():
    async def fault(request, forward):
        if request.url.host == "broken" and request.url.path == "/resources":
            return httpx.Response(
                200,
                json={
                    "gpu_type": "A100",
                    "free_gpus": 1,
                    "gpu_vram_gb": 80,
                    "performance": 99,
                    "gpu_count": 1,  # allocated_gpus missing
                },
            )
        return await forward(request)

    async def scenario(client, _):
        await client.post("/jobs", json=job("malformed-resource"))
        result = await await_terminal(client, "malformed-resource")
        assert result["state"] == "completed"
        assert result["selected_worker"] == "a100"
        assert (await client.get("/metrics")).json()["worker_network_errors"] >= 1

    asyncio.run(
        run_with_network(
            ["broken", "a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
            fault=fault,
        )
    )


def test_malformed_successful_reservation_reply_reconciles_without_redispatch():
    async def fault(request, forward):
        response = await forward(request)
        if request.url.path == "/reservations":
            return httpx.Response(201, json=[])  # worker actually reserved GPU
        return response

    async def scenario(client, apps):
        await client.post("/jobs", json=job("uncertain-reserve"))
        result = await await_terminal(client, "uncertain-reserve")
        assert result["state"] == "completed"
        assert result["selected_worker"] == "a100"
        assert result["reservation_id"]
        assert (await apps["a100"].state.worker.resources())["allocated_gpus"] == 0
        assert (await client.get("/metrics")).json()["reservation_successes"] == 1

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
            fault=fault,
        )
    )


def test_ambiguous_reservation_retries_transient_status_lookup_on_same_worker():
    lookups = 0

    async def fault(request, forward):
        nonlocal lookups
        if request.url.path == "/reservations":
            await forward(request)
            return httpx.Response(201, json=[])
        if request.method == "GET" and request.url.path.startswith("/jobs/"):
            lookups += 1
            if lookups == 1:
                return httpx.Response(404, json={"detail": "not yet visible"})
        return await forward(request)

    async def scenario(client, apps):
        await client.post("/jobs", json=job("late-reservation"))
        result = await await_terminal(client, "late-reservation")
        assert result["state"] == "completed"
        assert result["selected_worker"] == "a100"
        assert lookups >= 2
        assert (await apps["a100"].state.worker.metrics())["reservations_granted"] == 1

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
            fault=fault,
        )
    )


def test_start_http_error_after_worker_started_is_reconciled():
    async def fault(request, forward):
        response = await forward(request)
        if request.url.path.endswith("/start"):
            return httpx.Response(503, json={"detail": "gateway response lost"})
        return response

    async def scenario(client, _):
        await client.post("/jobs", json=job("uncertain-start"))
        result = await await_terminal(client, "uncertain-start")
        assert result["state"] == "completed"
        assert result["selected_worker"] == "a100"

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
            fault=fault,
        )
    )


def test_reservation_reply_with_wrong_job_identity_is_reconciled():
    async def fault(request, forward):
        response = await forward(request)
        if request.url.path == "/reservations":
            await response.aread()
            forged = response.json()
            forged["job_id"] = "different-job"
            forged["reservation_id"] = "wrong-reservation"
            return httpx.Response(201, json=forged)
        return response

    async def scenario(client, apps):
        await client.post("/jobs", json=job("reservation-identity"))
        result = await await_terminal(client, "reservation-identity")
        worker_job = await apps["a100"].state.worker.get_job("reservation-identity")
        assert result["state"] == "completed", result["reason"]
        assert result["reservation_id"] == worker_job.reservation_id

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
            fault=fault,
        )
    )


def test_ambiguous_reservation_never_dispatches_to_a_second_worker():
    async def fault(request, forward):
        response = await forward(request)
        if request.url.host == "a100" and request.url.path == "/reservations":
            return httpx.Response(201, json=[])
        if request.url.host == "a100" and request.url.path.startswith("/jobs/"):
            return httpx.Response(200, json={"job_id": "wrong", "state": "reserved"})
        return response

    async def scenario(client, apps):
        await client.post("/jobs", json=job("uncertain-no-redispatch"))
        result = await await_terminal(client, "uncertain-no-redispatch")
        assert result["state"] == "failed"
        assert "reservation outcome unknown" in result["reason"]
        assert result["attempts"] == 1
        assert result["query_count"] == 2
        assert (await apps["l40"].state.worker.resources())["allocated_gpus"] == 0

    asyncio.run(
        run_with_network(
            ["a100", "l40"],
            {"a100": worker("a100", "A100", 80, 3), "l40": worker("l40", "L40", 48, 1)},
            scenario,
            fault=fault,
        )
    )


def test_monitor_rejects_malformed_worker_job_status_after_bounded_polls():
    async def fault(request, forward):
        if request.method == "GET" and request.url.path.startswith("/jobs/"):
            return httpx.Response(200, json={"state": "running"})
        return await forward(request)

    async def scenario(client, _):
        await client.post("/jobs", json=job("malformed-monitor", work_units=20))
        result = await await_terminal(client, "malformed-monitor")
        assert result["state"] == "failed"
        assert "outcome unknown" in result["reason"]
        assert result["selected_worker"] == "a100"
        assert (await client.get("/metrics")).json()["worker_network_errors"] >= 3

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
            fault=fault,
            max_monitor_errors=3,
        )
    )


def test_permanent_start_failure_is_bounded_without_double_dispatch():
    starts = []

    async def fault(request, forward):
        if request.url.path.endswith("/start"):
            starts.append(request.url.host)
            return httpx.Response(503, json={"detail": "cannot start"})
        return await forward(request)

    async def scenario(client, apps):
        await client.post("/jobs", json=job("never-started"))
        result = await await_terminal(client, "never-started")
        assert result["state"] == "failed"
        assert result["reason"] == "worker never started; outcome unknown"
        assert result["selected_worker"] == "a100"
        assert 1 < len(starts) <= 3 and set(starts) == {"a100"}
        assert (await apps["l40"].state.worker.resources())["allocated_gpus"] == 0

    asyncio.run(
        run_with_network(
            ["a100", "l40"],
            {"a100": worker("a100", "A100", 80, 3), "l40": worker("l40", "L40", 48, 1)},
            scenario,
            fault=fault,
            max_monitor_errors=3,
        )
    )


def test_monitor_timeout_for_worker_that_never_finishes():
    async def scenario(client, _):
        await client.post("/jobs", json=job("slow-job", work_units=500))
        result = await await_terminal(client, "slow-job")
        assert result["state"] == "failed"
        assert result["reason"] == "worker monitoring timed out; outcome unknown"
        assert result["selected_worker"] == "a100"

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
            monitor_timeout_seconds=0.05,
        )
    )


def test_simultaneous_duplicate_submissions_create_one_reservation():
    async def scenario(client, apps):
        replies = await asyncio.gather(
            *(client.post("/jobs", json=job("same-id")) for _ in range(12))
        )
        assert [r.status_code for r in replies].count(202) == 1
        assert [r.status_code for r in replies].count(200) == 11
        assert (await await_terminal(client, "same-id"))["state"] == "completed"
        assert (await client.get("/metrics")).json()["jobs_submitted"] == 1
        assert (await apps["a100"].state.worker.metrics())["reservations_granted"] == 1

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
        )
    )


def test_scheduler_shutdown_marks_cancelled_monitor_terminal():
    async def scenario():
        app = worker("a100", "A100", 80, 2)
        settings = SchedulerSettings(
            workers=(WorkerEndpoint("a100", "http://a100:8000"),),
            poll_seconds=0.005,
        )
        try:
            async with httpx.AsyncClient(
                transport=WorkerRouter({"a100": app})
            ) as remote:
                scheduler = Scheduler(settings, client=remote)
                await scheduler.submit(
                    SchedulingRequest(**job("shutdown", work_units=500))
                )
                for _ in range(100):
                    if scheduler.status("shutdown")["selected_worker"]:
                        break
                    await asyncio.sleep(0.005)
                assert scheduler.status("shutdown")["selected_worker"] == "a100"
                await scheduler.close()
                assert scheduler.status("shutdown")["state"] == "failed"
                assert "scheduler stopped" in scheduler.status("shutdown")["reason"]
                assert scheduler.metrics()["jobs_failed"] == 1
                with pytest.raises(SchedulerClosedError):
                    await scheduler.submit(SchedulingRequest(**job("after-close")))
        finally:
            await app.state.worker.close()

    asyncio.run(scenario())


def test_immediate_close_does_not_leave_unstarted_submission_queued():
    async def scenario():
        async with httpx.AsyncClient(transport=WorkerRouter({})) as remote:
            settings = SchedulerSettings(
                workers=(WorkerEndpoint("a100", "http://a100:8000"),),
            )
            scheduler = Scheduler(settings, client=remote)
            await scheduler.submit(SchedulingRequest(**job("unstarted")))
            await scheduler.close()  # before the task receives a CPU timeslice
            result = scheduler.status("unstarted")
            assert result["state"] == "failed"
            assert "scheduler stopped" in result["reason"]
            assert scheduler.metrics()["jobs_failed"] == 1
            with pytest.raises(SchedulerClosedError):
                await scheduler.submit(SchedulingRequest(**job("after-close")))

    asyncio.run(scenario())


def test_invalid_job_ids_and_nonfinite_json_return_422_without_dispatch():
    async def scenario(client, apps):
        for job_id in ("../escape", "bad/slash", "spaces are bad"):
            response = await client.post("/jobs", json=job(job_id))
            assert response.status_code == 422
        for invalid in (b"NaN", b"Infinity", b"-Infinity"):
            body = (
                b'{"job_id":"bad-number","gpu_count":1,"vram_gb":'
                + invalid
                + b',"work_units":1}'
            )
            response = await client.post(
                "/jobs", content=body, headers={"content-type": "application/json"}
            )
            assert response.status_code == 422
            assert response.json()["detail"]
        assert (await client.get("/metrics")).json()["jobs_submitted"] == 0
        assert (await apps["a100"].state.worker.metrics())["reservations_granted"] == 0

    asyncio.run(
        run_with_network(
            ["a100"],
            {"a100": worker("a100", "A100", 80, 2)},
            scenario,
        )
    )
