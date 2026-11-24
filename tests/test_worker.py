"""Actual ASGI requests exercising worker reservation races and cleanup."""

import asyncio
import json

import pytest
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
            for body, detail in (
                (
                    job("wrong-type", compatible_gpu_types=["L40"]),
                    "incompatible_gpu_type",
                ),
                (job("wrong-memory", vram_gb=81), "insufficient_gpu_vram"),
                (job("too-large", gpu_count=3), "insufficient_free_gpus"),
            ):
                response = await client.post("/reservations", json=body)
                assert response.status_code == 409
                assert response.json()["detail"] == detail
            for body in (
                job("invalid-count", gpu_count=0),
                job("invalid-work", work_units=0),
                job("invalid-vram", vram_gb=-1),
                job(""),
            ):
                assert (
                    await client.post("/reservations", json=body)
                ).status_code == 422
            # Empty compatible_gpu_types means any GPU type.
            assert (
                await client.post(
                    "/reservations", json=job("any-type", compatible_gpu_types=[])
                )
            ).status_code == 201
        await app.state.worker.close()

    asyncio.run(check())


def test_atomic_concurrent_requests_and_duplicate_reservations():
    async def check():
        app = create_app(settings(gpu_count=2))
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            replies = await asyncio.gather(
                *(client.post("/reservations", json=job(f"r-{i}")) for i in range(25))
            )
            assert sum(response.status_code == 201 for response in replies) == 2
            assert sum(response.status_code == 409 for response in replies) == 23
            assert (await client.get("/resources")).json()["free_gpus"] == 0
            winners = [
                response.json() for response in replies if response.status_code == 201
            ]
            for winner in winners:
                response = await client.post(
                    "/reservations", json=job(winner["job_id"])
                )
                assert response.status_code == 200
                assert response.json() == winner
            collision = await client.post(
                "/reservations", json=job(winners[0]["job_id"], work_units=10)
            )
            assert collision.status_code == 409
            assert (
                collision.json()["detail"]
                == "job_id_already_used_with_different_request"
            )
            assert (await client.get("/resources")).json()["allocated_gpus"] == 2
            metrics = (await client.get("/metrics")).json()
            assert metrics["reservations_granted"] == 2
            assert metrics["reservation_conflicts"] == 24
            assert metrics["reservation_duplicates"] == 2
        await app.state.worker.close()

    asyncio.run(check())


def test_running_jobs_release_gpus_and_start_is_idempotent():
    async def check():
        app = create_app(settings())
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            initial = await client.post("/reservations", json=job(gpu_count=2))
            assert initial.status_code == 201
            reserved = initial.json()
            assert reserved["state"] == "reserved"
            assert reserved["worker_id"] == "test-a100"
            assert reserved["reservation_id"]
            assert (await client.get("/resources")).json()["free_gpus"] == 0
            assert (await client.post("/jobs/missing/start")).status_code == 404
            first, second = await asyncio.gather(
                *(client.post("/jobs/job-1/start") for _ in range(2))
            )
            assert first.status_code == second.status_code == 200
            assert first.json() == second.json()
            assert first.json()["state"] == "running"
            assert (
                await client.post("/reservations", json=job("too-soon"))
            ).status_code == 409
            await asyncio.sleep(0.10)
            final = (await client.get("/jobs/job-1")).json()
            assert final["state"] == "completed"
            assert final["reservation_id"] == reserved["reservation_id"]
            assert (await client.get("/resources")).json()["free_gpus"] == 2
            repeated_start = await client.post("/jobs/job-1/start")
            assert repeated_start.status_code == 200
            assert repeated_start.json()["state"] == "completed"
            metrics = (await client.get("/metrics")).json()
            assert metrics["jobs_started"] == metrics["jobs_completed"] == 1
            assert metrics["jobs_failed"] == 0
            assert metrics["busy_gpu_seconds"] > 0
            assert 0 < metrics["gpu_utilization"] <= 1
        await app.state.worker.close()

    asyncio.run(check())


def test_concurrent_identical_job_id_has_one_reservation():
    async def check():
        app = create_app(settings(gpu_count=1))
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            replies = await asyncio.gather(
                *(
                    client.post("/reservations", json=job("identical"))
                    for _ in range(15)
                )
            )
            assert sorted(response.status_code for response in replies) == [
                200
            ] * 14 + [201]
            assert len({response.json()["reservation_id"] for response in replies}) == 1
            resources = (await client.get("/resources")).json()
            assert resources["allocated_gpus"] == 1
            assert resources["active_jobs"] == 1
            metrics = (await client.get("/metrics")).json()
            assert metrics["reservations_granted"] == 1
            assert metrics["reservation_duplicates"] == 14
        await app.state.worker.close()

    asyncio.run(check())


def test_reservation_lease_expiry_releases_capacity_and_blocks_late_start():
    async def check():
        app = create_app(settings(reservation_lease_seconds=0.03))
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            reserved = await client.post("/reservations", json=job(gpu_count=2))
            assert reserved.status_code == 201
            await asyncio.sleep(0.07)
            expired = (await client.get("/jobs/job-1")).json()
            assert expired["state"] == "failed"
            assert expired["error"] == "reservation_expired"
            assert (await client.get("/resources")).json()["free_gpus"] == 2
            assert (await client.post("/jobs/job-1/start")).status_code == 409
            # A retry retains its original expired reservation; it cannot create
            # a second execution for the same job ID on this worker.
            duplicate = await client.post("/reservations", json=job(gpu_count=2))
            assert duplicate.status_code == 200
            assert duplicate.json() == expired
            replacement = await client.post(
                "/reservations", json=job("replacement", gpu_count=2)
            )
            assert replacement.status_code == 201
            metrics = (await client.get("/metrics")).json()
            assert metrics["reservations_expired"] == 1
            assert metrics["jobs_failed"] == 1
        await app.state.worker.close()

    asyncio.run(check())


def test_cancelled_execution_fails_and_releases_once():
    async def check():
        app = create_app(settings(sim_seconds_per_work_unit=100.0))
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            assert (
                await client.post("/reservations", json=job(gpu_count=2))
            ).status_code == 201
            assert (await client.post("/jobs/job-1/start")).status_code == 200
            # Shutdown can happen before the new task gets its first timeslice.
            await app.state.worker.close()
            assert (await client.get("/jobs/job-1")).json()["state"] == "failed"
            assert (await client.get("/resources")).json()["allocated_gpus"] == 0
            assert (await client.get("/metrics")).json()["jobs_failed"] == 1
            await app.state.worker.close()
            assert (await client.get("/metrics")).json()["jobs_failed"] == 1
            assert (
                await client.post("/reservations", json=job("late"))
            ).status_code == 409

    asyncio.run(check())


def test_started_job_outlives_reservation_lease():
    async def check():
        app = create_app(
            settings(reservation_lease_seconds=0.02, sim_seconds_per_work_unit=0.10)
        )
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            assert (
                await client.post("/reservations", json=job(gpu_count=2))
            ).status_code == 201
            assert (await client.post("/jobs/job-1/start")).status_code == 200
            await asyncio.sleep(0.07)
            assert (await client.get("/jobs/job-1")).json()["state"] == "running"
            assert (await client.get("/resources")).json()["allocated_gpus"] == 2
            assert (await client.get("/metrics")).json()["reservations_expired"] == 0
        await app.state.worker.close()

    asyncio.run(check())


def test_shutdown_releases_unstarted_reservation():
    async def check():
        app = create_app(settings())
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            assert (
                await client.post("/reservations", json=job(gpu_count=2))
            ).status_code == 201
            await app.state.worker.close()
            response = await client.get("/jobs/job-1")
            assert response.json()["state"] == "failed"
            assert response.json()["error"] == "worker_shutdown"
            resources = (await client.get("/resources")).json()
            assert resources["allocated_gpus"] == 0
            assert resources["active_jobs"] == 0
            assert resources["free_gpus"] == 2

    asyncio.run(check())


def test_settings_read_all_worker_environment_variables():
    configured = WorkerSettings.from_env(
        {
            "WORKER_ID": "node-2",
            "GPU_TYPE": "L40",
            "GPU_COUNT": "4",
            "GPU_VRAM_GB": "48",
            "GPU_PERFORMANCE": "1.5",
            "SIM_SECONDS_PER_WORK_UNIT": "0.25",
            "RESERVATION_LEASE_SECONDS": "12",
        }
    )
    assert configured.worker_id == "node-2"
    assert configured.gpu_type == "L40"
    assert configured.gpu_count == 4
    assert configured.gpu_vram_gb == 48
    assert configured.gpu_performance == 1.5
    assert configured.sim_seconds_per_work_unit == 0.25
    assert configured.reservation_lease_seconds == 12


@pytest.mark.parametrize("gpu_count", [0, -1, True, 2.5, "2"])
def test_worker_settings_requires_whole_gpu_counts(gpu_count):
    with pytest.raises(ValueError, match="GPU_COUNT must be a positive integer"):
        WorkerSettings(gpu_count=gpu_count)


@pytest.mark.parametrize(
    "invalid_value", [0, -1, True, float("nan"), float("inf"), "1.0"]
)
@pytest.mark.parametrize(
    "field",
    [
        "gpu_vram_gb",
        "gpu_performance",
        "sim_seconds_per_work_unit",
        "reservation_lease_seconds",
    ],
)
def test_worker_settings_rejects_invalid_positive_floats(field, invalid_value):
    with pytest.raises(ValueError, match="must be a finite positive number"):
        WorkerSettings(**{field: invalid_value})


def test_rejects_nonfinite_numbers_and_malformed_json_with_422():
    async def check():
        app = create_app(settings())
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            for value in ("NaN", "Infinity", "-Infinity"):
                for field in ("vram_gb", "work_units", "gpu_count"):
                    body = job(f"nonfinite-{field}")
                    body[field] = 1
                    payload = json.dumps(body).replace(
                        f'"{field}": 1', f'"{field}": {value}'
                    )
                    response = await client.post(
                        "/reservations",
                        content=payload,
                        headers={"content-type": "application/json"},
                    )
                    assert response.status_code == 422, (value, field, response.text)
                    assert response.json()["detail"][0]["type"]
            response = await client.post(
                "/reservations",
                content='{"job_id": "broken",',
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 422
            assert response.json()["detail"][0]["type"] == "json_invalid"
            assert (await client.get("/resources")).json()["free_gpus"] == 2
        await app.state.worker.close()

    asyncio.run(check())


@pytest.mark.parametrize(
    "invalid_id",
    [
        "job?query",
        "job#fragment",
        "job/part",
        "job%2Fpart",
        ".",
        "..",
        "job\\other",
        "a b",
        "a\n",
        "a" * 129,
    ],
)
def test_rejects_job_ids_that_cannot_round_trip_in_http_path(invalid_id):
    async def check():
        app = create_app(settings())
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            response = await client.post("/reservations", json=job(invalid_id))
            assert response.status_code == 422, response.text
            assert (await client.get("/resources")).json()["free_gpus"] == 2
        await app.state.worker.close()

    asyncio.run(check())


@pytest.mark.parametrize("field", ["gpu_count", "vram_gb", "work_units"])
def test_boolean_gpu_or_workload_quantities_are_not_treated_as_one(field):
    async def check():
        app = create_app(settings())
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://worker"
        ) as client:
            payload = job(f"boolean-{field}", **{field: True})
            response = await client.post("/reservations", json=payload)
            assert response.status_code == 422
            assert response.json()["detail"][0]["loc"] == ["body", field]
            assert (await client.get("/resources")).json()["free_gpus"] == 2
        await app.state.worker.close()

    asyncio.run(check())
