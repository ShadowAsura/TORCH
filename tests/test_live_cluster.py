"""Opt-in tests for the running, idle Docker Compose cluster."""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import httpx
import pytest

from demo import worker_urls

pytestmark = pytest.mark.skipif(
    os.getenv("TORCH_LIVE_TESTS") != "1",
    reason="requires an idle running TORCH Docker Compose cluster",
)


@pytest.fixture
def client():
    with httpx.Client(timeout=5, trust_env=False) as session:
        yield session


def read_json(client, url):
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def wait_for_job(client, url, states, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = read_json(client, url)
        if record["state"] in states:
            return record
        time.sleep(0.05)
    pytest.fail(f"{url} never reached {states}")


def test_concurrent_reservations_over_real_http(client):
    worker = worker_urls(None)["worker-l40"]
    before = read_json(client, f"{worker}/resources")
    assert before["allocated_gpus"] == 0
    assert before["free_gpus"] > 0
    before_metrics = read_json(client, f"{worker}/metrics")
    prefix = uuid4().hex
    requests = [
        {
            "job_id": f"race-{prefix}-{i}",
            "gpu_count": before["free_gpus"],
            "vram_gb": 1,
            "compatible_gpu_types": [before["gpu_type"]],
            "work_units": 2,
        }
        for i in range(12)
    ]
    with ThreadPoolExecutor(max_workers=12) as pool:
        replies = list(
            pool.map(
                lambda body: client.post(f"{worker}/reservations", json=body), requests
            )
        )

    assert sorted(response.status_code for response in replies) == [201] + [409] * 11
    winner = next(
        i for i, response in enumerate(replies) if response.status_code == 201
    )
    reservation = replies[winner].json()
    reserved = read_json(client, f"{worker}/resources")
    assert reserved["free_gpus"] == 0
    assert reserved["allocated_gpus"] == before["free_gpus"]
    assert reserved["available_vram_gb"] == 0

    duplicate = client.post(f"{worker}/reservations", json=requests[winner])
    assert duplicate.status_code == 200
    assert duplicate.json()["reservation_id"] == reservation["reservation_id"]
    job_url = f"{worker}/jobs/{reservation['job_id']}"
    client.post(f"{job_url}/start").raise_for_status()
    assert (
        wait_for_job(client, job_url, {"completed", "failed"})["state"] == "completed"
    )
    assert read_json(client, f"{worker}/resources") == before
    after_metrics = read_json(client, f"{worker}/metrics")
    assert after_metrics["jobs_started"] == before_metrics["jobs_started"] + 1
    assert after_metrics["jobs_completed"] == before_metrics["jobs_completed"] + 1


def submit_live_job(client, scheduler, gpu_type, work_units=2):
    job_id = f"live-{uuid4().hex}"
    response = client.post(
        f"{scheduler}/jobs",
        json={
            "job_id": job_id,
            "gpu_count": 1,
            "vram_gb": 1,
            "compatible_gpu_types": [gpu_type],
            "work_units": work_units,
            "policy": "centralized",
        },
    )
    assert response.status_code == 202, response.text
    return job_id


def wait_for_health(client, worker, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if read_json(client, f"{worker}/health")["status"] == "ok":
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    pytest.fail(f"worker {worker} did not recover")


def test_worker_crash_and_recovery_over_real_http(client):
    workers = worker_urls(None)
    worker = workers["worker-t4"]
    scheduler = os.getenv(
        "TORCH_SCHEDULER_URL",
        f"http://localhost:{os.getenv('TORCH_SCHEDULER_PORT', '8000')}",
    )
    initial = read_json(client, f"{worker}/resources")
    assert initial["allocated_gpus"] == 0
    job_id = submit_live_job(client, scheduler, "T4", work_units=300)
    job_url = f"{scheduler}/jobs/{job_id}"
    record = wait_for_job(client, job_url, {"running", "failed", "completed"})
    assert record["state"] == "running"
    assert record["selected_worker"] == "worker-t4"
    before = read_json(client, f"{scheduler}/metrics")

    try:
        subprocess.run(
            ["docker", "compose", "kill", "worker-t4"], check=True, timeout=30
        )
        result = wait_for_job(client, job_url, {"completed", "failed"}, timeout=40)
        assert result["state"] == "failed"
        assert result["selected_worker"] == "worker-t4"
        for name, url in workers.items():
            if name != "worker-t4":
                assert client.get(f"{url}/jobs/{job_id}").status_code == 404

        healthy_id = submit_live_job(client, scheduler, "A100")
        healthy = wait_for_job(
            client, f"{scheduler}/jobs/{healthy_id}", {"completed", "failed"}
        )
        assert healthy["state"] == "completed"
        assert healthy["selected_worker"] == "worker-a100"
        after = read_json(client, f"{scheduler}/metrics")
        assert after["worker_network_errors"] > before["worker_network_errors"]
    finally:
        subprocess.run(
            ["docker", "compose", "start", "worker-t4"], check=True, timeout=30
        )
        wait_for_health(client, worker)

    recovered_id = submit_live_job(client, scheduler, "T4")
    recovered = wait_for_job(
        client, f"{scheduler}/jobs/{recovered_id}", {"completed", "failed"}
    )
    assert recovered["state"] == "completed"
    assert recovered["selected_worker"] == "worker-t4"
    assert read_json(client, f"{worker}/resources") == initial
