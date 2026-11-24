import httpx
import pytest

from demo import poll_job


def test_demo_reports_scheduler_failure_without_contacting_failed_worker():
    def respond(request):
        assert request.url.host == "scheduler"
        return httpx.Response(
            200,
            json={
                "job_id": "job-1",
                "state": "failed",
                "selected_worker": "worker",
                "reason": "worker unavailable",
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(RuntimeError, match="worker unavailable"),
    ):
        poll_job(
            client,
            "http://scheduler",
            {"worker": "http://worker"},
            "job-1",
            1,
            0.01,
        )


def test_demo_rejects_status_for_another_job():
    def respond(request):
        return httpx.Response(200, json={"job_id": "other-job", "state": "completed"})

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(RuntimeError, match="wrong job"),
    ):
        poll_job(client, "http://scheduler", {}, "job-1", 1, 0.01)
