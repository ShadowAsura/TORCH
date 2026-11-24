"""Smoke-test TORCH through real scheduler and worker HTTP requests."""

import argparse
import json
import math
import os
import sys
import time
from uuid import uuid4

import httpx


def request(client, method, base, path, **kwargs):
    response = client.request(method, base.rstrip("/") + path, **kwargs)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not return a JSON object")
    return data


def worker_urls(entries):
    if entries is None:
        return {
            name: f"http://localhost:{os.getenv(env, port)}"
            for name, env, port in (
                ("worker-l40", "TORCH_L40_PORT", "8001"),
                ("worker-a100", "TORCH_A100_PORT", "8002"),
                ("worker-t4", "TORCH_T4_PORT", "8003"),
            )
        }
    workers = {}
    for entry in entries:
        name, sep, url = entry.partition("=")
        if not sep or not name or not url.startswith(("http://", "https://")):
            raise ValueError(f"invalid --worker {entry!r}; use NAME=http://HOST:PORT")
        if name in workers:
            raise ValueError(f"duplicate worker: {name}")
        workers[name] = url
    return workers


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def poll_job(client, scheduler, workers, job_id, timeout, interval):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request(client, "GET", scheduler, f"/jobs/{job_id}")
        check(job.get("job_id") == job_id, "scheduler returned the wrong job")
        if job.get("state") == "failed":
            raise RuntimeError(f"{job_id} failed: {job.get('reason') or job}")
        selected = job.get("selected_worker")
        check(
            not selected or selected in workers, f"unknown selected worker: {selected}"
        )
        worker_job = (
            request(client, "GET", workers[selected], f"/jobs/{job_id}")
            if selected
            else {}
        )
        if selected:
            check(worker_job.get("job_id") == job_id, "worker returned the wrong job")
        if worker_job.get("state") == "failed":
            raise RuntimeError(f"{job_id} failed: {job.get('reason') or worker_job}")
        if job.get("state") == worker_job.get("state") == "completed":
            return job, worker_job
        time.sleep(interval)
    raise RuntimeError(f"{job_id} did not complete within {timeout:g}s")


def run(args):
    scheduler = args.url
    workers = worker_urls(args.worker)
    policies = ("centralized", "sampling") if args.policy == "both" else (args.policy,)
    with httpx.Client(timeout=5.0) as client:
        check(
            request(client, "GET", scheduler, "/health").get("status") == "ok",
            "scheduler is unhealthy",
        )
        advertised = request(client, "GET", scheduler, "/workers")["workers"]
        check(
            {item["worker_id"] for item in advertised} == set(workers),
            "worker names differ from scheduler configuration; use --worker NAME=URL",
        )

        initial = {}
        for name, url in workers.items():
            health = request(client, "GET", url, "/health")
            check(
                health.get("status") == "ok" and health.get("worker_id") == name,
                f"{name} is unhealthy",
            )
            initial[name] = request(client, "GET", url, "/resources")
            check(initial[name].get("worker_id") == name, f"{name} resources mismatch")

        gpu_types = args.gpu_type or sorted({r["gpu_type"] for r in initial.values()})
        results = []
        for policy in policies:
            before = {
                name: request(client, "GET", url, "/resources")
                for name, url in workers.items()
            }
            job_id = f"demo-{policy}-{uuid4().hex[:12]}"
            payload = {
                "job_id": job_id,
                "gpu_count": args.gpu_count,
                "vram_gb": args.vram_gb,
                "compatible_gpu_types": gpu_types,
                "work_units": args.work_units,
                "policy": policy,
            }
            accepted = request(client, "POST", scheduler, "/jobs", json=payload)
            check(accepted.get("job_id") == job_id, "scheduler returned wrong job ID")
            job, worker_job = poll_job(
                client, scheduler, workers, job_id, args.timeout, args.interval
            )
            selected = job["selected_worker"]
            count = job.get("query_count")
            queried = job.get("queried_workers")
            check(
                job.get("policy") == policy
                and isinstance(count, int)
                and count > 0
                and isinstance(queried, list)
                and len(queried) == count,
                f"invalid policy or query count for {job_id}: {job}",
            )
            check(
                selected in queried and all(name in workers for name in queried),
                "scheduler reported an inconsistent worker selection",
            )
            check(
                worker_job.get("worker_id") == selected
                and worker_job.get("gpu_count") == args.gpu_count
                and bool(job.get("reservation_id"))
                and job["reservation_id"] == worker_job.get("reservation_id"),
                "worker reservation does not match the scheduled job",
            )

            after = {
                name: request(client, "GET", url, "/resources")
                for name, url in workers.items()
            }
            for name in workers:
                check(
                    all(
                        after[name][key] == before[name][key]
                        for key in ("free_gpus", "allocated_gpus")
                    ),
                    f"{name} did not return to its initial GPU capacity",
                )
            results.append(
                {
                    "job_id": job_id,
                    "policy": policy,
                    "state": job["state"],
                    "selected_worker": selected,
                    "query_count": count,
                    "queried_workers": queried,
                    "worker_state": worker_job["state"],
                    "free_gpus_before": before[selected]["free_gpus"],
                    "free_gpus_after": after[selected]["free_gpus"],
                    "scheduling_latency_ms": job.get("scheduling_latency_ms"),
                }
            )

        return {
            "jobs": results,
            "workers": initial,
            "scheduler_metrics": request(client, "GET", scheduler, "/metrics"),
            "worker_metrics": {
                name: request(client, "GET", url, "/metrics")
                for name, url in workers.items()
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.getenv(
            "TORCH_SCHEDULER_URL",
            f"http://localhost:{os.getenv('TORCH_SCHEDULER_PORT', '8000')}",
        ),
    )
    parser.add_argument(
        "--worker",
        action="append",
        metavar="NAME=URL",
        help="repeat for each worker; defaults to localhost:8001-8003",
    )
    parser.add_argument(
        "--policy", choices=("both", "centralized", "sampling"), default="both"
    )
    parser.add_argument("--gpu-type", action="append")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--vram-gb", type=float, default=8.0)
    parser.add_argument("--work-units", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.gpu_count < 1 or not all(
        math.isfinite(x) and x > 0
        for x in (args.vram_gb, args.work_units, args.timeout, args.interval)
    ):
        parser.error("job size, timeout and interval must be positive")
    try:
        result = run(args)
    except (httpx.HTTPError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"TORCH network demo FAILED: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("TORCH network demo PASS")
        for job in result["jobs"]:
            print(
                f"  {job['policy']}: {job['job_id']} completed on "
                f"{job['selected_worker']} (queried {job['query_count']}: "
                f"{', '.join(job['queried_workers'])}); "
                f"free GPUs {job['free_gpus_before']} -> {job['free_gpus_after']}"
            )
        metrics = result["scheduler_metrics"]
        print(
            f"  metrics: submitted={metrics['jobs_submitted']}, "
            f"completed={metrics['jobs_completed']}, "
            f"worker queries={metrics['worker_state_queries']}"
        )
        for name, metrics in result["worker_metrics"].items():
            print(
                f"  {name}: completed={metrics['jobs_completed']}, "
                f"allocated={metrics['allocated_gpus']}, "
                f"utilization={metrics['gpu_utilization']:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
