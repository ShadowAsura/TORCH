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




def run(args):
    scheduler = args.url
    workers = worker_urls(args.worker)
    policies = ("centralized", "sampling") if args.policy == "both" else (args.policy,)
    with httpx.Client(timeout=5.0) as client:

        initial = {}

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
            check(worker_job.get("worker_id") == selected, "worker job mismatch")

            after = request(client, "GET", workers[selected], "/resources")
            check(
                all(
                    after[key] == before[selected][key]
                    for key in ("free_gpus", "allocated_gpus")
                ),
                f"{selected} did not release reserved GPUs",
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
                    "free_gpus_after": after["free_gpus"],
                    "scheduling_latency_ms": job.get("scheduling_latency_ms"),
                }
            )





