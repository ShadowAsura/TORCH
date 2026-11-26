"""
Submit a seeded workload to a running TORCH cluster and check the results.

Start a cluster first (see the compose files in this folder), then run from
the repository root:

    python -m deploy.demo --cluster deploy/centralized.json
    python -m deploy.demo --cluster deploy/gossip.json --jobs 30 --seed 3

Jobs are submitted at their workload arrival times multiplied by
`--execution-scale` (keep it equal to the cluster's TORCH_EXECUTION_SCALE).
The demo waits until every job completes, checks that every GPU is free
again and that no message failed while it ran, and prints per-job results
and the traffic it caused. Counters are compared with a snapshot taken just
before submitting, so startup traffic (for example, gossip to peers that
were not up yet) does not count. It exits nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import httpx

from torchsched.codec import to_data
from torchsched.workloads import poisson_workload


def fail(message: str) -> None:
    print(f"FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def wait_healthy(client: httpx.Client, urls: dict[str, str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    pending = dict(urls)
    while pending:
        for node_id, url in list(pending.items()):
            try:
                if client.get(f"{url}/health").json().get("ok"):
                    del pending[node_id]
            except (httpx.HTTPError, ValueError):
                pass
        if pending and time.monotonic() > deadline:
            fail(f"nodes not healthy: {sorted(pending)}")
        if pending:
            time.sleep(0.5)


def traffic(client: httpx.Client, urls: dict[str, str]) -> dict[str, Counter]:
    """
    Messages sent, bytes sent, and send failures, summed over all nodes.
    """
    totals = {"messages_sent": Counter(), "bytes_sent": Counter(), "send_failures": Counter()}
    for url in urls.values():
        metrics = client.get(f"{url}/metrics").json()
        for key, total in totals.items():
            total.update(metrics[key])
    return totals


def observations(client: httpx.Client, urls: dict[str, str]) -> list[dict]:
    found = []
    for url in urls.values():
        found += client.get(f"{url}/observations").json()["observations"]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cluster", required=True, help="cluster config JSON")
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arrival-rate", type=float, default=5.0, help="jobs per workload second")
    parser.add_argument("--execution-scale", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for jobs")
    args = parser.parse_args()

    config = json.loads(Path(args.cluster).read_text())
    urls = config["published"]
    if config["design"] == "centralized":
        targets = [config.get("centralized", {}).get("coordinator", "coordinator")]
    else:
        targets = [worker["id"] for worker in config["workers"]]
    workers = [worker["id"] for worker in config["workers"]]

    with httpx.Client(timeout=5) as client:
        wait_healthy(client, urls, timeout=60)
        print(f"{config['design']} cluster healthy: {', '.join(urls)}")
        if config["design"] == "gossip":
            time.sleep(2 * config["gossip"]["interval"])  # let views fill in

        before = traffic(client, urls)
        workload = poisson_workload(args.seed, args.jobs, args.arrival_rate)
        started = time.monotonic()
        for index, submission in enumerate(workload):
            delay = started + submission.at * args.execution_scale - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            target = targets[index % len(targets)]
            response = client.post(f"{urls[target]}/jobs", json=to_data(submission.job))
            if response.status_code != 202:
                fail(f"submitting {submission.job.job_id} to {target}: {response.text}")
        print(f"submitted {len(workload)} jobs to {', '.join(targets)}")

        deadline = time.monotonic() + args.timeout
        while True:
            seen = observations(client, urls)
            completed = {o["metric"]["job_id"] for o in seen if o["type"] == "JobCompleted"}
            failed = [o for o in seen if o["type"] == "JobFailed"]
            if failed:
                fail(f"jobs failed: {failed}")
            if len(completed) == len(workload):
                break
            if time.monotonic() > deadline:
                fail(f"only {len(completed)} of {len(workload)} jobs completed")
            time.sleep(0.25)

        accepted = {o["metric"]["job_id"]: o for o in seen if o["type"] == "JobAccepted"}
        starts = {o["metric"]["job_id"]: o for o in seen if o["type"] == "JobStarted"}
        finishes = {o["metric"]["job_id"]: o for o in seen if o["type"] == "JobCompleted"}
        forwards = Counter(o["metric"]["job_id"] for o in seen if o["type"] == "JobForwarded")
        print(f"\n{'job':8s} {'origin':14s} {'worker':14s} {'gpu':6s} {'delay':>7s} {'run':>6s} {'fwd':>4s}")
        for submission in workload:
            job_id = submission.job.job_id
            start, finish = starts[job_id], finishes[job_id]
            print(
                f"{job_id:8s} {accepted[job_id]['node_id']:14s} {start['metric']['worker']:14s} "
                f"{start['metric']['gpu_type']:6s} {start['time'] - accepted[job_id]['time']:7.3f} "
                f"{finish['time'] - start['time']:6.2f} {forwards[job_id]:4d}"
            )
        delays = [starts[j]["time"] - accepted[j]["time"] for j in starts]
        print(f"\nmean scheduling delay {mean(delays):.3f} s (wall clock, includes network)")

        for worker in workers:
            status = client.get(f"{urls[worker]}/status").json()["status"]
            capacity = [group["count"] for group in status["spec"]["gpus"]]
            if status["free_gpus"] != capacity:
                fail(f"{worker} still holds GPUs: free {status['free_gpus']} of {capacity}")
        print("every GPU released")

        after = traffic(client, urls)
        sent = after["messages_sent"] - before["messages_sent"]
        size = after["bytes_sent"] - before["bytes_sent"]
        failures = after["send_failures"] - before["send_failures"]
        print(f"traffic during the demo: {sum(sent.values())} messages, {sum(size.values()) / 1000:.1f} KB")
        for kind, count in sent.most_common():
            print(f"  {kind:15s} {count:6d}  {size[kind] / 1000:8.1f} KB")
        if failures:
            fail(f"messages failed to send: {dict(failures)}")
        print("no failed messages\nOK")


if __name__ == "__main__":
    main()
