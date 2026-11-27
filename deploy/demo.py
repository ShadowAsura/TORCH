"""
Submit a seeded workload to a running TORCH cluster and check the results.

Start a cluster first (see compose.yaml in this folder), then run from the
repository root with the same design:

    python -m deploy.demo --cluster deploy/cluster.json --design centralized
    python -m deploy.demo --cluster deploy/cluster.json --design gossip --jobs 30 --seed 3

Jobs are submitted at their workload arrival times multiplied by
`--execution-scale` (keep it equal to the cluster's TORCH_EXECUTION_SCALE).
The demo waits until every job completes, checks that every GPU is free
again and that no message failed while it ran, and prints per-job results
and the traffic it caused. Counters are compared with a snapshot taken just
before submitting, so startup traffic (for example, gossip to peers that
were not up yet) does not count. Job ids carry a per-run prefix, because
schedulers remember finished jobs and would ignore reused ids, so the demo
can run again on the same cluster. It exits nonzero if any check fails.

`--scenario FILE` injects the faults a scenario file describes (see
`torchsched.scenarios`; examples are in deploy/scenarios). Its times are
workload seconds, scaled by `--execution-scale` like arrivals, and random
crashes are picked with `--seed`:

    python -m deploy.demo --cluster deploy/cluster.json --design gossip \\
        --scenario deploy/scenarios/crash-restart.json

* A crash kills the node's container (`docker compose kill`, so the whole
  node goes down at once); a restart starts it again with empty state.
  Just before each kill the demo saves the node's jobs, observations and
  counters (`GET /jobs`, `/observations`, `/metrics`), which live in memory.
* A partition disconnects the nodes' containers from the cluster network
  (and connects a group of several to a network of its own), so they keep
  running but cannot reach, or be reached by, anyone else. The demo cannot
  reach them either until the partition heals, so here every partition
  must heal, and a node cannot crash while it is cut off.
* Losses are installed on every node with `PUT /faults` (the cluster must
  run with `--allow-fault-injection`, as compose.yaml does) and cleared
  when the demo ends.

Jobs are submitted only to nodes the scenario never crashes or cuts off.
Jobs a crashed node's scheduler held (forwarded to it) are lost with it, as
there is no retry policy yet; they are reported but not counted as
failures. Failed sends are expected with crashes and partitions, as are
dropped messages with losses; both are reported.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import mean

import httpx

from torchsched.codec import to_data
from torchsched.scenarios import Partition, Scenario, loss_to_data, scenario_argument
from torchsched.workloads import poisson_workload

COUNTERS = ("messages_sent", "bytes_sent", "send_failures", "dropped")
SEND_TIMEOUT = 5.0  # HttpRuntime's default request timeout


def fail(message: str) -> None:
    print(f"FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def healthy(client: httpx.Client, urls: dict[str, str], timeout: float) -> set[str]:
    """
    Wait up to `timeout` seconds for `urls` to report healthy; return the
    nodes that did not.
    """
    deadline = time.monotonic() + timeout
    pending = dict(urls)
    while pending:
        for node_id, url in list(pending.items()):
            try:
                if client.get(f"{url}/health").json().get("ok"):
                    del pending[node_id]
            except (httpx.HTTPError, ValueError):
                pass
        if not pending or time.monotonic() > deadline:
            break
        time.sleep(0.5)
    return set(pending)


def wait_healthy(client: httpx.Client, urls: dict[str, str], timeout: float) -> None:
    if unhealthy := healthy(client, urls, timeout):
        fail(f"nodes not healthy: {sorted(unhealthy)}")


def counters(client: httpx.Client, url: str) -> dict[str, Counter]:
    """
    One node's messages sent, bytes sent, send failures and injected drops,
    by kind.
    """
    metrics = client.get(f"{url}/metrics").json()
    return {key: Counter(metrics[key]) for key in COUNTERS}


def node_observations(client: httpx.Client, url: str) -> list[dict]:
    return client.get(f"{url}/observations").json()["observations"]


class ScenarioRunner:
    """
    Inject a resolved scenario's faults into the Compose cluster: losses up
    front, then crashes, restarts, partitions and heals at their times, in
    a background thread.
    """

    def __init__(
        self,
        scenario: Scenario,
        urls: dict[str, str],
        compose: str,
        scale: float,
        seed: int,
    ):
        self.scenario = scenario
        self.urls = urls
        self.compose = ["docker", "compose", "-f", compose]
        self.scale = scale
        self.seed = seed
        events: list[tuple[float, str, object]] = []
        for crash in scenario.crashes:
            events.append((crash.at, "crash", crash.node))
            if crash.restart_at is not None:
                events.append((crash.restart_at, "restart", crash.node))
        for index, partition in enumerate(scenario.partitions):
            events.append((partition.at, "cut", index))
            assert partition.heal_at is not None
            events.append((partition.heal_at, "heal", index))
        # At equal times, heal and restart before cutting and crashing.
        order = {"heal": 0, "restart": 1, "cut": 2, "crash": 3}
        self.events = sorted(events, key=lambda e: (e[0], order[e[1]]))
        self.down: set[str] = set()  # crashed, not restarted
        self.cut: set[str] = set()  # partitioned, not healed
        self.lost_with: dict[str, str] = {}  # job id -> the node whose scheduler held it
        self.saved_observations: list[dict] = []
        self.saved_counters: dict[str, dict[str, Counter]] = {
            node_id: {key: Counter() for key in COUNTERS} for node_id in scenario.crashed
        }
        self.errors: list[str] = []
        self._active: dict[int, str | None] = {}  # partition index -> its own network
        self._containers: dict[str, str] = {}
        self._network = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """
        Install losses on every node and start the clock: scenario time 0
        is now.
        """
        self._started = time.monotonic()
        self._wall_start = time.time()
        if self.scenario.partitions:
            self._find_containers()
        with httpx.Client(timeout=5) as client:
            for node_id in self.urls:
                self._install_losses(client, node_id)
        self._thread.start()

    def join(self) -> None:
        self._thread.join()

    @property
    def finished(self) -> bool:
        """
        Whether every event has happened.
        """
        return not self._thread.is_alive()

    def cleanup(self) -> None:
        """
        Stop, heal any partition still in place, and clear losses. Crashed
        nodes stay down.
        """
        self._stop.set()
        self._thread.join(timeout=30)
        for index in list(self._active):
            try:
                self._heal(index)
            except (subprocess.CalledProcessError, OSError) as error:
                print(f"could not heal partition {index}: {error}", file=sys.stderr)
        if self.scenario.losses:
            with httpx.Client(timeout=5) as client:
                for node_id, url in self.urls.items():
                    if node_id not in self.down:
                        try:
                            client.put(f"{url}/faults", json={"losses": []})
                        except httpx.HTTPError:
                            pass

    def _run(self) -> None:
        for at, action, target in self.events:
            delay = self._started + at * self.scale - time.monotonic()
            if self._stop.wait(max(delay, 0)):
                return
            try:
                match action:
                    case "crash":
                        self._save(str(target))
                        self._docker_compose("kill", str(target))
                        self.down.add(str(target))
                        label = f"crashed {target}"
                    case "restart":
                        self._docker_compose("start", str(target))
                        self.down.discard(str(target))
                        self._after_restart(str(target))
                        label = f"restarted {target}"
                    case "cut":
                        self._cut(int(target))
                        label = f"cut off {self._names(int(target))}"
                    case _:
                        self._heal(int(target))
                        label = f"healed {self._names(int(target))}"
            except (httpx.HTTPError, subprocess.CalledProcessError, OSError) as error:
                self.errors.append(f"{action} {target}: {error}")
                return
            print(f"{label} at +{time.monotonic() - self._started:.2f} s")

    def _names(self, index: int) -> str:
        return ", ".join(sorted(self.scenario.partitions[index].nodes))

    def _save(self, node_id: str) -> None:
        url = self.urls[node_id]
        with httpx.Client(timeout=5) as client:
            # A node crashing again soon after a restart may still be booting.
            if healthy(client, {node_id: url}, timeout=30):
                raise OSError(f"{node_id} is not healthy, so its state cannot be saved")
            for job_id in client.get(f"{url}/jobs").json()["jobs"]:
                self.lost_with[job_id] = node_id
            self.saved_observations += node_observations(client, url)
            for key, counter in counters(client, url).items():
                self.saved_counters[node_id][key] += counter

    def _after_restart(self, node_id: str) -> None:
        """
        A restarted node has forgotten its injected losses; install them
        again once it is up.
        """
        if not self.scenario.losses:
            return
        with httpx.Client(timeout=5) as client:
            if healthy(client, {node_id: self.urls[node_id]}, timeout=30):
                raise OSError(f"{node_id} did not come back up")
            self._install_losses(client, node_id)

    def _install_losses(self, client: httpx.Client, node_id: str) -> None:
        """
        Send the scenario's losses to `node_id`, with scenario times turned
        into wall-clock times.
        """
        if not self.scenario.losses:
            return
        losses = []
        for loss in self.scenario.losses:
            data = loss_to_data(loss)
            data["start"] = self._wall_start + loss.start * self.scale
            if loss.end is not None:
                data["end"] = self._wall_start + loss.end * self.scale
            losses.append(data)
        seed = self.seed * 1000 + sorted(self.urls).index(node_id)
        response = client.put(f"{self.urls[node_id]}/faults", json={"losses": losses, "seed": seed})
        if response.status_code == 403:
            raise OSError(f"{node_id} runs without --allow-fault-injection")
        response.raise_for_status()

    def _find_containers(self) -> None:
        nodes = self.scenario.partitioned
        for node_id in sorted(nodes):
            output = self._docker_compose("ps", "-q", node_id)
            self._containers[node_id] = output.strip()
        first = self._containers[min(nodes)]
        networks = self._docker(
            "inspect", "-f", "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}", first
        ).split()
        if len(networks) != 1:
            raise OSError(f"expected one cluster network, found {networks}")
        self._network = networks[0]

    def _cut(self, index: int) -> None:
        partition: Partition = self.scenario.partitions[index]
        own = None
        if len(partition.nodes) > 1:
            own = f"torch-partition-{index}-{int(self._wall_start)}"
            self._docker("network", "create", own)
        self._active[index] = own
        for node_id in sorted(partition.nodes):
            self._docker("network", "disconnect", self._network, self._containers[node_id])
            if own is not None:
                self._docker(
                    "network", "connect", "--alias", node_id, own, self._containers[node_id]
                )
            self.cut.add(node_id)

    def _heal(self, index: int) -> None:
        own = self._active.pop(index)
        for node_id in sorted(self.scenario.partitions[index].nodes):
            container = self._containers[node_id]
            # Join the cluster network before leaving the group's own, so the
            # node is never on no network at all.
            self._docker("network", "connect", "--alias", node_id, self._network, container)
            if own is not None:
                self._docker("network", "disconnect", own, container)
            self.cut.discard(node_id)
        if own is not None:
            self._docker("network", "rm", own)

    def _docker_compose(self, *command: str) -> str:
        return self._docker(*self.compose[1:], *command)

    def _docker(self, *command: str) -> str:
        done = subprocess.run(["docker", *command], check=True, capture_output=True, text=True)
        return done.stdout


def check_scenario(parser: argparse.ArgumentParser, scenario: Scenario, nodes: list[str]) -> None:
    """
    Reject what the demo cannot do: unknown nodes, partitions that never
    heal, and crashes of a node while it is cut off.
    """
    if unknown := scenario.nodes - set(nodes):
        parser.error(f"the scenario names nodes this design does not run: {sorted(unknown)}")
    for partition in scenario.partitions:
        if partition.heal_at is None:
            parser.error("every partition needs a heal_at: the demo must reach cut-off nodes")
        for crash in scenario.crashes:
            if crash.node in partition.nodes and partition.active(crash.at):
                parser.error(f"{crash.node} crashes while cut off; its state could not be saved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cluster", required=True, help="cluster config JSON")
    parser.add_argument("--design", help="override the config's design")
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arrival-rate", type=float, default=5.0, help="jobs per workload second")
    parser.add_argument("--execution-scale", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for jobs")
    parser.add_argument("--scenario", type=scenario_argument, help="scenario file of faults")
    parser.add_argument(
        "--compose", default="deploy/compose.yaml", help="Compose file of the cluster"
    )
    args = parser.parse_args()

    config = json.loads(Path(args.cluster).read_text())
    if args.design:
        config["design"] = args.design
    if not config.get("design"):
        parser.error("the config has no design; pass --design")
    workers = [worker["id"] for worker in config["workers"]]
    if config["design"] == "centralized":
        targets = [config.get("centralized", {}).get("coordinator", "coordinator")]
        nodes = workers + targets
    else:
        targets = nodes = workers
    # The config may publish nodes this design does not run (the coordinator).
    urls = {node_id: config["published"][node_id] for node_id in nodes}

    run = f"{int(time.time()) % 100_000:05d}"
    workload = [
        replace(s, job=replace(s.job, job_id=f"{run}-{s.job.job_id}"))
        for s in poisson_workload(args.seed, args.jobs, args.arrival_rate)
    ]
    job_ids = {submission.job.job_id for submission in workload}

    runner = None
    scenario = Scenario()
    if args.scenario is not None:
        # Random crashes hit workers only, as in the simulator experiments.
        try:
            scenario = args.scenario.resolve(args.seed, workers, horizon=workload[-1].at)
        except ValueError as error:
            parser.error(str(error))
        check_scenario(parser, scenario, nodes)
        affected = scenario.crashed | scenario.partitioned
        targets = [node_id for node_id in targets if node_id not in affected]
        if not targets:
            parser.error("the scenario crashes or cuts off every node jobs are submitted to")
        runner = ScenarioRunner(scenario, urls, args.compose, args.execution_scale, args.seed)
        for crash in scenario.crashes:
            restart = f", restart at {crash.restart_at:.1f}" if crash.restart_at else ""
            print(f"scenario: crash {crash.node} at {crash.at:.1f}{restart} (workload s)")
    down = runner.down if runner else set()

    def observations(client: httpx.Client) -> list[dict]:
        """
        What every node recorded about this run's jobs, including what
        crashed nodes recorded before they went down.
        """
        found = list(runner.saved_observations) if runner else []
        for node_id, url in urls.items():
            try:
                found += node_observations(client, url)
            except httpx.HTTPError:
                if node_id not in scenario.crashed | scenario.partitioned:
                    raise
        return [o for o in found if o["metric"].get("job_id") in job_ids]

    with httpx.Client(timeout=5) as client:
        wait_healthy(client, urls, timeout=60)
        print(f"{config['design']} cluster healthy: {', '.join(urls)}")
        if config["design"] in ("gossip", "periodic"):
            time.sleep(2 * config[config["design"]]["interval"])  # let views fill in

        before = {node_id: counters(client, url) for node_id, url in urls.items()}
        try:
            started = time.monotonic()
            if runner:
                try:
                    runner.start()
                except (httpx.HTTPError, subprocess.CalledProcessError, OSError) as error:
                    fail(f"starting the scenario: {error}")
            for index, submission in enumerate(workload):
                delay = started + submission.at * args.execution_scale - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                target = targets[index % len(targets)]
                response = client.post(f"{urls[target]}/jobs", json=to_data(submission.job))
                if response.status_code != 202:
                    fail(f"submitting {submission.job.job_id} to {target}: {response.text}")
            print(f"submitted {len(workload)} jobs (ids {run}-job-N) to {', '.join(targets)}")

            deadline = time.monotonic() + args.timeout
            while True:
                if runner and runner.errors:
                    fail("; ".join(runner.errors))
                seen = observations(client)
                completed = {o["metric"]["job_id"] for o in seen if o["type"] == "JobCompleted"}
                failed = [o for o in seen if o["type"] == "JobFailed"]
                if failed:
                    fail(f"jobs failed: {failed}")
                lost = (set(runner.lost_with) & job_ids) - completed if runner else set()
                if (runner is None or runner.finished) and completed | lost == job_ids:
                    break
                if time.monotonic() > deadline:
                    fail(
                        f"only {len(completed)} of {len(workload)} jobs completed, {len(lost)} lost"
                    )
                time.sleep(0.25)
            if runner:
                runner.join()
                if runner.errors:
                    fail("; ".join(runner.errors))
                wait_healthy(client, {n: u for n, u in urls.items() if n not in down}, 60)
                if scenario.partitions:
                    # A send in flight when a partition healed can take a full
                    # request timeout to fail; count it in this run.
                    time.sleep(SEND_TIMEOUT + 1)
                seen = observations(client)  # cut-off nodes are reachable again
        finally:
            if runner:
                runner.cleanup()

        lease_seconds = float(config.get("lease_seconds", 30.0))
        report(
            client,
            run,
            workload,
            seen,
            lost,
            runner,
            urls,
            workers,
            before,
            scenario,
            lease_seconds,
        )


def report(
    client, run, workload, seen, lost, runner, urls, workers, before, scenario, lease_seconds
) -> None:
    """
    Print per-job results, faults and traffic; fail if a GPU is still held
    or, without crashes or partitions, if a send failed.
    """
    down = runner.down if runner else set()
    accepted = {o["metric"]["job_id"]: o for o in seen if o["type"] == "JobAccepted"}
    finishes = {o["metric"]["job_id"]: o for o in seen if o["type"] == "JobCompleted"}
    forwards = Counter(o["metric"]["job_id"] for o in seen if o["type"] == "JobForwarded")
    runs = Counter(o["metric"]["job_id"] for o in seen if o["type"] == "JobStarted")
    # The start of the run that completed: the last start on the worker that
    # finished it.
    starts = {}
    for o in sorted(seen, key=lambda o: o["time"]):
        finish = finishes.get(o["metric"].get("job_id"))
        if (
            o["type"] == "JobStarted"
            and finish
            and o["metric"]["worker"] == finish["metric"]["worker"]
        ):
            starts[o["metric"]["job_id"]] = o
    print(
        f"\n{'job':8s} {'origin':14s} {'worker':14s} {'gpu':6s} {'delay':>7s} "
        f"{'run':>6s} {'fwd':>4s} {'runs':>5s}"
    )
    for submission in workload:
        job_id = submission.job.job_id
        short = job_id.removeprefix(f"{run}-")
        if job_id not in finishes:
            holder = runner.lost_with[job_id]
            print(f"{short:8s} {accepted[job_id]['node_id']:14s} lost with {holder}")
            continue
        start, finish = starts.get(job_id), finishes[job_id]
        if start is None:
            # Its StartReply was lost and its JobFinished arrived first.
            print(
                f"{short:8s} {accepted[job_id]['node_id']:14s} "
                f"{finish['metric']['worker']:14s} (start not observed)"
            )
            continue
        print(
            f"{short:8s} {accepted[job_id]['node_id']:14s} {start['metric']['worker']:14s} "
            f"{start['metric']['gpu_type']:6s} {start['time'] - accepted[job_id]['time']:7.3f} "
            f"{finish['time'] - start['time']:6.2f} {forwards[job_id]:4d} {runs[job_id]:5d}"
        )
    delays = [starts[j]["time"] - accepted[j]["time"] for j in starts]
    print(f"\nmean scheduling delay {mean(delays):.3f} s (wall clock, includes network)")
    if runner:
        interrupted = Counter(o["metric"]["reason"] for o in seen if o["type"] == "JobInterrupted")
        abandoned = Counter(o["metric"]["request"] for o in seen if o["type"] == "RequestAbandoned")
        lost_names = sorted(j.removeprefix(f"{run}-") for j in lost)
        print(f"interrupted runs: {sum(interrupted.values())} {dict(interrupted)}")
        print(f"abandoned requests: {sum(abandoned.values())} {dict(abandoned)}")
        print(f"jobs lost with crashed nodes: {len(lost)} {lost_names}")

    # A reservation whose reply was lost holds its GPUs until its lease
    # expires, so with faults wait out a lease before checking.
    deadline = time.monotonic() + (lease_seconds + 5 if runner else 0)
    for worker in workers:
        if worker in down:
            continue
        while True:
            status = client.get(f"{urls[worker]}/status").json()["status"]
            capacity = [group["count"] for group in status["spec"]["gpus"]]
            if status["free_gpus"] == capacity:
                break
            if time.monotonic() > deadline:
                fail(f"{worker} still holds GPUs: free {status['free_gpus']} of {capacity}")
            time.sleep(0.5)
    print(
        "every GPU released" + (f" (down, not checked: {', '.join(sorted(down))})" if down else "")
    )

    # A crashed node's counters restart from zero, so its traffic is what it
    # sent before each crash plus what it sent since its last restart.
    totals = {key: Counter() for key in COUNTERS}
    for node_id, url in urls.items():
        current = counters(client, url) if node_id not in down else {}
        saved = runner.saved_counters.get(node_id, {}) if runner else {}
        for key in COUNTERS:
            total = current.get(key, Counter()) + saved.get(key, Counter())
            totals[key] += total - before[node_id][key]
    sent, size, failures, dropped = (totals[key] for key in COUNTERS)
    print(
        f"traffic during the demo: {sum(sent.values())} messages, {sum(size.values()) / 1000:.1f} KB"
    )
    for kind, count in sent.most_common():
        print(f"  {kind:15s} {count:6d}  {size[kind] / 1000:8.1f} KB")
    if scenario.losses:
        print(f"dropped by injected losses: {sum(dropped.values())} {dict(dropped)}")
    if scenario.crashes or scenario.partitions:
        print(
            f"failed sends (expected with crashes and partitions): {sum(failures.values())} {dict(failures)}"
        )
    elif failures:
        fail(f"messages failed to send: {dict(failures)}")
    else:
        print("no failed messages")
    print("OK")


if __name__ == "__main__":
    main()
