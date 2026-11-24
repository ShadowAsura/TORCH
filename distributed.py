"""Discrete-event model of decentralized scheduling on heterogeneous GPU workers."""

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from random import Random

import simpy

from simulator import Job, Node, NodeResources, Placement, Reservation, Simulation


POLICIES = ("gossip", "periodic", "sampling", "global-gossip")


@dataclass(frozen=True)
class Settings:
    policy: str = "gossip"
    seed: int = 42
    radius: int = 1
    sample_size: int = 2
    update_interval: float = 0.5
    max_staleness: float = 3
    latency: float = 0.02
    retry_interval: float = 0.25
    max_rounds: int = 1000

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"unknown distributed policy: {self.policy}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("radius", "sample_size", "max_rounds"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("update_interval", "retry_interval", "max_staleness", "latency"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.update_interval == 0 or self.retry_interval == 0:
            raise ValueError("update_interval and retry_interval must be positive")


@dataclass(frozen=True)
class Faults:
    worker: str | None = None
    crash_at: float = 0
    recover_at: float | None = None
    drop_status_until: float = 0

    def __post_init__(self):
        for name in ("crash_at", "recover_at", "drop_status_until"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.worker is None and (self.crash_at or self.recover_at is not None):
            raise ValueError("crash and recovery require a worker")
        if self.recover_at is not None and self.recover_at <= self.crash_at:
            raise ValueError("recover_at must follow crash_at")


@dataclass(frozen=True)
class Status:
    index: int
    node: Node
    free_gpus: int
    version: int
    observed_at: float

    def fits(self, job: Job) -> bool:
        return (
            0 < job.gpu_count <= self.free_gpus
            and job.min_vram_gb <= self.node.vram_gb
            and self.node.gpu_type in job.runtime_by_gpu
        )


class Worker:
    def __init__(self, env: simpy.Environment, index: int, node: Node):
        self.env = env
        self.index = index
        self.resources = NodeResources(node)
        self.version = 0
        self.down = env.event()

    def status(self) -> Status | None:
        if self.down.triggered:
            return None
        return Status(
            self.index, self.resources.node, self.resources.free_gpus,
            self.version, self.env.now,
        )

    def reserve(self, job: Job) -> Reservation | None:
        if self.down.triggered:
            return None
        reservation = self.resources.try_reserve(job)
        if reservation is not None:
            self.version += 1
        return reservation

    def release(self, reservation: Reservation) -> None:
        self.resources.release(reservation)
        self.version += 1


@dataclass
class Scheduler:
    index: int
    peers: tuple[int, ...]
    random: Random
    view: dict[int, Status] = field(default_factory=dict)

    def receive(self, statuses: tuple[Status, ...], now: float, max_age: float):
        for status in statuses:
            if status.index not in self.peers or now - status.observed_at > max_age:
                continue
            old = self.view.get(status.index)
            if old is None or (status.version, status.observed_at) > (old.version, old.observed_at):
                self.view[status.index] = status


def neighborhoods(count: int, radius: int, seed: int) -> list[tuple[int, ...]]:
    """Build a ring with seeded, bidirectional shortcuts."""
    offsets = range(1, min(radius, count - 1) + 1)
    peers = [
        {(index + sign * offset) % count for offset in offsets for sign in (-1, 1)}
        for index in range(count)
    ]
    shuffled = list(range(count))
    Random(seed).shuffle(shuffled)
    for left, right in zip(shuffled[::2], shuffled[1::2]):
        peers[left].add(right)
        peers[right].add(left)
    return [tuple(sorted(neighbors)) for neighbors in peers]


@dataclass
class Traffic:
    messages: Counter = field(default_factory=Counter)
    status_records: int = 0
    dropped_status_messages: int = 0
    reservation_attempts: int = 0
    failed_reservations: int = 0
    reservation_latency_total: float = 0
    retries: int = 0
    worker_crashes: int = 0
    worker_recoveries: int = 0
    interrupted_jobs: int = 0
    jobs_exhausted: int = 0

    def summary(self) -> dict:
        return {
            "messages": sum(self.messages.values()),
            "messages_by_kind": dict(sorted(self.messages.items())),
            "status_records": self.status_records,
            "dropped_status_messages": self.dropped_status_messages,
            "reservation_attempts": self.reservation_attempts,
            "failed_reservations": self.failed_reservations,
            "mean_reservation_latency": (
                self.reservation_latency_total / self.reservation_attempts
                if self.reservation_attempts else 0
            ),
            "retries": self.retries,
            "worker_crashes": self.worker_crashes,
            "worker_recoveries": self.worker_recoveries,
            "interrupted_jobs": self.interrupted_jobs,
            "jobs_exhausted": self.jobs_exhausted,
        }


def simulate_distributed(
    nodes: list[Node], jobs: list[Job], settings: Settings, faults: Faults = Faults()
) -> tuple[Simulation, Traffic]:
    if faults.worker is not None and not any(node.name == faults.worker for node in nodes):
        raise ValueError(f"unknown worker: {faults.worker}")
    env = simpy.Environment()
    result = Simulation(elapsed=0)
    traffic = Traffic()
    if not nodes:
        result.rejected.extend(jobs)
        result.elapsed = max((job.arrival for job in jobs), default=0)
        return result, traffic

    workers = [Worker(env, index, node) for index, node in enumerate(nodes)]
    peers = neighborhoods(len(nodes), settings.radius, settings.seed)
    if settings.policy == "global-gossip":
        peers = [
            tuple(peer for peer in range(len(nodes)) if peer != index)
            for index in range(len(nodes))
        ]
    schedulers = [
        Scheduler(index, neighbors, Random(settings.seed + index))
        for index, neighbors in enumerate(peers)
    ]

    def transmit_status(target: Scheduler, payload: tuple[Status, ...]):
        traffic.messages["status_update"] += 1
        traffic.status_records += len(payload)
        yield env.timeout(settings.latency)
        if env.now < faults.drop_status_until:
            traffic.dropped_status_messages += 1
        else:
            target.receive(payload, env.now, settings.max_staleness)

    def disseminate(scheduler: Scheduler):
        while True:
            own = workers[scheduler.index].status()
            if own is not None and scheduler.peers:
                if settings.policy == "periodic":
                    targets = scheduler.peers
                    payload = (own,)
                else:
                    targets = (scheduler.random.choice(scheduler.peers),)
                    payload = (own,) + tuple(
                        status for status in scheduler.view.values()
                        if env.now - status.observed_at <= settings.max_staleness
                    )
                for target in targets:
                    env.process(transmit_status(schedulers[target], payload))
            yield env.timeout(settings.update_interval)

    def probe(index: int):
        traffic.messages["probe_request"] += 1
        yield env.timeout(settings.latency)
        status = workers[index].status()
        traffic.messages["probe_reply"] += 1
        traffic.status_records += int(status is not None)
        yield env.timeout(settings.latency)
        if status is not None and env.now < faults.drop_status_until:
            traffic.dropped_status_messages += 1
            return None
        return status

    def discover(scheduler: Scheduler):
        own = workers[scheduler.index].status()
        statuses = [own] if own is not None else []
        if settings.policy == "sampling":
            count = min(settings.sample_size - 1, len(scheduler.peers))
            selected = scheduler.random.sample(scheduler.peers, count)
            queries = [env.process(probe(index)) for index in selected]
            replies = yield env.all_of(queries)
            statuses.extend(status for status in replies.values() if status is not None)
        else:
            statuses.extend(
                status for status in scheduler.view.values()
                if env.now - status.observed_at <= settings.max_staleness
            )
        return sorted(statuses, key=lambda status: status.index)

    def reserve(scheduler: Scheduler, index: int, job: Job):
        traffic.reservation_attempts += 1
        started = env.now
        remote = scheduler.index != index
        worker = workers[index]
        if remote:
            traffic.messages["reserve_request"] += 1
            yield env.timeout(settings.latency)
        reservation = worker.reserve(job)
        down = worker.down
        if remote:
            traffic.messages["reserve_reply"] += 1
            yield env.timeout(settings.latency)
        if reservation is not None and down.triggered:
            worker.release(reservation)
            reservation = None
        traffic.reservation_latency_total += env.now - started
        if reservation is None:
            traffic.failed_reservations += 1
            return None
        return reservation, down

    def schedule(job: Job, origin: int):
        yield env.timeout(job.arrival)
        # Static feasibility checks live in the simulation harness. Schedulers
        # discover available capacity locally and never consult this inventory.
        if not any(worker.resources.supports(job) for worker in workers):
            result.rejected.append(job)
            return
        index = origin
        for attempt in range(settings.max_rounds):
            interrupted = False
            for hop in range(len(schedulers)):
                index = (origin + hop) % len(schedulers)
                scheduler = schedulers[index]
                statuses = yield env.process(discover(scheduler))
                for status in statuses:
                    if not status.fits(job):
                        continue
                    grant = yield env.process(reserve(scheduler, status.index, job))
                    if grant is None:
                        continue
                    token, down = grant
                    worker = workers[status.index]
                    start = env.now
                    finish = env.timeout(job.runtime_by_gpu[status.node.gpu_type])
                    events = yield finish | down
                    worker.release(token)
                    if finish in events:
                        result.placements.append(Placement(job, status.node, start, env.now))
                        return
                    traffic.interrupted_jobs += 1
                    result.interrupted_gpu_seconds += job.gpu_count * (env.now - start)
                    interrupted = True
                    break
                if interrupted:
                    break
                if hop + 1 < len(schedulers):
                    traffic.messages["forward_job"] += 1
                    yield env.timeout(settings.latency)
            if attempt + 1 < settings.max_rounds:
                traffic.retries += 1
                origin = (index + 1) % len(schedulers)
                traffic.messages["forward_job"] += 1
                yield env.timeout(settings.retry_interval + settings.latency)
        traffic.jobs_exhausted += 1
        result.rejected.append(job)

    def crash(worker: Worker):
        yield env.timeout(faults.crash_at)
        worker.version += 1
        worker.down.succeed()
        traffic.worker_crashes += 1
        if faults.recover_at is not None:
            yield env.timeout(faults.recover_at - faults.crash_at)
            worker.version += 1
            worker.down = env.event()
            traffic.worker_recoveries += 1

    if jobs:
        for worker in workers:
            if worker.resources.node.name == faults.worker:
                env.process(crash(worker))
        if settings.policy != "sampling":
            for scheduler in schedulers:
                env.process(disseminate(scheduler))
        processes = [env.process(schedule(job, index % len(nodes))) for index, job in enumerate(jobs)]
        env.run(until=env.all_of(processes))
    result.elapsed = env.now
    return result, traffic
