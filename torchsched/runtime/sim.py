"""
A deterministic discrete-event runtime for experiments and tests.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import count
from random import Random

from torchsched.codec import encode
from torchsched.node import Node
from torchsched.types.actions import (
    Action,
    CancelTimer,
    Execute,
    Observe,
    Send,
    SetTimer,
    TimerKey,
)
from torchsched.types.events import (
    Boot,
    Event,
    ExecutionFinished,
    JobSubmitted,
    Received,
    TimerFired,
)
from torchsched.types.metrics import JobCompleted, JobFailed, Metric
from torchsched.types.models import GroupIndex, Job, NodeId


@dataclass(frozen=True, slots=True)
class Observation:
    time: float
    node_id: NodeId
    metric: Metric


@dataclass(frozen=True, slots=True)
class Execution:
    """
    When a job actually ran, as seen by the worker that ran it.
    """

    node_id: NodeId
    job_id: str
    group: GroupIndex
    gpu_count: int
    start: float
    end: float
    interrupted: bool = False  # its worker crashed before it finished


@dataclass(frozen=True)
class MessageLoss:
    """
    Drop network messages sent during `[start, end)`.

    Each matching message is dropped with `probability`. Leaving `kinds`
    (message type names, such as "StatusPush"), `senders` or `receivers`
    as None matches every value. Messages a node sends to itself are never
    dropped.
    """

    start: float = 0.0
    end: float = math.inf
    probability: float = 1.0
    kinds: frozenset[str] | None = None
    senders: frozenset[NodeId] | None = None
    receivers: frozenset[NodeId] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be between 0 and 1")
        for name in ("kinds", "senders", "receivers"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, str):
                    raise TypeError(f"{name} must be a collection, not a string")
                object.__setattr__(self, name, frozenset(value))

    def matches(self, time: float, sender: NodeId, receiver: NodeId, kind: str) -> bool:
        return (
            self.start <= time < self.end
            and (self.kinds is None or kind in self.kinds)
            and (self.senders is None or sender in self.senders)
            and (self.receivers is None or receiver in self.receivers)
        )


@dataclass
class SimResult:
    end_time: float
    observations: list[Observation] = field(default_factory=list)
    executions: list[Execution] = field(default_factory=list)  # in finish order
    messages: Counter[str] = field(default_factory=Counter)  # network messages by type
    # encoded size of network messages by type, as `torchsched.codec` (and
    # so the HTTP runtime) would send them.
    message_bytes: Counter[str] = field(default_factory=Counter)
    undeliverable: int = 0  # messages addressed to unknown nodes
    # network messages dropped by `MessageLoss` or because the receiver was
    # down, by type (also in `messages`)
    dropped: Counter[str] = field(default_factory=Counter)
    # Jobs lost with a crashed scheduler (or submitted to a node that was
    # down), unless another scheduler later completed a copy of them.
    # TODO: add a retry policy, for example client resubmission.
    lost_jobs: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _Crash:
    restart_at: float | None


@dataclass(frozen=True)
class _Restart:
    pass


class Simulator:
    """
    Runs nodes in simulated time.

    We process events in time order, breaking ties by the
    order events were scheduled in. Thus, runs are deterministic.

    Every network message takes `latency` seconds. A message a node sends to
    itself arrives at the same time and is not counted as network traffic.
    An `Execute` finishes after the job's runtime on the reserved GPU type.

    `losses` drop network messages; random drops use a stream seeded by
    `seed`, so runs with losses are deterministic too.

    `crash` takes a whole node down, as when its machine fails: it handles
    nothing, messages to it are dropped, and its timers and running jobs
    are gone. A restarted node is rebuilt (`Node.rebuild`) with empty
    state and the next incarnation.
    """

    def __init__(
        self,
        nodes: Iterable[Node],
        *,
        latency: float = 0.0,
        losses: Iterable[MessageLoss] = (),
        seed: int = 0,
    ):
        if latency < 0:
            raise ValueError("latency must be nonnegative")
        self.nodes: dict[NodeId, Node] = {}
        for node in nodes:
            if node.node_id in self.nodes:
                raise ValueError(f"duplicate node {node.node_id!r}")
            self.nodes[node.node_id] = node
        self.latency = latency
        self.losses = tuple(losses)
        self._loss_rng = Random(f"loss:{seed}")
        self.now = 0.0
        self.result = SimResult(end_time=0.0)
        self._pq: list[tuple[float, int, NodeId, Event, int | None]] = []
        self._seq = count()
        self._timers: dict[tuple[NodeId, TimerKey], int] = {}
        self._unfinished: set[str] = set()  # submitted jobs not yet completed or failed
        self._running: dict[tuple[NodeId, str], tuple[Job, GroupIndex, float]] = {}
        self._down: set[NodeId] = set()
        for node_id in self.nodes:
            self._schedule(0.0, node_id, Boot())

    def submit(self, job: Job, *, at: float, node_id: NodeId) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"unknown node {node_id!r}")
        if at < self.now:
            raise ValueError("cannot submit a job in the past")
        self._unfinished.add(job.job_id)
        self._schedule(at, node_id, JobSubmitted(job))

    def crash(self, node_id: NodeId, *, at: float, restart_at: float | None = None) -> None:
        """
        Crash `node_id` at time `at`, and restart it at `restart_at` if given.
        """
        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError(f"unknown node {node_id!r}")
        if at < self.now:
            raise ValueError("cannot crash a node in the past")
        if restart_at is not None:
            if restart_at <= at:
                raise ValueError("restart_at must be after at")
            if node.rebuild is None:
                raise ValueError(f"node {node_id!r} cannot be rebuilt")
        self._schedule(at, node_id, _Crash(restart_at))

    def run(
        self, until: float | None = None, *, until_jobs_done: bool = False
    ) -> SimResult:
        """
        Process events until none remain or the next one is after `until`.

        Periodic protocols (gossip) never run out of events, so with
        `until_jobs_done` the run also stops as soon as every submitted job
        has completed or failed.
        """
        while self._pq and (until is None or self._pq[0][0] <= until):
            if until_jobs_done and not self._unfinished:
                break
            time, _, node_id, event, timer_token = heapq.heappop(self._pq)
            if isinstance(event, TimerFired):
                key = (node_id, event.key)
                if self._timers.get(key) != timer_token:
                    continue  # cancelled or replaced
                del self._timers[key]
            self.now = time
            if isinstance(event, _Crash | _Restart):
                self._crash_or_restart(node_id, event)
                continue
            if node_id in self._down:
                self._lost_while_down(event)
                continue
            if isinstance(event, ExecutionFinished):
                if (node_id, event.reservation_id) not in self._running:
                    continue  # a run that a crash already ended
                self._record_execution(node_id, event.reservation_id)
            self._apply(node_id, self.nodes[node_id].handle(event, time))
        self.result.end_time = self.now
        return self.result

    def _crash_or_restart(self, node_id: NodeId, event: _Crash | _Restart) -> None:
        node = self.nodes[node_id]
        if isinstance(event, _Restart):
            assert node.rebuild is not None
            self.nodes[node_id] = node.rebuild(node.ctx.incarnation + 1)
            self._down.discard(node_id)
            self._schedule(self.now, node_id, Boot())
            return
        if node_id in self._down:
            return
        self._down.add(node_id)
        for key in [key for key in self._timers if key[0] == node_id]:
            del self._timers[key]
        for key in [key for key in self._running if key[0] == node_id]:
            self._record_execution(*key, interrupted=True)
        if node.scheduler is not None:
            for job_id in node.scheduler.owned_jobs():
                if job_id in self._unfinished:
                    self._unfinished.discard(job_id)
                    self.result.lost_jobs.add(job_id)
        if event.restart_at is not None:
            self._schedule(event.restart_at, node_id, _Restart())

    def _lost_while_down(self, event: Event) -> None:
        if isinstance(event, Received):
            self.result.dropped[type(event.msg).__name__] += 1
        elif isinstance(event, JobSubmitted) and event.job.job_id in self._unfinished:
            self._unfinished.discard(event.job.job_id)
            self.result.lost_jobs.add(event.job.job_id)

    def _schedule(
        self, time: float, node_id: NodeId, event: Event, timer_token: int | None = None
    ) -> None:
        heapq.heappush(self._pq, (time, next(self._seq), node_id, event, timer_token))

    def _apply(self, node_id: NodeId, actions: list[Action]) -> None:
        for action in actions:
            match action:
                case Send(to, msg):
                    if to not in self.nodes:
                        self.result.undeliverable += 1
                    elif to == node_id:
                        self._schedule(self.now, to, Received(node_id, msg))
                    else:
                        kind = type(msg).__name__
                        self.result.messages[kind] += 1
                        self.result.message_bytes[kind] += len(encode(msg))
                        if self._lost(node_id, to, kind):
                            self.result.dropped[kind] += 1
                            continue
                        self._schedule(
                            self.now + self.latency, to, Received(node_id, msg)
                        )
                case SetTimer(key, delay):
                    token = next(self._seq)
                    self._timers[(node_id, key)] = token
                    self._schedule(self.now + delay, node_id, TimerFired(key), token)
                case CancelTimer(key):
                    self._timers.pop((node_id, key), None)
                case Execute(reservation_id, job, group, gpu_type):
                    self._running[(node_id, reservation_id)] = (job, group, self.now)
                    finish = self.now + job.runtime_by_gpu[gpu_type]
                    self._schedule(
                        finish, node_id, ExecutionFinished(reservation_id, True)
                    )
                case Observe(metric):
                    if isinstance(metric, (JobCompleted, JobFailed)):
                        self._unfinished.discard(metric.job_id)
                        self.result.lost_jobs.discard(metric.job_id)
                    self.result.observations.append(
                        Observation(self.now, node_id, metric)
                    )

    def _lost(self, sender: NodeId, receiver: NodeId, kind: str) -> bool:
        for loss in self.losses:
            if loss.matches(self.now, sender, receiver, kind) and (
                loss.probability >= 1 or self._loss_rng.random() < loss.probability
            ):
                return True
        return False

    def _record_execution(
        self, node_id: NodeId, reservation_id: str, *, interrupted: bool = False
    ) -> None:
        job, group, start = self._running.pop((node_id, reservation_id))
        self.result.executions.append(
            Execution(
                node_id, job.job_id, group, job.gpu_count, start, self.now, interrupted
            )
        )
