"""
A deterministic discrete-event runtime for experiments and tests.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import count

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
from torchsched.types.metrics import Metric
from torchsched.types.models import Job, NodeId


@dataclass(frozen=True, slots=True)
class Observation:
    time: float
    node_id: NodeId
    metric: Metric


@dataclass
class SimResult:
    end_time: float
    observations: list[Observation] = field(default_factory=list)
    messages: Counter[str] = field(default_factory=Counter)  # network messages by type
    # TODO: count message bytes too; needs an encoder for messages.
    undeliverable: int = 0  # messages addressed to unknown nodes


class Simulator:
    """
    Runs nodes in simulated time.

    We process events in time order, breaking ties by the
    order events were scheduled in. Thus, runs are deterministic.

    Every network message takes `latency` seconds. A message a node sends to
    itself arrives at the same time and is not counted as network traffic.
    An `Execute` finishes after the job's runtime on the reserved GPU type.
    """

    def __init__(self, nodes: Iterable[Node], *, latency: float = 0.0):
        if latency < 0:
            raise ValueError("latency must be nonnegative")
        self.nodes: dict[NodeId, Node] = {}
        for node in nodes:
            if node.node_id in self.nodes:
                raise ValueError(f"duplicate node {node.node_id!r}")
            self.nodes[node.node_id] = node
        self.latency = latency
        self.now = 0.0
        self.result = SimResult(end_time=0.0)
        self._pq: list[tuple[float, int, NodeId, Event, int | None]] = []
        self._seq = count()
        self._timers: dict[tuple[NodeId, TimerKey], int] = {}
        for node_id in self.nodes:
            self._schedule(0.0, node_id, Boot())

    def submit(self, job: Job, *, at: float, node_id: NodeId) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"unknown node {node_id!r}")
        if at < self.now:
            raise ValueError("cannot submit a job in the past")
        self._schedule(at, node_id, JobSubmitted(job))

    def run(self, until: float | None = None) -> SimResult:
        """
        Process events until none remain or the next one is after `until`.
        """
        while self._pq and (until is None or self._pq[0][0] <= until):
            time, _, node_id, event, timer_token = heapq.heappop(self._pq)
            if isinstance(event, TimerFired):
                key = (node_id, event.key)
                if self._timers.get(key) != timer_token:
                    continue  # cancelled or replaced
                del self._timers[key]
            self.now = time
            self._apply(node_id, self.nodes[node_id].handle(event, time))
        self.result.end_time = self.now
        return self.result

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
                        self.result.messages[type(msg).__name__] += 1
                        self._schedule(
                            self.now + self.latency, to, Received(node_id, msg)
                        )
                case SetTimer(key, delay):
                    token = next(self._seq)
                    self._timers[(node_id, key)] = token
                    self._schedule(self.now + delay, node_id, TimerFired(key), token)
                case CancelTimer(key):
                    self._timers.pop((node_id, key), None)
                case Execute(reservation_id, job, _, gpu_type):
                    finish = self.now + job.runtime_by_gpu[gpu_type]
                    self._schedule(
                        finish, node_id, ExecutionFinished(reservation_id, True)
                    )
                case Observe(metric):
                    self.result.observations.append(
                        Observation(self.now, node_id, metric)
                    )
