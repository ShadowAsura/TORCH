"""
The worker component: owns a node's GPU groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from torchsched.node import NodeContext
from torchsched.retransmit import Retransmitter
from torchsched.types.actions import Action, CancelTimer, Execute, Send, SetTimer
from torchsched.types.events import Event, ExecutionFinished, Received, TimerFired
from torchsched.types.messages import (
    FinishedAck,
    Heartbeat,
    JobFinished,
    ReserveReply,
    ReserveRequest,
    StartReply,
    StartRequest,
    StatusChanged,
)
from torchsched.types.models import GroupIndex, Job, NodeId, ReservationId, Status


@dataclass
class _Reservation:
    job: Job
    group: GroupIndex
    holder: NodeId  # the scheduler that reserved it; only it may start it
    request_id: str
    granted_key: tuple[int, int]
    running: bool = False


class Worker:
    """
    Grants reservations atomically, runs started jobs, and releases GPUs.

    A reservation that is not started within `lease_seconds` expires, so a
    scheduler that disappears cannot hold GPUs forever.

    `JobFinished` is resent every `request_timeout` seconds until the
    scheduler acknowledges it, at most `max_sends` times in all.

    While jobs run, every `heartbeat_interval` seconds each scheduler with
    jobs here gets a `Heartbeat` listing them, so it can tell a long job
    from a dead worker.

    Every change to free GPUs increments `version` and sends `StatusChanged`
    to this node's dissemination component.
    """

    name: ClassVar[str] = "worker"
    accepts: ClassVar[frozenset[type]] = frozenset(
        {ReserveRequest, StartRequest, ExecutionFinished, FinishedAck}
    )

    def __init__(
        self,
        ctx: NodeContext,
        *,
        lease_seconds: float = 30.0,
        request_timeout: float = 1.0,
        max_sends: int = 5,
        heartbeat_interval: float = 1.0,
    ):
        if not ctx.spec.gpus:
            raise ValueError("a worker needs at least one GPU group")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        self.ctx = ctx
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_armed = False
        self.version = 0
        self._free = [group.count for group in ctx.spec.gpus]
        self._reservations: dict[ReservationId, _Reservation] = {}
        # Granted requests by (scheduler, request_id), so a retried request
        # gets the same grant instead of a second reservation.
        self._granted: dict[tuple[NodeId, str], ReservationId] = {}
        self._outbox = Retransmitter(timeout=request_timeout, max_sends=max_sends)
        self._next_id = 0

    def status(self, now: float) -> Status:
        return Status(
            self.ctx.spec, tuple(self._free), self.ctx.incarnation, self.version, now
        )

    def handle(self, event: Event, now: float) -> list[Action]:
        match event:
            case Received(sender, ReserveRequest() as request):
                return self._reserve(sender, request, now)
            case Received(sender, StartRequest() as request):
                return self._start(sender, request)
            case ExecutionFinished():
                return self._finished(event, now)
            case TimerFired(("lease", reservation_id)):
                return self._expire(reservation_id, now)
            case Received(_, FinishedAck(reservation_id)):
                return self._outbox.acknowledged(("finished", reservation_id))
            case TimerFired(("heartbeat",)):
                return self._heartbeat()
            case TimerFired(("resend", *key)):
                # A scheduler that never acknowledges is gone; stop resending.
                actions, _ = self._outbox.timed_out(tuple(key))
                return actions
        return []

    def _reserve(
        self, sender: NodeId, request: ReserveRequest, now: float
    ) -> list[Action]:
        job = request.job
        # to make sure that requests are idempotent
        existing = self._granted.get((sender, request.request_id))
        if existing is not None:
            held = self._reservations[existing]
            return [Send(sender, self._grant_reply(request, existing, held))]

        group, reason = self._choose_group(job, request.group, now)
        if group is None:
            reply = ReserveReply(
                request.request_id,
                job.job_id,
                False,
                reason=reason,
                status=self.status(now),
            )
            return [Send(sender, reply)]

        self._free[group] -= job.gpu_count
        self.version += 1
        reservation_id = f"{self.ctx.node_id}:{self.ctx.incarnation}:{self._next_id}"
        self._next_id += 1
        held = _Reservation(
            job, group, sender, request.request_id, (self.ctx.incarnation, self.version)
        )
        self._reservations[reservation_id] = held
        self._granted[(sender, request.request_id)] = reservation_id
        return [
            Send(sender, self._grant_reply(request, reservation_id, held)),
            SetTimer(("lease", reservation_id), self.lease_seconds),
            self._changed(now),
        ]

    def _choose_group(
        self, job: Job, requested: GroupIndex | None, now: float
    ) -> tuple[GroupIndex | None, str | None]:
        if requested is None:
            fitting = self.status(now).fitting_groups(job)
            if fitting:
                return fitting[0], None
            return None, "no GPU group has room for the job"
        groups = self.ctx.spec.gpus
        if not 0 <= requested < len(groups) or not job.can_use(groups[requested]):
            return None, "requested GPU group cannot run the job"
        if self._free[requested] < job.gpu_count:
            return None, "requested GPU group lacks free GPUs"
        return requested, None

    def _start(self, sender: NodeId, request: StartRequest) -> list[Action]:
        reservation_id = request.reservation_id
        held = self._reservations.get(reservation_id)
        if held is None or held.holder != sender:
            reply = StartReply(reservation_id, False, "unknown or expired reservation")
            return [Send(sender, reply)]
        # check to make sure haven't already seen this msg
        if held.running:
            return [Send(sender, StartReply(reservation_id, True))]
        held.running = True
        gpu_type = self.ctx.spec.gpus[held.group].gpu_type
        actions: list[Action] = [
            CancelTimer(("lease", reservation_id)),
            Execute(reservation_id, held.job, held.group, gpu_type),
            Send(sender, StartReply(reservation_id, True)),
        ]
        if not self._heartbeat_armed:
            self._heartbeat_armed = True
            actions.append(SetTimer(("heartbeat",), self.heartbeat_interval))
        return actions

    def _heartbeat(self) -> list[Action]:
        running: dict[NodeId, list[ReservationId]] = {}
        for reservation_id, held in self._reservations.items():
            if held.running:
                running.setdefault(held.holder, []).append(reservation_id)
        if not running:
            self._heartbeat_armed = False
            return []
        actions: list[Action] = [
            Send(holder, Heartbeat(tuple(ids))) for holder, ids in running.items()
        ]
        actions.append(SetTimer(("heartbeat",), self.heartbeat_interval))
        return actions

    def _finished(self, event: ExecutionFinished, now: float) -> list[Action]:
        held = self._reservations.get(event.reservation_id)
        if held is None or not held.running:
            return []
        self._release(event.reservation_id)
        notice = JobFinished(
            held.job.job_id,
            event.reservation_id,
            event.succeeded,
            (self.ctx.incarnation, self.version),
            event.reason,
        )
        key = ("finished", event.reservation_id)
        return [*self._outbox.send(key, held.holder, notice), self._changed(now)]

    def _expire(self, reservation_id: ReservationId, now: float) -> list[Action]:
        held = self._reservations.get(reservation_id)
        if held is None or held.running:
            return []
        self._release(reservation_id)
        return [self._changed(now)]

    def _release(self, reservation_id: ReservationId) -> None:
        held = self._reservations.pop(reservation_id)
        self._granted.pop((held.holder, held.request_id), None)
        self._free[held.group] += held.job.gpu_count
        self.version += 1

    def _grant_reply(
        self, request: ReserveRequest, reservation_id: ReservationId, held: _Reservation
    ) -> ReserveReply:
        return ReserveReply(
            request.request_id,
            held.job.job_id,
            True,
            reservation_id,
            held.group,
            order_key=held.granted_key,
        )

    def _changed(self, now: float) -> Send:
        return Send(self.ctx.node_id, StatusChanged(self.status(now)))
