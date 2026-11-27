"""
The scheduler component: owns submitted jobs and places them on workers.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar

from torchsched.interfaces.policies import ForwardingPolicy, PlacementPolicy
from torchsched.node import NodeContext
from torchsched.retransmit import Key, Retransmitter
from torchsched.types.actions import Action, Observe, Send, SetTimer
from torchsched.types.events import Event, JobSubmitted, Received, TimerFired
from torchsched.types.messages import (
    CapacityAvailable,
    FinishedAck,
    ForwardAck,
    ForwardJob,
    JobFinished,
    LookupReply,
    LookupRequest,
    ReserveReply,
    ReserveRequest,
    StartReply,
    StartRequest,
    StatusPush,
)
from torchsched.types.metrics import (
    JobAccepted,
    JobCompleted,
    JobFailed,
    JobForwarded,
    JobStarted,
    PlacementRetried,
    RequestAbandoned,
    ReservationRejected,
)
from torchsched.types.models import (
    Assignment,
    GpuType,
    GroupIndex,
    Job,
    JobId,
    NodeId,
    ReservationId,
    Status,
)


class _Phase(Enum):
    PENDING = "pending"  # waiting to be placed
    RESERVING = "reserving"  # ReserveRequest in flight
    STARTING = "starting"  # reserved; StartRequest in flight
    RUNNING = "running"
    FORWARDING = "forwarding"  # ForwardJob sent; waiting for its ForwardAck


@dataclass
class _Job:
    job: Job
    seq: int  # submission order, for FIFO policies
    phase: _Phase = _Phase.PENDING
    worker: NodeId | None = None
    group: GroupIndex | None = None
    gpu_type: GpuType | None = None
    reservation_id: ReservationId | None = None
    request_id: str | None = None  # the ReserveRequest in flight, if RESERVING
    handoff_id: str | None = None  # the ForwardJob in flight, if FORWARDING
    # Schedulers that tried this job on its current lap, ending with this one.
    visited: tuple[NodeId, ...] = ()


@dataclass(frozen=True)
class _Held:
    """
    A reservation a worker granted this scheduler.

    Kept until its `JobFinished` arrives, even if the scheduler gave up on
    it (for example, no `StartReply` came back), because the job may be
    running there anyway.
    """

    job_id: JobId
    worker: NodeId
    group: GroupIndex
    gpu_count: int


@dataclass(frozen=True)
class _Assumed:
    """
    A change this scheduler caused that a status record may not show yet.

    It applies to records of `node_id` older than `order_key`, and is
    dropped once a record at or past `order_key` arrives.

    It's safe to use these assumptions because `order_key` is the version the worker
    reported right AFTER making the change (in the grant reply or `JobFinished`),
    not a version the scheduler saw BEFORE asking. Thus, we can safely say that:
    * if a view has order_key < our order_key, this assumption applies
    * if a view has order_key >= our order key, the view already reflects
        the assumption and thus we can forget our assumption
    """

    node_id: NodeId
    group: GroupIndex
    delta: int  # negative for a reservation, positive for a release
    order_key: tuple[int, int]  # (incarnation, version)


class Scheduler:
    """
    Places queued jobs in rounds: look up candidates, plan, reserve, start.

    One round runs at a time. A new round starts when something may have
    changed: a job arrives or finishes, the view gains capacity, a start
    fails, or (if `retry_interval` is set) a retry timer fires while jobs
    are still waiting.

    With a `forwarding` policy, a job that nothing in this node's view has
    room for is handed to another scheduler, which then owns it. When the
    policy has no next hop (the lap is over), the job waits here and starts
    a new lap when the retry timer fires.

    Views can lag behind the scheduler's own actions. So, we store
    the reservations and releases we initiated, and apply those as assumptions
    on top of views that predate them

    Messages to other nodes can be lost. Reserve requests, start requests
    and forwarded jobs are resent every `request_timeout` seconds until
    answered, at most `max_sends` times in all; then the scheduler gives up
    (`RequestAbandoned`) and queues the job again. Giving up can be wrong
    when only the replies were lost, so a job may run more than once. The
    first run to finish counts, and later ones are ignored.
    """

    name: ClassVar[str] = "scheduler"
    accepts: ClassVar[frozenset[type]] = frozenset(
        {
            JobSubmitted,
            ForwardJob,
            CapacityAvailable,
            LookupReply,
            ReserveReply,
            StartReply,
            JobFinished,
            ForwardAck,
        }
    )

    def __init__(
        self,
        ctx: NodeContext,
        *,
        placement: PlacementPolicy,
        forwarding: ForwardingPolicy | None = None,
        retry_interval: float | None = None,
        request_timeout: float = 1.0,
        max_sends: int = 5,
    ):
        if retry_interval is not None and retry_interval <= 0:
            raise ValueError("retry_interval must be positive")
        self.ctx = ctx
        self.placement = placement
        self.forwarding = forwarding
        self.retry_interval = retry_interval
        self._jobs: dict[JobId, _Job] = {}
        # Jobs completed or failed here, and handoffs received, so that
        # resent messages are not mistaken for new work.
        # TODO: bound these sets' memory, for example by age.
        self._done: set[JobId] = set()
        self._handoffs_seen: set[str] = set()
        self._by_reservation: dict[ReservationId, _Held] = {}
        self._reserving: dict[str, JobId] = {}  # request_id -> job
        self._outbox = Retransmitter(timeout=request_timeout, max_sends=max_sends)
        self._assumed: list[_Assumed] = []
        self._lookup_id: str | None = None
        self._dirty = False  # something changed since the current round began
        self._retry_armed = False
        self._next_seq = 0
        self._next_id = 0

    def handle(self, event: Event, now: float) -> list[Action]:
        match event:
            case JobSubmitted(job):
                return self._submitted(job)
            case Received(sender, ForwardJob() as forward):
                return self._received_forward(sender, forward)
            case Received(_, ForwardAck(job_id, handoff_id)):
                return self._forward_acked(job_id, handoff_id)
            case Received(_, CapacityAvailable()):
                return self._request_round()
            case Received(_, LookupReply() as reply):
                return self._plan(reply, now)
            case Received(sender, ReserveReply() as reply):
                return self._reserved(sender, reply)
            case Received(_, StartReply() as reply):
                return self._started(reply)
            case Received(sender, JobFinished() as notice):
                return self._finished(sender, notice)
            case TimerFired(("retry",)):
                return self._retry()
            case TimerFired(("resend", *key)):
                return self._resend(tuple(key))
        return []

    # ==================================================
    # Rounds
    # ==================================================

    def _round_active(self) -> bool:
        # The lookup goes to this node's own dissemination, so its reply is
        # never lost. Reserve requests are resent and eventually abandoned,
        # so a round always ends.
        return self._lookup_id is not None or bool(self._reserving)

    def _pending(self) -> list[_Job]:
        waiting = (j for j in self._jobs.values() if j.phase is _Phase.PENDING)
        return sorted(waiting, key=lambda j: j.seq)

    def _request_round(self) -> list[Action]:
        self._dirty = True
        return self._maybe_start_round()

    def _maybe_start_round(self) -> list[Action]:
        if self._round_active() or not self._dirty or not self._pending():
            return []
        self._dirty = False
        self._lookup_id = self._new_id("lookup")
        return [Send(self.ctx.node_id, LookupRequest(self._lookup_id))]

    def _round_step_done(self) -> list[Action]:
        if self._round_active():
            return []
        actions = self._maybe_start_round()
        if (
            not actions
            and self.retry_interval is not None
            and not self._retry_armed
            and self._pending()
        ):
            self._retry_armed = True
            actions.append(SetTimer(("retry",), self.retry_interval))
        return actions

    def _retry(self) -> list[Action]:
        self._retry_armed = False
        actions: list[Action] = []
        for tracked in self._pending():
            # Waiting ends the lap; if still unplaceable, start a new one.
            tracked.visited = (self.ctx.node_id,)
            actions.append(
                Observe(PlacementRetried(tracked.job.job_id, "no capacity found"))
            )
        return actions + self._request_round()

    # ==================================================
    # Placement
    # ==================================================

    def _submitted(self, job: Job) -> list[Action]:
        if job.job_id in self._jobs or job.job_id in self._done:
            return []
        self._track(job, (self.ctx.node_id,))
        return [Observe(JobAccepted(job.job_id)), *self._request_round()]

    def _received_forward(self, sender: NodeId, forward: ForwardJob) -> list[Action]:
        job = forward.job
        # Acknowledge even duplicates: the sender's earlier ack may be lost.
        actions: list[Action] = [Send(sender, ForwardAck(job.job_id, forward.handoff_id))]
        if forward.handoff_id in self._handoffs_seen or job.job_id in self._done:
            return actions  # a resend
        self._handoffs_seen.add(forward.handoff_id)
        visited = (*forward.visited, self.ctx.node_id)
        tracked = self._jobs.get(job.job_id)
        if tracked is None:
            # Not counted as accepted again: it entered the system elsewhere.
            self._track(job, visited)
            return actions + self._request_round()
        if tracked.phase is _Phase.FORWARDING:
            # Our handoff got through (only its ack is missing), and the job
            # has come back around the ring. It is ours again.
            assert tracked.handoff_id is not None
            actions += self._outbox.acknowledged(("forward", tracked.handoff_id))
            tracked.phase = _Phase.PENDING
            tracked.handoff_id = None
            tracked.visited = visited
            return actions + self._request_round()
        # TODO: a second copy of a job this scheduler is placing, which only
        # happens after an abandoned handoff that had in fact arrived.
        return actions

    def _forward_acked(self, job_id: JobId, handoff_id: str) -> list[Action]:
        actions = self._outbox.acknowledged(("forward", handoff_id))
        tracked = self._jobs.get(job_id)
        if (
            tracked is not None
            and tracked.phase is _Phase.FORWARDING
            and tracked.handoff_id == handoff_id
        ):
            del self._jobs[job_id]
        return actions

    def _track(self, job: Job, visited: tuple[NodeId, ...]) -> None:
        self._jobs[job.job_id] = _Job(job, self._next_seq, visited=visited)
        self._next_seq += 1

    def _plan(self, reply: LookupReply, now: float) -> list[Action]:
        if reply.lookup_id != self._lookup_id:
            return []
        self._lookup_id = None
        candidates = self._apply_assumed(reply.candidates)
        specs = {status.node_id: status.spec for status in candidates}
        pending = self._pending()
        plan = self.placement.plan([j.job for j in pending], candidates, now)

        actions: list[Action] = []
        for assignment in plan:
            tracked = self._jobs[assignment.job_id]
            request_id = self._new_id("reserve")
            tracked.phase = _Phase.RESERVING
            tracked.worker = assignment.node_id
            tracked.group = assignment.group
            tracked.gpu_type = specs[assignment.node_id].gpus[assignment.group].gpu_type
            tracked.request_id = request_id
            self._reserving[request_id] = assignment.job_id
            request = ReserveRequest(request_id, tracked.job, assignment.group)
            actions += self._outbox.send(("reserve", request_id), assignment.node_id, request)
        if self.forwarding is not None:
            actions += self._forward_unplaceable(pending, plan, candidates)
        return actions + self._round_step_done()

    def _forward_unplaceable(
        self,
        pending: list[_Job],
        plan: list[Assignment],
        candidates: list[Status],
    ) -> list[Action]:
        """
        Hand off pending jobs that no candidate has room for after `plan`.

        A job that fits but was left out (for example, behind a blocked
        head under strict FIFO) stays: a later round here can place it.
        """
        assert self.forwarding is not None
        free = {status.node_id: list(status.free_gpus) for status in candidates}
        for assignment in plan:
            job = self._jobs[assignment.job_id].job
            free[assignment.node_id][assignment.group] -= job.gpu_count
        assigned = {assignment.job_id for assignment in plan}
        neighbors = self.ctx.neighbors()

        actions: list[Action] = []
        for tracked in pending:
            job = tracked.job
            if job.job_id in assigned or _has_room(job, candidates, free):
                continue
            hop = self.forwarding.next_hop(job, tracked.visited, neighbors)
            if hop is None:
                continue  # lap over; wait for the retry timer
            # Keep the job until the hop acknowledges it, in case this is lost.
            tracked.phase = _Phase.FORWARDING
            tracked.handoff_id = self._new_id("handoff")
            forward = ForwardJob(job, tracked.visited, tracked.handoff_id)
            actions += self._outbox.send(("forward", tracked.handoff_id), hop, forward)
            actions.append(Observe(JobForwarded(job.job_id, hop)))
        if actions:
            # Forwarding a blocked head can unblock the jobs behind it.
            self._dirty = True
        return actions

    def _apply_assumed(self, candidates: tuple[Status, ...]) -> list[Status]:
        latest = {status.node_id: status.order_key() for status in candidates}
        self._assumed = [
            a
            for a in self._assumed
            if a.node_id not in latest or latest[a.node_id] < a.order_key
        ]
        deltas: dict[NodeId, dict[GroupIndex, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for assumed in self._assumed:
            deltas[assumed.node_id][assumed.group] += assumed.delta

        adjusted = []
        for status in candidates:
            changes = deltas.get(status.node_id)
            if changes:
                free = tuple(
                    min(group.count, max(0, free + changes.get(index, 0)))
                    for index, (group, free) in enumerate(
                        zip(status.spec.gpus, status.free_gpus)
                    )
                )
                status = replace(status, free_gpus=free)
            adjusted.append(status)
        return adjusted

    # ==================================================
    # Reservation, start, completion
    # ==================================================

    def _reserved(self, sender: NodeId, reply: ReserveReply) -> list[Action]:
        actions = self._outbox.acknowledged(("reserve", reply.request_id))
        job_id = self._reserving.pop(reply.request_id, None)
        if job_id is None:
            # Abandoned or answered already. A late grant expires with its lease.
            return actions
        tracked = self._jobs[job_id]
        tracked.request_id = None
        if reply.granted and reply.reservation_id is not None:
            group = reply.group if reply.group is not None else tracked.group
            assert group is not None and reply.order_key is not None
            tracked.group = group
            tracked.phase = _Phase.STARTING
            tracked.reservation_id = reply.reservation_id
            self._by_reservation[reply.reservation_id] = _Held(
                job_id, sender, group, tracked.job.gpu_count
            )
            # Assume the reservation only now, keyed to the grant's version
            # (see `_Assumed`). Until the reply arrived, the open round kept
            # anything from being planned against these GPUs.
            self._assumed.append(
                _Assumed(sender, group, -tracked.job.gpu_count, reply.order_key)
            )
            start = StartRequest(reply.reservation_id)
            actions += self._outbox.send(("start", reply.reservation_id), sender, start)
        else:
            tracked.phase = _Phase.PENDING
            reason = reply.reason or "reservation rejected"
            actions.append(Observe(ReservationRejected(job_id, sender, reason)))
            if reply.status is not None:
                # Correct our view with the worker's current state. Without
                # this, later rounds tend to plan against the same stale
                # record and get rejected by the same worker again. Sent
                # before any new round's lookup, so that lookup sees it.
                actions.append(Send(self.ctx.node_id, StatusPush((reply.status,))))
        return actions + self._round_step_done()

    def _started(self, reply: StartReply) -> list[Action]:
        actions = self._outbox.acknowledged(("start", reply.reservation_id))
        held = self._by_reservation.get(reply.reservation_id)
        tracked = self._jobs.get(held.job_id) if held is not None else None
        if (
            tracked is None
            or tracked.phase is not _Phase.STARTING
            or tracked.reservation_id != reply.reservation_id
        ):
            return actions
        if not reply.started:
            # The reservation expired before the start arrived; try again.
            del self._by_reservation[reply.reservation_id]
            tracked.phase = _Phase.PENDING
            tracked.reservation_id = None
            return actions + self._request_round()
        tracked.phase = _Phase.RUNNING
        assert tracked.worker and tracked.group is not None and tracked.gpu_type
        started = JobStarted(
            tracked.job.job_id,
            tracked.worker,
            tracked.group,
            tracked.gpu_type,
            tracked.job.gpu_count,
        )
        return [*actions, Observe(started)]

    # TODO: handle worker crashes. A job running on a worker that crashes is
    # never reported finished, so it is lost silently. One way to detect
    # it: a record whose `incarnation` is higher than the grant's means the
    # worker restarted, so its jobs should be requeued (and recorded with a
    # new `JobInterrupted` metric).
    def _finished(self, sender: NodeId, notice: JobFinished) -> list[Action]:
        # Acknowledge even duplicates: the worker's earlier ack may be lost.
        actions: list[Action] = [Send(sender, FinishedAck(notice.reservation_id))]
        held = self._by_reservation.pop(notice.reservation_id, None)
        if held is None:
            return actions
        self._assumed.append(
            _Assumed(held.worker, held.group, held.gpu_count, notice.order_key)
        )
        tracked = self._jobs.pop(held.job_id, None)
        if tracked is None:
            return actions  # a duplicate run of a job that already finished
        # The first run to finish counts; stop whatever else the job was doing.
        # TODO: cancel a duplicate run that is still going on another worker.
        actions += self._stop_in_flight(tracked)
        self._done.add(held.job_id)
        if notice.succeeded:
            outcome: Action = Observe(JobCompleted(held.job_id, sender))
        else:
            outcome = Observe(JobFailed(held.job_id, notice.reason or "execution failed"))
        return [*actions, outcome, *self._request_round()]

    def _stop_in_flight(self, tracked: _Job) -> list[Action]:
        """
        Stop resending this job's outstanding request, if it has one.
        """
        if tracked.phase is _Phase.RESERVING and tracked.request_id is not None:
            self._reserving.pop(tracked.request_id, None)
            return self._outbox.acknowledged(("reserve", tracked.request_id))
        if tracked.phase is _Phase.STARTING and tracked.reservation_id is not None:
            return self._outbox.acknowledged(("start", tracked.reservation_id))
        if tracked.phase is _Phase.FORWARDING and tracked.handoff_id is not None:
            return self._outbox.acknowledged(("forward", tracked.handoff_id))
        return []

    # ==================================================
    # Lost messages
    # ==================================================

    def _resend(self, key: Key) -> list[Action]:
        actions, abandoned = self._outbox.timed_out(key)
        if abandoned is None:
            return actions
        kind, ident = key
        tracked: _Job | None = None
        if kind == "reserve":
            job_id = self._reserving.pop(ident, None)
            tracked = self._jobs.get(job_id) if job_id is not None else None
            if tracked is not None:
                tracked.request_id = None
        elif kind == "start":
            held = self._by_reservation.get(ident)
            tracked = self._jobs.get(held.job_id) if held is not None else None
            if tracked is not None and tracked.reservation_id != ident:
                tracked = None
            # `held` stays: the job may be running, and its finish still counts.
        elif kind == "forward":
            assert isinstance(abandoned.msg, ForwardJob)
            tracked = self._jobs.get(abandoned.msg.job.job_id)
            if tracked is not None and tracked.handoff_id != ident:
                tracked = None
            if tracked is not None:
                tracked.handoff_id = None
        if tracked is None:
            return self._round_step_done()
        # The receiver may have acted on the request anyway, so the job may
        # run twice; the first run to finish counts.
        tracked.phase = _Phase.PENDING
        tracked.reservation_id = None
        tracked.visited = (self.ctx.node_id,)
        request = type(abandoned.msg).__name__
        return [
            Observe(RequestAbandoned(tracked.job.job_id, abandoned.to, request)),
            *self._request_round(),
        ]

    def _new_id(self, kind: str) -> str:
        self._next_id += 1
        return f"{self.ctx.node_id}:{self.ctx.incarnation}:{kind}-{self._next_id}"


def _has_room(job: Job, candidates: list[Status], free: dict[NodeId, list[int]]) -> bool:
    return any(
        job.can_use(group) and free[status.node_id][index] >= job.gpu_count
        for status in candidates
        for index, group in enumerate(status.spec.gpus)
    )
