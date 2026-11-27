"""
Unit tests for the sans-I/O worker: feed events, check returned actions.
"""

import pytest

from torchsched.node import NodeContext
from torchsched.types.actions import CancelTimer, Execute, Send, SetTimer
from torchsched.types.events import ExecutionFinished, Received, TimerFired
from torchsched.types.messages import (
    JobFinished,
    ReserveReply,
    ReserveRequest,
    StartReply,
    StartRequest,
    StatusChanged,
)
from torchsched.types.models import GpuGroup, Job, NodeSpec
from torchsched.worker import Worker

SPEC = NodeSpec("w1", [GpuGroup("L40", 2, 48), GpuGroup("A100", 4, 80)])


def make_worker(lease=10.0):
    return Worker(NodeContext(SPEC, incarnation=3, seed=0), lease_seconds=lease)


def job(job_id="j1", gpus=2, vram=40, runtimes=None):
    return Job(job_id, gpus, vram, runtimes or {"L40": 4.0, "A100": 2.0})


def sent(actions, kind):
    return [a.msg for a in actions if isinstance(a, Send) and isinstance(a.msg, kind)]


def reserve(worker, request_id="r1", j=None, group=None, sender="sched", now=0.0):
    request = ReserveRequest(request_id, j or job(), group)
    return worker.handle(Received(sender, request), now)


def test_reserve_grants_atomically_and_reports_the_change():
    worker = make_worker()
    actions = reserve(worker, group=1)
    [reply] = sent(actions, ReserveReply)
    assert reply.granted and reply.group == 1 and reply.job_id == "j1"
    assert reply.reservation_id == "w1:3:0"
    assert reply.order_key == (3, 1)
    assert SetTimer(("lease", reply.reservation_id), 10.0) in actions
    [changed] = sent(actions, StatusChanged)
    assert changed.status.free_gpus == (2, 2)
    assert worker.status(1.0).free_gpus == (2, 2)
    assert worker.status(1.0).order_key() == (3, 1)
    # Replies go to the requester; status changes go to this node.
    assert {a.to for a in actions if isinstance(a, Send)} == {"sched", "w1"}


def test_reserve_rejects_without_changing_state():
    worker = make_worker()
    cases = [
        (job(gpus=3), 0, "cannot run"),  # L40 group only has 2 GPUs
        (job(vram=64), 0, "cannot run"),  # L40 has 48 GB
        (job(runtimes={"H100": 1.0}), None, "no GPU group"),
        (job(), 7, "cannot run"),  # no such group
    ]
    for j, group, reason in cases:
        actions = reserve(worker, j=j, group=group)
        [reply] = sent(actions, ReserveReply)
        assert not reply.granted and reason in reply.reason
        assert not sent(actions, StatusChanged)
    assert worker.status(0).free_gpus == (2, 4) and worker.version == 0

    reserve(worker, "r1", group=0)
    [reply] = sent(reserve(worker, "r2", job("j2"), group=0), ReserveReply)
    assert not reply.granted and "lacks free GPUs" in reply.reason


def test_rejection_carries_current_status_and_grant_does_not():
    worker = make_worker()
    [grant] = sent(reserve(worker, "r1", group=0, now=1.0), ReserveReply)
    assert grant.granted and grant.status is None
    [reject] = sent(reserve(worker, "r2", job("j2"), group=0, now=2.0), ReserveReply)
    assert not reject.granted
    assert reject.status == worker.status(2.0)
    assert reject.status.free_gpus == (0, 4)
    assert reject.status.order_key() == (3, 1)


def test_unspecified_group_takes_the_first_that_fits():
    worker = make_worker()
    [first] = sent(reserve(worker, "r1", job("j1")), ReserveReply)
    [second] = sent(reserve(worker, "r2", job("j2")), ReserveReply)
    assert (first.group, second.group) == (0, 1)


def test_duplicate_request_repeats_the_grant():
    worker = make_worker()
    [first] = sent(reserve(worker), ReserveReply)
    again = reserve(worker)
    assert sent(again, ReserveReply) == [first]
    assert not sent(again, StatusChanged)
    assert worker.status(0).free_gpus == (0, 4)
    # The same request id from another scheduler is a different request.
    [other] = sent(reserve(worker, sender="sched-2", group=1), ReserveReply)
    assert other.reservation_id != first.reservation_id


def test_start_runs_the_job_and_finishing_releases_gpus():
    worker = make_worker()
    [granted] = sent(reserve(worker, group=1), ReserveReply)
    rid = granted.reservation_id

    actions = worker.handle(Received("sched", StartRequest(rid)), 1.0)
    assert CancelTimer(("lease", rid)) in actions
    assert Execute(rid, job(), 1, "A100") in actions
    assert sent(actions, StartReply) == [StartReply(rid, True)]
    # Starting twice is harmless.
    again = worker.handle(Received("sched", StartRequest(rid)), 1.5)
    assert again == [Send("sched", StartReply(rid, True))]

    actions = worker.handle(ExecutionFinished(rid, True), 3.0)
    [notice] = sent(actions, JobFinished)
    assert notice == JobFinished("j1", rid, True, (3, 2))
    assert [a.to for a in actions if isinstance(a, Send)] == ["sched", "w1"]
    assert worker.status(3.0).free_gpus == (2, 4)
    # A duplicate finish is ignored.
    assert worker.handle(ExecutionFinished(rid, True), 3.0) == []


def test_only_the_holder_can_start_a_reservation():
    worker = make_worker()
    [granted] = sent(reserve(worker), ReserveReply)
    actions = worker.handle(Received("intruder", StartRequest(granted.reservation_id)), 1)
    [reply] = sent(actions, StartReply)
    assert not reply.started
    assert not [a for a in actions if isinstance(a, Execute)]


def test_lease_expiry_releases_and_blocks_a_late_start():
    worker = make_worker()
    [granted] = sent(reserve(worker), ReserveReply)
    rid = granted.reservation_id

    actions = worker.handle(TimerFired(("lease", rid)), 10.0)
    [changed] = sent(actions, StatusChanged)
    assert changed.status.free_gpus == (2, 4)
    [late] = sent(worker.handle(Received("sched", StartRequest(rid)), 11.0), StartReply)
    assert not late.started and "expired" in late.reason
    # Once released, the request id can be granted again.
    [fresh] = sent(reserve(worker), ReserveReply)
    assert fresh.granted and fresh.reservation_id != rid


def test_lease_timer_after_start_does_nothing():
    worker = make_worker()
    [granted] = sent(reserve(worker), ReserveReply)
    worker.handle(Received("sched", StartRequest(granted.reservation_id)), 1.0)
    assert worker.handle(TimerFired(("lease", granted.reservation_id)), 10.0) == []
    assert worker.status(10.0).free_gpus == (0, 4)


def test_worker_requires_gpus_and_a_positive_lease():
    with pytest.raises(ValueError):
        Worker(NodeContext(NodeSpec("coordinator"), 1, 0))
    with pytest.raises(ValueError):
        make_worker(lease=0)
