"""
Stale or reordered status records must never make a scheduler double-book.

These tests drive the scheduler directly and play the worker and the view
by hand, so they control exactly which (possibly stale) record arrives when.
"""

from torchsched.dissemination.centralized import CentralizedDissemination
from torchsched.node import NodeContext
from torchsched.placement.first_fit import FirstFit
from torchsched.scheduler import Scheduler
from torchsched.types.actions import Send
from torchsched.types.events import JobSubmitted, Received
from torchsched.types.messages import (
    CapacityAvailable,
    JobFinished,
    LookupReply,
    LookupRequest,
    ReserveReply,
    ReserveRequest,
    StartRequest,
    StatusPush,
)
from torchsched.types.models import GpuGroup, Job, NodeSpec, Status

WORKER = NodeSpec("w1", [GpuGroup("A100", 2, 80)])
COORDINATOR = NodeSpec("coordinator")


def record(version, free):
    """
    A status record from w1 at `version` with `free` GPUs.
    """
    return Status(WORKER, (free,), incarnation=1, version=version, observed_at=0.0)


def job(job_id):
    return Job(job_id, 2, 40, {"A100": 1.0})


def sends(actions, kind):
    return [a for a in actions if isinstance(a, Send) and isinstance(a.msg, kind)]


class Harness:
    """
    A scheduler whose lookups are answered with whatever record we choose.
    """

    def __init__(self):
        ctx = NodeContext(COORDINATOR, incarnation=1, seed=0)
        self.scheduler = Scheduler(ctx, placement=FirstFit())
        self.lookup_id = None

    def feed(self, event):
        actions = self.scheduler.handle(event, 0.0)
        for send in sends(actions, LookupRequest):
            self.lookup_id = send.msg.lookup_id
        return actions

    def submit(self, job_id):
        return self.feed(JobSubmitted(job(job_id)))

    def answer_lookup(self, *candidates):
        """
        Reply to the outstanding lookup; return the ReserveRequests sent.
        """
        assert self.lookup_id is not None, "no lookup outstanding"
        lookup_id, self.lookup_id = self.lookup_id, None
        actions = self.feed(Received("coordinator", LookupReply(lookup_id, candidates)))
        return [send.msg for send in sends(actions, ReserveRequest)]

    def grant(self, request, version):
        reply = ReserveReply(
            request.request_id,
            request.job.job_id,
            True,
            f"res-{request.job.job_id}",
            group=0,
            order_key=(1, version),
        )
        return self.feed(Received("w1", reply))

    def finish(self, job_id, version):
        notice = JobFinished(job_id, f"res-{job_id}", True, (1, version))
        return self.feed(Received("w1", notice))


def test_stale_record_arriving_after_a_grant_does_not_double_book():
    h = Harness()
    h.submit("j1")
    [request] = h.answer_lookup(record(version=4, free=2))
    assert request.job.job_id == "j1"
    # The worker grants j1, moving to version 5 (no free GPUs).
    assert sends(h.grant(request, version=5), StartRequest)

    # j2 arrives. The view still holds the version-4 record, from before the
    # grant: the push for version 5 is late. The assumption must still apply.
    h.submit("j2")
    assert h.answer_lookup(record(version=4, free=2)) == []

    # Once the version-5 record arrives the assumption is dropped, and the
    # record alone says the GPUs are busy.
    h.feed(Received("coordinator", CapacityAvailable()))
    assert h.answer_lookup(record(version=5, free=0)) == []


def test_updates_while_the_request_is_in_flight_do_not_drop_the_assumption():
    h = Harness()
    h.submit("j1")
    [request] = h.answer_lookup(record(version=4, free=2))

    # Before the request reaches the worker, the worker makes many local
    # changes (versions 5..9), and their records reach the scheduler first.
    # j2 arrives meanwhile; no round can run while j1's request is
    # unanswered, so nothing is planned against those records yet.
    h.submit("j2")
    assert h.lookup_id is None

    # The grant comes after all of them, at version 10. That closes the
    # round, and the waiting j2 starts a new one.
    h.grant(request, version=10)

    # Every record from the in-flight window predates the grant, so the
    # assumption still applies to each of them and j2 is not placed.
    for version in range(5, 10):
        assert h.answer_lookup(record(version=version, free=2)) == []
        h.feed(Received("coordinator", CapacityAvailable()))

    # Only a record that includes the grant (version >= 10) replaces it.
    assert h.answer_lookup(record(version=10, free=0)) == []


def test_release_assumption_frees_gpus_before_the_record_catches_up():
    h = Harness()
    h.submit("j1")
    [request] = h.answer_lookup(record(version=4, free=2))
    h.grant(request, version=5)
    h.submit("j2")
    assert h.answer_lookup(record(version=4, free=2)) == []

    # j1 finishes; the worker is now at version 6 with both GPUs free, but
    # no record newer than version 5 has arrived yet.
    h.finish("j1", version=6)
    [placed] = h.answer_lookup(record(version=5, free=0))
    assert placed.job.job_id == "j2"


def test_record_from_before_both_grant_and_release_nets_to_free():
    h = Harness()
    h.submit("j1")
    [request] = h.answer_lookup(record(version=4, free=2))
    h.grant(request, version=5)
    h.finish("j1", version=6)
    h.submit("j2")
    # The version-4 record predates both changes: -2 then +2 leaves 2 free.
    # Ignoring both assumptions gives the same answer; this test guards
    # against applying the reservation but forgetting the release.
    [placed] = h.answer_lookup(record(version=4, free=2))
    assert placed.job.job_id == "j2"


def test_coordinator_view_ignores_records_that_arrive_out_of_order():
    ctx = NodeContext(COORDINATOR, incarnation=1, seed=0)
    view = CentralizedDissemination(ctx, coordinator="coordinator")
    view.handle(Received("w1", StatusPush((record(version=5, free=0),))), 0.0)
    # The late version-4 push is older than what the view holds: dropped.
    actions = view.handle(Received("w1", StatusPush((record(version=4, free=2),))), 0.0)
    assert view.view["w1"].version == 5
    assert actions == []  # and it does not claim capacity appeared


def test_rejection_status_corrects_the_view_before_the_next_lookup():
    h = Harness()
    h.submit("j1")
    [request] = h.answer_lookup(record(version=4, free=2))
    h.submit("j2")  # arrives mid-round, so the round ends with a new lookup

    current = record(version=7, free=0)
    reply = ReserveReply(
        request.request_id, "j1", False, reason="lacks free GPUs", status=current
    )
    actions = h.feed(Received("w1", reply))
    kinds = [type(a.msg) for a in actions if isinstance(a, Send)]
    # The correction goes to our own dissemination, ahead of the next lookup.
    assert kinds == [StatusPush, LookupRequest]
    [push] = sends(actions, StatusPush)
    assert push.to == "coordinator"
    assert push.msg == StatusPush((current,))


def test_rejection_without_status_sends_no_correction():
    h = Harness()
    h.submit("j1")
    [request] = h.answer_lookup(record(version=4, free=2))
    reply = ReserveReply(request.request_id, "j1", False, reason="lacks free GPUs")
    assert not sends(h.feed(Received("w1", reply)), StatusPush)
