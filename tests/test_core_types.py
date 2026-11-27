"""
Tests for the torchsched core types and Node composition.
"""

import ast
import dataclasses
from pathlib import Path

import pytest

from torchsched.node import Node, NodeContext
from torchsched.types.actions import CancelTimer, Send, SetTimer
from torchsched.types.events import Boot, ExecutionFinished, JobSubmitted, Received, TimerFired
from torchsched.types.messages import (
    ForwardJob,
    LookupRequest,
    Message,
    ProbeRequest,
    ReserveRequest,
)
from torchsched.types.models import GpuGroup, Job, NodeSpec, Status

# A workstation with mixed cards: one 24 GB RTX 4090 and two 48 GB A6000s.
MIXED = NodeSpec("ws-1", [GpuGroup("RTX4090", 1, 24), GpuGroup("A6000", 2, 48)])


def job(job_id="j1", gpu_count=1, vram=16, runtimes=None):
    return Job(job_id, gpu_count, vram, runtimes or {"RTX4090": 2.0, "A6000": 3.0})


# ==================================================
# Models
# ==================================================


def test_gpu_group_and_node_spec_validation():
    with pytest.raises(ValueError):
        GpuGroup("A100", 0, 80)
    with pytest.raises(ValueError):
        GpuGroup("A100", 1, 0)
    with pytest.raises(ValueError):
        GpuGroup("", 1, 80)
    with pytest.raises(ValueError):
        NodeSpec("")

    assert MIXED.gpus == (GpuGroup("RTX4090", 1, 24), GpuGroup("A6000", 2, 48))
    assert MIXED.gpu_count == 3
    coordinator = NodeSpec("coordinator")
    assert coordinator.gpus == () and coordinator.gpu_count == 0


def test_job_validation_and_immutability():
    for bad in (
        dict(job_id=""),
        dict(gpu_count=0),
        dict(vram=-1),
        dict(runtimes={"A6000": 0}),
    ):
        with pytest.raises(ValueError):
            job(**bad)
    with pytest.raises(ValueError):
        Job("j", 1, 1, {})

    original = {"A6000": 3.0}
    j = job(runtimes=original)
    original["A6000"] = 99.0
    assert j.runtime_by_gpu["A6000"] == 3.0
    with pytest.raises(TypeError):
        j.runtime_by_gpu["A6000"] = 1.0  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        j.gpu_count = 2  # type: ignore[misc]


def test_jobs_and_messages_containing_jobs_are_hashable():
    a = job(runtimes={"RTX4090": 2.0, "A6000": 3.0})
    b = job(runtimes={"A6000": 3.0, "RTX4090": 2.0})
    assert a == b and hash(a) == hash(b)
    assert len({a, b, job(job_id="j2")}) == 2
    assert hash(ReserveRequest("r1", a, 1)) == hash(ReserveRequest("r1", b, 1))
    assert {ForwardJob(a, ("n1",), "h1"), ForwardJob(b, ("n1",), "h1")} == {ForwardJob(a, ("n1",), "h1")}


def test_compatibility_considers_each_group_separately():
    assert job(gpu_count=1, vram=16).can_use(MIXED.gpus[0])
    # Needs 2 GPUs: only the A6000 group has two, even though the node has three.
    two = job(gpu_count=2, vram=16)
    assert not two.can_use(MIXED.gpus[0])
    assert two.can_use(MIXED.gpus[1])
    # 3 GPUs exceed every single group; jobs never span groups.
    assert not any(job(gpu_count=3).can_use(group) for group in MIXED.gpus)
    # Type mismatch and VRAM limits.
    assert not any(job(runtimes={"A100": 1.0}).can_use(group) for group in MIXED.gpus)
    assert not job(vram=40).can_use(MIXED.gpus[0])
    assert job(vram=40).can_use(MIXED.gpus[1])


def test_status_reports_free_gpus_per_group():
    status = Status(MIXED, [1, 1], incarnation=1, version=4, observed_at=2.0)
    assert status.free_gpus == (1, 1)
    assert status.node_id == "ws-1"
    assert status.order_key() == (1, 4)
    assert status.fitting_groups(job(gpu_count=1, vram=16)) == (0, 1)
    assert status.fitting_groups(job(gpu_count=1, vram=40)) == (1,)
    # Compatible with the A6000 group, but only one of its two GPUs is free.
    assert status.fitting_groups(job(gpu_count=2)) == ()

    with pytest.raises(ValueError):
        Status(MIXED, [1], 1, 0, 0.0)
    with pytest.raises(ValueError):
        Status(MIXED, [2, 0], 1, 0, 0.0)
    with pytest.raises(ValueError):
        Status(MIXED, [-1, 0], 1, 0, 0.0)


def test_status_order_puts_a_restart_after_any_earlier_version():
    before_crash = Status(MIXED, [0, 0], incarnation=1, version=50, observed_at=1.0)
    after_restart = Status(MIXED, [1, 2], incarnation=2, version=0, observed_at=2.0)
    assert after_restart.order_key() > before_crash.order_key()


# ==================================================
# Messages
# ==================================================


def test_messages_are_registered_and_names_must_be_unique():
    assert Message.registry["ReserveRequest"] is ReserveRequest
    with pytest.raises(TypeError, match="duplicate message type"):
        # A different class reusing an existing message name.
        type("ReserveRequest", (Message,), {"__slots__": ()})


def test_messages_are_frozen():
    message = ReserveRequest("r1", job())
    assert message.group is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.request_id = "r2"  # type: ignore[misc]


# ==================================================
# Node composition
# ==================================================


class Fake:
    """
    A component that records what it handled and returns canned actions.
    """

    name = "fake"
    accepts: frozenset[type] = frozenset()

    def __init__(self, on_boot=(), reply=()):
        self.handled = []
        self.on_boot = list(on_boot)
        self.reply = list(reply)

    def handle(self, event, now):
        self.handled.append((event, now))
        return list(self.on_boot) if isinstance(event, Boot) else list(self.reply)


class FakeWorker(Fake):
    name = "worker"
    accepts = frozenset({ReserveRequest, ExecutionFinished})

    def status(self, now):
        return Status(MIXED, [1, 2], 1, 0, now)


class FakeMembership(Fake):
    name = "membership"

    def neighbors(self):
        return frozenset({"n2", "n3"})


class FakeDissemination(Fake):
    name = "dissemination"
    accepts = frozenset({LookupRequest})


class FakeScheduler(Fake):
    name = "scheduler"
    accepts = frozenset({JobSubmitted})


def make_node(**components):
    """
    Build a Node from fakes. Pass a component to override it, or worker=None
    for a scheduler-only node.
    """
    ctx = NodeContext(MIXED, incarnation=1, seed=42)
    parts = dict(
        worker=FakeWorker(),
        membership=FakeMembership(),
        dissemination=FakeDissemination(),
        scheduler=FakeScheduler(),
    )
    parts.update(components)
    node = Node(ctx, **parts)
    return node, {name: part for name, part in parts.items() if part is not None}


def test_boot_reaches_every_component_and_timer_keys_are_namespaced():
    node, parts = make_node(
        worker=None,
        dissemination=FakeDissemination(on_boot=[SetTimer(("gossip",), 0.5)]),
        scheduler=FakeScheduler(on_boot=[CancelTimer(("retry", "j1"))]),
    )
    actions = node.handle(Boot(), 0.0)
    assert actions == [
        SetTimer(("dissemination", "gossip"), 0.5),
        CancelTimer(("scheduler", "retry", "j1")),
    ]
    for name in ("membership", "dissemination", "scheduler"):
        assert parts[name].handled == [(Boot(), 0.0)]


def test_timer_fires_only_at_its_owner_with_the_prefix_removed():
    node, parts = make_node(
        dissemination=FakeDissemination(reply=[SetTimer(("gossip",), 0.5)])
    )
    actions = node.handle(TimerFired(("dissemination", "gossip")), 1.0)
    assert parts["dissemination"].handled == [(TimerFired(("gossip",)), 1.0)]
    assert parts["scheduler"].handled == []
    # Re-arming from inside the handler is namespaced again.
    assert actions == [SetTimer(("dissemination", "gossip"), 0.5)]


def test_messages_and_events_route_by_type():
    node, parts = make_node(
        worker=FakeWorker(reply=[Send("n2", ProbeRequest("p1"))]),
    )
    reserve = Received("n2", ReserveRequest("r1", job()))
    assert node.handle(reserve, 2.0) == [Send("n2", ProbeRequest("p1"))]
    assert parts["worker"].handled == [(reserve, 2.0)]

    node.handle(Received("ws-1", LookupRequest("l1")), 2.0)
    assert parts["dissemination"].handled[-1][0].msg == LookupRequest("l1")

    node.handle(JobSubmitted(job()), 3.0)
    node.handle(ExecutionFinished("res-1", succeeded=True), 4.0)
    assert parts["scheduler"].handled[-1][0] == JobSubmitted(job())
    assert parts["worker"].handled[-1][0] == ExecutionFinished("res-1", True)


def test_unhandled_messages_are_dropped():
    node, parts = make_node()
    assert node.handle(Received("n9", ProbeRequest("p1")), 0.0) == []
    assert all(not part.handled for part in parts.values())


def test_conflicting_components_are_rejected():
    class AlsoAcceptsLookup(FakeScheduler):
        accepts = frozenset({LookupRequest})

    with pytest.raises(ValueError, match="LookupRequest accepted by both"):
        make_node(scheduler=AlsoAcceptsLookup())

    class NamedLikeWorker(FakeScheduler):
        name = "worker"

    with pytest.raises(ValueError, match="duplicate component name"):
        make_node(scheduler=NamedLikeWorker())


# ==================================================
# NodeContext
# ==================================================


def test_context_reads_are_bound_by_node():
    ctx = NodeContext(MIXED, incarnation=1, seed=42)
    with pytest.raises(RuntimeError):
        ctx.neighbors()

    node, _ = make_node()
    assert node.ctx.neighbors() == frozenset({"n2", "n3"})
    assert node.ctx.local_status(5.0).observed_at == 5.0
    assert node.node_id == "ws-1"

    scheduler_only, _ = make_node(worker=None)
    assert scheduler_only.ctx.local_status(5.0) is None


def test_random_streams_are_deterministic_and_independent():
    def draws(ctx, purpose):
        return [ctx.rng(purpose).random() for _ in range(3)]

    a = NodeContext(MIXED, incarnation=1, seed=42)
    b = NodeContext(MIXED, incarnation=1, seed=42)
    # Extra draws from one stream do not shift another stream.
    a.rng("gossip").random()
    assert draws(a, "sample") == draws(b, "sample")
    assert draws(NodeContext(MIXED, 2, 42), "sample") != draws(
        NodeContext(MIXED, 1, 42), "sample"
    )
    assert a.rng("gossip") is a.rng("gossip")


# ==================================================
# Core rule: no I/O, real clocks, or global randomness
# ==================================================

FORBIDDEN_MODULES = {
    "asyncio",
    "fastapi",
    "httpx",
    "simpy",
    "socket",
    "threading",
    "time",
    "uvicorn",
}

# Runtimes that drive nodes over real I/O. They are the only modules allowed
# to break the rule; nothing in the core may import them.
IO_RUNTIMES = {"runtime/http.py"}


def test_core_does_not_import_io_or_global_randomness():
    core = Path(__file__).resolve().parents[1] / "torchsched"
    violations = []
    for path in sorted(core.rglob("*.py")):
        relative = path.relative_to(core).as_posix()
        if relative in IO_RUNTIMES:
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                # `import random` invites module-level random.random() calls.
                if "random" in modules:
                    violations.append(f"{path.name}: import random")
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
                if node.module == "random" and any(
                    alias.name != "Random" for alias in node.names
                ):
                    violations.append(f"{path.name}: from random import non-Random name")
            else:
                continue
            for module in modules:
                if module.split(".")[0] in FORBIDDEN_MODULES:
                    violations.append(f"{path.name}: imports {module}")
    assert not violations, violations


def test_only_runtimes_import_io_runtimes():
    core = Path(__file__).resolve().parents[1] / "torchsched"
    runtime_modules = {
        "torchsched." + name.removesuffix(".py").replace("/", ".") for name in IO_RUNTIMES
    }
    violations = []
    for path in sorted(core.rglob("*.py")):
        if path.relative_to(core).as_posix() in IO_RUNTIMES:
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            violations += [f"{path.name}: imports {n}" for n in names if n in runtime_modules]
    assert not violations, violations
