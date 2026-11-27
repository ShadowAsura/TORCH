"""
Tests for the message encoder and simulator byte counts.
"""

import json
from types import MappingProxyType

import pytest

from torchsched import clusters
from torchsched.codec import decode, encode
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.types.messages import (
    CapacityAvailable,
    FinishedAck,
    ForwardAck,
    ForwardJob,
    Heartbeat,
    JobFinished,
    LookupReply,
    LookupRequest,
    Message,
    ProbeReply,
    ProbeRequest,
    ReserveReply,
    ReserveRequest,
    StartReply,
    StartRequest,
    StatusChanged,
    StatusPush,
)
from torchsched.types.models import GpuGroup, Job, NodeSpec, Status

SPEC = NodeSpec("w1", [GpuGroup("A100", 2, 80.0), GpuGroup("L40", 1, 48.0)])
STATUS = Status(SPEC, (1, 0), incarnation=2, version=7, observed_at=3.25)
JOB = Job("j1", 2, 40.0, {"A100": 1.5, "L40": 2.0})

# One instance of every message type, using as many fields as possible.
SAMPLES = {
    ReserveRequest: ReserveRequest("r1", JOB, 1),
    ReserveReply: ReserveReply("r1", "j1", True, "res-1", 0, order_key=(2, 8)),
    StartRequest: StartRequest("res-1"),
    StartReply: StartReply("res-1", False, "expired"),
    JobFinished: JobFinished("j1", "res-1", True, (2, 9), None),
    FinishedAck: FinishedAck("res-1"),
    Heartbeat: Heartbeat(("res-1", "res-2")),
    ProbeRequest: ProbeRequest("p1"),
    ProbeReply: ProbeReply("p1", STATUS),
    StatusPush: StatusPush((STATUS, Status(NodeSpec("w2", [GpuGroup("T4", 4, 16.0)]), (4,), 1, 0, 0.0))),
    StatusChanged: StatusChanged(STATUS),
    CapacityAvailable: CapacityAvailable(),
    LookupRequest: LookupRequest("l1"),
    LookupReply: LookupReply("l1", (STATUS,)),
    ForwardJob: ForwardJob(JOB, ("a", "b"), "a:1:handoff-3"),
    ForwardAck: ForwardAck("j1", "a:1:handoff-3"),
}


def test_every_registered_message_has_a_sample():
    """
    Add a sample above when adding a message type.
    """
    assert set(SAMPLES) == set(Message.registry.values())


@pytest.mark.parametrize("msg", SAMPLES.values(), ids=lambda m: type(m).__name__)
def test_round_trip_preserves_values_and_types(msg):
    decoded = decode(encode(msg))
    assert decoded == msg
    assert type(decoded) is type(msg)


def test_decoded_models_keep_their_exact_types():
    status = decode(encode(StatusPush((STATUS,)))).statuses[0]
    assert isinstance(status.free_gpus, tuple) and isinstance(status.spec.gpus, tuple)
    assert isinstance(status.spec.gpus[0], GpuGroup)
    job = decode(encode(ForwardJob(JOB, ("a",), "h1"))).job
    assert isinstance(job.runtime_by_gpu, MappingProxyType)
    reply = decode(encode(ReserveReply("r", "j", False, reason="busy", status=STATUS)))
    assert reply.status == STATUS and reply.order_key is None
    assert decode(encode(JobFinished("j", "r", True, (1, 2)))).order_key == (1, 2)


def test_encoding_is_compact_json_with_a_type_tag():
    data = encode(StartRequest("res-1"))
    assert data == b'{"type":"StartRequest","body":{"reservation_id":"res-1"}}'


def body(msg):
    return json.loads(encode(msg))["body"]


def raw(type_name, body):
    return json.dumps({"type": type_name, "body": body}).encode()


@pytest.mark.parametrize(
    "data,problem",
    [
        (b"not json", "not JSON"),
        (b"[]", "exactly 'type' and 'body'"),
        (raw("Nope", {}), "unknown message type"),
        (raw("StartRequest", {}), "missing"),
        (raw("StartRequest", {"reservation_id": "r", "extra": 1}), "unexpected"),
        (raw("StartRequest", {"reservation_id": 5}), "expected a string"),
        (raw("StartReply", {"reservation_id": "r", "started": 1, "reason": None}), "boolean"),
        (raw("JobFinished", {**body(JobFinished("j", "r", True, (1, 2))), "order_key": [1]}), "2 items"),
        (raw("ReserveRequest", {**body(ReserveRequest("r", JOB)), "group": 1.5}), "integer"),
        # Model validation runs on decode: more free GPUs than the group has.
        (raw("StatusChanged", {"status": {**body(StatusChanged(STATUS))["status"], "free_gpus": [9, 0]}}), "within"),
    ],
)
def test_malformed_input_is_rejected(data, problem):
    with pytest.raises(ValueError, match=problem):
        decode(data)


def test_unencodable_values_are_rejected():
    with pytest.raises(TypeError):
        encode("not a message")
    with pytest.raises(ValueError):  # JSON has no NaN
        encode(StatusChanged(Status(SPEC, (1, 0), 1, 0, float("nan"))))


def test_simulator_counts_bytes_of_network_messages_only():
    specs = [NodeSpec("w1", [GpuGroup("A100", 2, 80.0)])]
    sim = Simulator(clusters.centralized(specs, placement=FirstFit()), latency=0.01)
    sim.submit(Job("j1", 1, 40.0, {"A100": 1.0}), at=0.0, node_id="coordinator")
    result = sim.run()
    assert set(result.message_bytes) == set(result.messages)
    # Self-sends (LookupRequest, StatusChanged, ...) are not network traffic.
    assert "LookupRequest" not in result.messages and "StatusChanged" not in result.messages
    start = len(encode(StartRequest("w1:1:0")))
    assert result.message_bytes["StartRequest"] == start * result.messages["StartRequest"]
    assert all(result.message_bytes[k] > 0 for k in result.messages)
