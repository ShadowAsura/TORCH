"""
Fault scenarios: crashes, network partitions and message loss.

A scenario is JSON; every section is optional:

    {
      "crashes": [
        {"node": "a100-1", "at": 5, "restart_at": 30},
        {"node": "l40-2", "at": 20}
      ],
      "random_crashes": {"count": 2, "downtime": 5, "window": [0, 20]},
      "partitions": [{"nodes": ["l40-1", "l40-2"], "at": 10, "heal_at": 30}],
      "losses": [
        {"probability": 0.05},
        {"end": 5, "kinds": ["StatusPush"]}
      ]
    }

Times are in workload seconds, the clock job arrivals use, so a scenario
means the same thing in the simulator (`Simulator.apply`) and in the
Docker demo (which scales times by its execution scale).

* A crash takes the whole node down; without `restart_at` it stays down. A
  node may crash more than once, but only after it has restarted.
* `random_crashes` crashes `count` distinct nodes (not ones crashed by name)
  at uniform times in `window`, by default while jobs arrive, each down for
  `downtime` or for good. `Scenario.resolve` picks them from a seed.
* A partition cuts `nodes` off from every other node (both directions)
  until `heal_at`, or for good. The cut-off nodes still reach each other. A
  node is in at most one partition at a time.
* A loss drops matching network messages; see `MessageLoss`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from typing import Any

from torchsched.types.models import NodeId


def _floats(record: object, *names: str) -> None:
    """
    Store times as floats, so an int never leaks into simulated time.
    """
    for name in names:
        value = getattr(record, name)
        if value is not None:
            object.__setattr__(record, name, float(value))


@dataclass(frozen=True, slots=True)
class Crash:
    """
    `node` goes down at `at` and, if `restart_at` is given, comes back then
    with empty state.
    """

    node: NodeId
    at: float
    restart_at: float | None = None

    def __post_init__(self) -> None:
        _floats(self, "at", "restart_at")
        if self.at < 0:
            raise ValueError(f"crash of {self.node!r}: at must be nonnegative")
        if self.restart_at is not None and self.restart_at <= self.at:
            raise ValueError(f"crash of {self.node!r}: restart_at must be after at")


@dataclass(frozen=True, slots=True)
class RandomCrashes:
    """
    `count` distinct nodes crash at uniform times in `window` (by default,
    while jobs arrive), each restarting `downtime` later if given.
    """

    count: int
    downtime: float | None = None
    window: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("random_crashes: count must be nonnegative")
        if self.downtime is not None and self.downtime <= 0:
            raise ValueError("random_crashes: downtime must be positive")
        if self.window is not None and not 0 <= self.window[0] <= self.window[1]:
            raise ValueError("random_crashes: window must be [start, end] with 0 <= start <= end")


@dataclass(frozen=True, slots=True)
class Partition:
    """
    From `at` until `heal_at` (or for good), messages between `nodes` and
    every other node are lost, in both directions.
    """

    nodes: frozenset[NodeId]
    at: float
    heal_at: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.nodes, str):
            raise TypeError("nodes must be a collection, not a string")
        object.__setattr__(self, "nodes", frozenset(self.nodes))
        _floats(self, "at", "heal_at")
        if not self.nodes:
            raise ValueError("a partition needs at least one node")
        if self.at < 0:
            raise ValueError("partition: at must be nonnegative")
        if self.heal_at is not None and self.heal_at <= self.at:
            raise ValueError("partition: heal_at must be after at")

    def active(self, time: float) -> bool:
        return self.at <= time and (self.heal_at is None or time < self.heal_at)

    def cuts(self, time: float, sender: NodeId, receiver: NodeId) -> bool:
        return self.active(time) and (sender in self.nodes) != (receiver in self.nodes)


@dataclass(frozen=True, slots=True)
class MessageLoss:
    """
    Drop network messages sent during `[start, end)` (`end` None: forever).

    Each matching message is dropped with `probability`. Leaving `kinds`
    (message type names, such as "StatusPush"), `senders` or `receivers`
    as None matches every value. Messages a node sends to itself are never
    dropped.
    """

    start: float = 0.0
    end: float | None = None
    probability: float = 1.0
    kinds: frozenset[str] | None = None
    senders: frozenset[NodeId] | None = None
    receivers: frozenset[NodeId] | None = None

    def __post_init__(self) -> None:
        _floats(self, "start", "end", "probability")
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be between 0 and 1")
        if self.end is not None and self.end <= self.start:
            raise ValueError("a loss's end must be after its start")
        for name in ("kinds", "senders", "receivers"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, str):
                    raise TypeError(f"{name} must be a collection, not a string")
                object.__setattr__(self, name, frozenset(value))

    def matches(self, time: float, sender: NodeId, receiver: NodeId, kind: str) -> bool:
        return (
            self.start <= time
            and (self.end is None or time < self.end)
            and (self.kinds is None or kind in self.kinds)
            and (self.senders is None or sender in self.senders)
            and (self.receivers is None or receiver in self.receivers)
        )


def is_dropped(
    losses: Iterable[MessageLoss],
    partitions: Iterable[Partition],
    rng: Random,
    time: float,
    sender: NodeId,
    receiver: NodeId,
    kind: str,
) -> bool:
    """
    Whether a network message is lost: cut by a partition, or dropped by a
    loss. Only losses with probability below 1 draw from `rng`.
    """
    if any(partition.cuts(time, sender, receiver) for partition in partitions):
        return True
    return any(
        loss.matches(time, sender, receiver, kind)
        and (loss.probability >= 1 or rng.random() < loss.probability)
        for loss in losses
    )


@dataclass(frozen=True, slots=True)
class Scenario:
    """
    Faults to inject. Crashes are in time order, and crashes of one node
    never overlap: each starts after the previous one's restart. Partitions
    that share a node never overlap in time.
    """

    crashes: tuple[Crash, ...] = ()
    random_crashes: RandomCrashes | None = None
    partitions: tuple[Partition, ...] = ()
    losses: tuple[MessageLoss, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.crashes, key=lambda c: c.at))
        object.__setattr__(self, "crashes", ordered)
        last: dict[NodeId, Crash] = {}
        for crash in ordered:
            previous = last.get(crash.node)
            if previous is not None and (
                previous.restart_at is None or crash.at <= previous.restart_at
            ):
                raise ValueError(f"{crash.node!r} crashes at {crash.at} while already down")
            last[crash.node] = crash
        partitions = tuple(sorted(self.partitions, key=lambda p: p.at))
        object.__setattr__(self, "partitions", partitions)
        for i, first in enumerate(partitions):
            for second in partitions[i + 1 :]:
                shared = first.nodes & second.nodes
                if shared and first.active(second.at):
                    raise ValueError(f"{sorted(shared)} are in two partitions at {second.at}")
        object.__setattr__(self, "losses", tuple(self.losses))

    @property
    def crashed(self) -> set[NodeId]:
        """
        Every node crashed by name.
        """
        return {crash.node for crash in self.crashes}

    @property
    def partitioned(self) -> set[NodeId]:
        """
        Every node that is ever cut off.
        """
        return {node for partition in self.partitions for node in partition.nodes}

    @property
    def nodes(self) -> set[NodeId]:
        """
        Every node named anywhere in the scenario.
        """
        named = self.crashed | self.partitioned
        for loss in self.losses:
            named |= (loss.senders or frozenset()) | (loss.receivers or frozenset())
        return named

    def down_at_end(self) -> set[NodeId]:
        """
        Nodes whose last crash has no restart.
        """
        last = {crash.node: crash for crash in self.crashes}
        return {node for node, crash in last.items() if crash.restart_at is None}

    def resolve(self, seed: int, nodes: Iterable[NodeId], horizon: float) -> Scenario:
        """
        This scenario with `random_crashes` replaced by the crashes `seed`
        picks among `nodes` (minus nodes crashed by name). `horizon` is the
        last job arrival, the end of the default window.
        """
        if self.random_crashes is None:
            return self
        spec = self.random_crashes
        pool = sorted(set(nodes) - self.crashed)
        if spec.count > len(pool):
            raise ValueError(f"random_crashes: {spec.count} crashes but {len(pool)} nodes")
        start, end = spec.window if spec.window is not None else (0.0, horizon)
        rng = Random(f"crashes:{seed}")
        picked = []
        for node in rng.sample(pool, spec.count):
            at = rng.uniform(start, end)
            restart_at = at + spec.downtime if spec.downtime is not None else None
            picked.append(Crash(node, at, restart_at))
        return replace(self, crashes=(*self.crashes, *picked), random_crashes=None)


def _fields(entry: Any, what: str, required: set[str], optional: set[str]) -> Mapping:
    if not isinstance(entry, Mapping):
        raise TypeError(f"{what} must be an object: {entry!r}")
    if unknown := set(entry) - required - optional:
        raise ValueError(f"unknown {what} fields: {sorted(unknown)}")
    if missing := required - set(entry):
        raise ValueError(f"{what} needs {', '.join(sorted(missing))}: {dict(entry)}")
    return entry


def _time(value: Any) -> float | None:
    return None if value is None else float(value)


def _names(value: Any) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError(f"expected a list of names, not {value!r}")
    return frozenset(str(name) for name in value)


def loss_from_data(data: Any) -> MessageLoss:
    entry = _fields(
        data, "loss", set(), {"start", "end", "probability", "kinds", "senders", "receivers"}
    )
    return MessageLoss(
        start=float(entry.get("start", 0.0)),
        end=_time(entry.get("end")),
        probability=float(entry.get("probability", 1.0)),
        kinds=_names(entry.get("kinds")),
        senders=_names(entry.get("senders")),
        receivers=_names(entry.get("receivers")),
    )


def loss_to_data(loss: MessageLoss) -> dict[str, Any]:
    data: dict[str, Any] = {"start": loss.start, "probability": loss.probability}
    if loss.end is not None:
        data["end"] = loss.end
    for name in ("kinds", "senders", "receivers"):
        value = getattr(loss, name)
        if value is not None:
            data[name] = sorted(value)
    return data


def scenario_from_data(data: Any) -> Scenario:
    """
    Parse a scenario from decoded JSON, rejecting unknown fields.
    """
    data = _fields(data, "scenario", set(), {"crashes", "random_crashes", "partitions", "losses"})
    crashes = []
    for entry in data.get("crashes", []):
        entry = _fields(entry, "crash", {"node", "at"}, {"restart_at"})
        crashes.append(
            Crash(str(entry["node"]), float(entry["at"]), _time(entry.get("restart_at")))
        )
    random_crashes = None
    if (entry := data.get("random_crashes")) is not None:
        entry = _fields(entry, "random_crashes", {"count"}, {"downtime", "window"})
        window = entry.get("window")
        if window is not None and len(window) != 2:
            raise ValueError("random_crashes: window must be [start, end]")
        random_crashes = RandomCrashes(
            int(entry["count"]),
            _time(entry.get("downtime")),
            None if window is None else (float(window[0]), float(window[1])),
        )
    partitions = []
    for entry in data.get("partitions", []):
        entry = _fields(entry, "partition", {"nodes", "at"}, {"heal_at"})
        nodes = _names(entry["nodes"])
        assert nodes is not None
        partitions.append(Partition(nodes, float(entry["at"]), _time(entry.get("heal_at"))))
    losses = [loss_from_data(entry) for entry in data.get("losses", [])]
    return Scenario(tuple(crashes), random_crashes, tuple(partitions), tuple(losses))


def load_scenario(path: str | Path) -> Scenario:
    return scenario_from_data(json.loads(Path(path).read_text()))


def scenario_argument(path: str) -> Scenario:
    """
    `load_scenario` as an argparse `type`, so bad files are usage errors.
    """
    try:
        return load_scenario(path)
    except (OSError, ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from None
