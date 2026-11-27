"""
Run one node as an HTTP service.

Each process hosts one `Node`. Messages travel as `torchsched.codec` bytes
in `POST /messages`, with the sender in the `X-Torch-Sender` header. A
single task feeds events to the node one at a time, so the node's
one-event-at-a-time guarantee holds just as in the simulator.

Run a node from a cluster config (see `torchsched.clusters.from_config`):

    python -m torchsched.runtime.http --cluster deploy/cluster.json --design centralized --node coordinator

The config's `"addresses"` maps every node id to its base URL. `--design`
overrides the config's `"design"`, so one file can hold every design's
section.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

from torchsched.clusters import from_config
from torchsched.codec import decode, encode, from_data, to_data
from torchsched.node import Node
from torchsched.runtime.sim import Observation
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
from torchsched.types.messages import Message
from torchsched.types.models import Job, NodeId, ReservationId

log = logging.getLogger(__name__)

SENDER_HEADER = "X-Torch-Sender"


class HttpRuntime:
    """
    Drives one node with real time, real timers, and HTTP messages.

    `addresses` maps node ids to base URLs. A message to an unknown node, or
    one whose POST fails, is dropped and counted in `send_failures`, like a
    message lost on the network. `execution_scale` multiplies every job's
    runtime, so demos can run simulated jobs faster than real time.
    """

    def __init__(
        self,
        node: Node,
        addresses: Mapping[NodeId, str],
        *,
        execution_scale: float = 1.0,
        client: httpx.AsyncClient | None = None,
        timeout: float = 5.0,
    ):
        if execution_scale <= 0:
            raise ValueError("execution_scale must be positive")
        self.node = node
        self.addresses = {node_id: url.rstrip("/") for node_id, url in addresses.items()}
        self.execution_scale = execution_scale
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self.observations: list[Observation] = []
        self.messages_sent: Counter[str] = Counter()
        self.bytes_sent: Counter[str] = Counter()
        self.messages_received: Counter[str] = Counter()
        self.send_failures: Counter[str] = Counter()
        self._queue: asyncio.Queue[Event] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started_at = 0.0
        self._timers: dict[TimerKey, asyncio.TimerHandle] = {}
        self._tasks: set[asyncio.Task] = set()
        self._consumer: asyncio.Task | None = None

    @property
    def node_id(self) -> NodeId:
        return self.node.node_id

    @property
    def started(self) -> bool:
        return self._queue is not None

    def now(self) -> float:
        """
        Wall-clock time in seconds.

        It must be shared across nodes, not per process: staleness compares
        a record's `observed_at`, stamped by another node, with this node's
        `now`. Like the design, this assumes loosely synchronized clocks.
        Timers use the event loop's monotonic clock instead.
        """
        return time.time()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._started_at = self.now()
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        self._queue = asyncio.Queue()
        self._consumer = asyncio.create_task(self._consume())
        self.deliver(Boot())

    async def close(self) -> None:
        for handle in self._timers.values():
            handle.cancel()
        self._timers.clear()
        tasks = [*self._tasks, *([self._consumer] if self._consumer else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    def deliver(self, event: Event) -> None:
        """
        Queue an event for the node (from the network, a timer, or a client).
        """
        if self._queue is None:
            raise RuntimeError("runtime not started")
        self._queue.put_nowait(event)

    async def _consume(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                actions = self.node.handle(event, self.now())
            except Exception:  # keep serving; one bad event must not stop the node
                log.exception("node %s failed to handle %r", self.node_id, event)
                continue
            self._apply(actions)

    def _apply(self, actions: list[Action]) -> None:
        for action in actions:
            match action:
                case Send(to, msg):
                    if to == self.node_id:
                        self.deliver(Received(self.node_id, msg))
                    else:
                        self._spawn(self._send(to, msg))
                case SetTimer(key, delay):
                    self._cancel_timer(key)
                    assert self._loop is not None
                    self._timers[key] = self._loop.call_later(delay, self._fire, key)
                case CancelTimer(key):
                    self._cancel_timer(key)
                case Execute(reservation_id, job, _, gpu_type):
                    duration = job.runtime_by_gpu[gpu_type] * self.execution_scale
                    self._spawn(self._execute(reservation_id, duration))
                case Observe(metric):
                    self.observations.append(Observation(self.now(), self.node_id, metric))

    def _spawn(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _fire(self, key: TimerKey) -> None:
        self._timers.pop(key, None)
        self.deliver(TimerFired(key))

    def _cancel_timer(self, key: TimerKey) -> None:
        handle = self._timers.pop(key, None)
        if handle is not None:
            handle.cancel()

    async def _execute(self, reservation_id: ReservationId, duration: float) -> None:
        await asyncio.sleep(duration)
        self.deliver(ExecutionFinished(reservation_id, True))

    async def _send(self, to: NodeId, msg: Message) -> None:
        kind = type(msg).__name__
        url = self.addresses.get(to)
        if url is None:
            self.send_failures[kind] += 1
            log.warning("node %s: no address for %s; dropped %s", self.node_id, to, kind)
            return
        body = encode(msg)
        self.messages_sent[kind] += 1
        self.bytes_sent[kind] += len(body)
        assert self._client is not None
        try:
            response = await self._client.post(
                f"{url}/messages",
                content=body,
                headers={SENDER_HEADER: self.node_id, "Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            self.send_failures[kind] += 1
            log.warning("node %s: sending %s to %s failed: %s", self.node_id, kind, to, error)

    def metrics(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "uptime": self.now() - self._started_at,
            "messages_sent": dict(self.messages_sent),
            "bytes_sent": dict(self.bytes_sent),
            "messages_received": dict(self.messages_received),
            "send_failures": dict(self.send_failures),
            "observations": len(self.observations),
        }


def create_app(runtime: HttpRuntime) -> FastAPI:
    """
    The HTTP API for one node. The app's lifespan starts and stops `runtime`.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title=f"TORCH node {runtime.node_id}", lifespan=lifespan)

    def require_started() -> None:
        if not runtime.started:
            raise HTTPException(503, "node is starting")

    @app.get("/health")
    async def health() -> dict:
        return {"node_id": runtime.node_id, "ok": runtime.started}

    @app.post("/messages", status_code=202)
    async def receive(request: Request) -> Response:
        require_started()
        sender = request.headers.get(SENDER_HEADER)
        if not sender:
            raise HTTPException(400, f"missing {SENDER_HEADER} header")
        try:
            msg = decode(await request.body())
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        runtime.messages_received[type(msg).__name__] += 1
        runtime.deliver(Received(sender, msg))
        return Response(status_code=202)

    @app.post("/jobs", status_code=202)
    async def submit(request: Request) -> dict:
        """
        Submit a job, given as the fields of `torchsched.types.models.Job`.
        """
        require_started()
        if runtime.node.scheduler is None:
            raise HTTPException(409, f"{runtime.node_id} does not schedule jobs")
        try:
            job = from_data(Job, await request.json())
        except (TypeError, ValueError) as error:
            raise HTTPException(422, f"invalid job: {error}") from None
        runtime.deliver(JobSubmitted(job))
        return {"job_id": job.job_id, "node_id": runtime.node_id}

    @app.get("/status")
    async def status() -> dict:
        """
        This node's own worker status, or null for a node without GPUs.
        """
        require_started()
        local = runtime.node.ctx.local_status(runtime.now())
        return {"node_id": runtime.node_id, "status": to_data(local) if local else None}

    @app.get("/observations")
    async def observations(since: int = 0) -> dict:
        """
        Metrics this node recorded, starting at index `since`.
        """
        entries = runtime.observations[max(since, 0) :]
        return {
            "next": since + len(entries),
            "observations": [
                {
                    "time": o.time,
                    "node_id": o.node_id,
                    "type": type(o.metric).__name__,
                    "metric": to_data(o.metric),
                }
                for o in entries
            ],
        }

    @app.get("/metrics")
    async def metrics() -> dict:
        require_started()
        return runtime.metrics()

    return app


def load_runtime(config: Mapping[str, Any], node_id: NodeId, **options) -> HttpRuntime:
    """
    Build `node_id`'s runtime from a cluster config with an `"addresses"` map.
    """
    nodes = {node.node_id: node for node in from_config(config)}
    if node_id not in nodes:
        raise ValueError(f"node {node_id!r} is not in the cluster ({sorted(nodes)})")
    addresses = config.get("addresses", {})
    missing = set(nodes) - set(addresses)
    if missing:
        raise ValueError(f"no address for {sorted(missing)}")
    return HttpRuntime(nodes[node_id], addresses, **options)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run one TORCH node over HTTP.")
    parser.add_argument("--cluster", required=True, help="cluster config JSON file")
    parser.add_argument("--node", required=True, help="this process's node id")
    parser.add_argument("--design", help="override the config's design")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--execution-scale",
        type=float,
        default=1.0,
        help="multiply job runtimes by this (for example 0.1 for faster demos)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = json.loads(Path(args.cluster).read_text())
    if args.design:
        config["design"] = args.design
    runtime = load_runtime(config, args.node, execution_scale=args.execution_scale)
    uvicorn.run(create_app(runtime), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
