# TORCH

TORCH studies decentralized scheduling of GPU jobs on heterogeneous
clusters, and what different amounts of consistency cost. Every scheduling
design is written once, as message-driven state machines in the
`torchsched` package. The same nodes run in a deterministic simulator for
experiments and as one HTTP process per node for deployments.

All compute is simulated: jobs run for their given runtime, so nothing
needs physical GPUs.

## Designs

| Design | How schedulers learn about free capacity |
| --- | --- |
| `centralized` | Workers report to one coordinator, which schedules every job from a complete view. |
| `gossip` | Every interval, each node sends its own status plus every fresh record it knows to one random neighbor. Nodes keep records only about their neighbors. On a full mesh this is "global gossip", OpenTela-style. |
| `periodic` | Every interval, each node sends only its own status to every neighbor (Navigator-style bounded staleness). |
| `sampling` | No standing view. For each placement round, a scheduler probes `sample_size - 1` random neighbors (Sparrow-style). |

The peer-to-peer designs (all but centralized) run a worker and a scheduler
on every node, with neighborhoods from a ring with seeded shortcuts or a
full mesh. In all of them:

- Workers own their GPUs. A reservation is granted or rejected atomically
  at the worker, so stale views can cause rejections but never overbooking.
  A rejection carries the worker's current status, which corrects the
  scheduler's view at once.
- Records older than the staleness limit count as unknown.
- A job that nothing in a scheduler's view can run is forwarded around the
  ring, and waiting jobs retry on an interval.
- Placement is first fit, either strict FIFO or with backfill.

## Layout

| Path | Contents |
| --- | --- |
| `torchsched/types/` | Models (`NodeSpec`, `GpuGroup`, `Job`, `Status`), messages, events, actions, metrics |
| `torchsched/interfaces/` | Component roles and policy protocols |
| `torchsched/node.py` | `Node`: routes events to its worker, membership, dissemination and scheduler components |
| `torchsched/worker.py`, `scheduler.py` | Two-phase reserve and start at workers; queued placement rounds at schedulers |
| `torchsched/dissemination/` | `centralized`, `gossip`, `periodic`, `sampling` |
| `torchsched/placement/`, `forwarding/`, `membership/`, `topology.py` | First fit, ring forwarding, static membership, neighborhoods |
| `torchsched/clusters.py` | Builds every node of a design, in code or from a JSON config |
| `torchsched/runtime/` | `sim.py` (deterministic simulator) and `http.py` (FastAPI + httpx) |
| `experiments/` | Paired comparisons with recorded legacy results, and performance history |
| `deploy/` | Docker image, cluster configs, Compose files, and an end-to-end demo |

The core has no I/O. A node takes one event at a time (`Node.handle(event,
now)`) and returns actions (send, set timer, execute, observe) for a
runtime to carry out. A test enforces that only runtimes import I/O
libraries.

## Setup

Python 3.11 or newer:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## Simulate

```python
from torchsched import clusters, topology
from torchsched.forwarding.ring import RingForwarding
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.types.metrics import JobCompleted
from torchsched.types.models import GpuGroup, NodeSpec
from torchsched.workloads import poisson_workload

specs = [NodeSpec(f"l40-{i}", [GpuGroup("L40", 2, 48)]) for i in range(1, 4)] + [
    NodeSpec(f"a100-{i}", [GpuGroup("A100", 2, 80)]) for i in range(1, 4)
]
ids = [spec.node_id for spec in specs]
nodes = clusters.gossip(
    specs,
    neighborhoods=topology.ring_with_shortcuts(ids, radius=1, seed=0),
    placement=FirstFit(backfill=True),
    forwarding=RingForwarding(ids),
    interval=0.5,
    max_staleness=3.0,
    retry_interval=0.25,
)
sim = Simulator(nodes, latency=0.02)
for i, submission in enumerate(poisson_workload(seed=0, count=50, arrival_rate=3.0)):
    sim.submit(submission.job, at=submission.at, node_id=ids[i % len(ids)])
result = sim.run(until=10_000, until_jobs_done=True)

done = sum(isinstance(o.metric, JobCompleted) for o in result.observations)
print(f"{done} jobs done by t={result.end_time:.1f} s, {sum(result.messages.values())} messages")
```

`result` also holds every observed metric with its time and node, every
execution (worker, GPUs, start and end), and message counts and encoded
bytes by message type. Runs are deterministic for a given seed.

To compare a design with the legacy simulator's results on the same 100
workloads, see [experiments/README.md](experiments/README.md):

```sh
python -m experiments.compare_with_legacy --design periodic --seeds 100
```

## Deploy

`deploy/` has one cluster config and one Compose file for every design;
`TORCH_DESIGN` picks the design. Every node is its own container, built
from one image. Published ports bind to 127.0.0.1 only, because the HTTP API
has no authentication.

```sh
export TORCH_DESIGN=gossip
docker compose -f deploy/compose.yaml up --build -d
python -m deploy.demo --cluster deploy/cluster.json --design $TORCH_DESIGN
docker compose -f deploy/compose.yaml down
```

The demo submits a seeded workload, waits for every job to complete,
checks that every GPU is released and no message failed, and prints
per-job placements and the traffic. Set `TORCH_DESIGN` to `periodic` or
`sampling` instead, or for `centralized` also set
`COMPOSE_PROFILES=centralized` so the coordinator starts. Every design
runs on the same four workers.

Each node serves `GET /health`, `POST /messages` (node-to-node),
`POST /jobs`, `GET /status`, `GET /observations` and `GET /metrics`. To run
a node outside Docker:

```sh
python -m torchsched.runtime.http --cluster deploy/cluster.json --design gossip --node l40-1 --port 8000
```

## Status

See [ROADMAP.md](ROADMAP.md) for what is done and what is next.

The earlier SimPy simulator (`simulator.py`, `distributed.py`,
`experiment.py`) and HTTP prototype (`torch_net/`) were replaced by
`torchsched` and removed after commit `0abb0e6`. Their results are recorded
in `experiments/data/` and `tests/data/`, and new code is checked against
them.
