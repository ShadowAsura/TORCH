# TORCH implementation roadmap

TORCH explores placement of GPU jobs on heterogeneous nodes. The original
SimPy research model and the new HTTP prototype serve different purposes:
SimPy provides repeatable algorithm comparisons; the networked implementation
shows how a scheduler coordinates independent workers through real messages.
Both currently use synthetic compute durations rather than real GPU kernels.

## Implementation stages

| Stage | Scope | Evidence to collect |
| --- | --- | --- |
| Models and configuration | Job ID, whole-GPU demand, minimum VRAM, allowed GPU types, workload size; configurable worker capacities | Model validation and parameter tests |
| Worker resource accounting | Readable GPU state, atomic reservations, idempotent duplicate requests, capacity checks | Competing reservation and invalid request tests |
| Worker execution lifecycle | Reserve, start, run simulated work, finish and release GPUs | State transitions, cleanup, failure and concurrency tests |
| Scheduler | Submit and poll jobs over HTTP; query workers; rank feasible workers; reserve before starting | End-to-end API tests, negative and busy-worker cases |
| Discovery comparison | Centralized scans every worker; sampling checks a bounded randomized subset | Worker query counts, placement results, controlled seed |
| Network deployment | One scheduler and three workers, separate network hostnames, Compose health checks | `docker compose up --build -d`, `docker compose ps` |
| Network smoke test | Submit jobs through HTTP, poll until complete, confirm worker state and resource release | `python demo.py` under a running stack |
| Measurements | Scheduler attempts, worker queries, retries, scheduling delay, execution time, worker utilization | Scheduler and worker `GET /metrics` snapshots |

The current networking scope is the centralized and sampling policies. The
SimPy implementation also supports gossip, periodic updates, and global gossip;
those algorithms should not be described as deployed HTTP services until they
have their own network implementations and tests.

## Verification

From a fresh checkout:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python -m unittest -v
docker compose config --quiet
docker compose up --build -d
python demo.py --url http://localhost:8000
docker compose down
```

The smoke run should complete jobs under both policies, print actual worker
query counts (normally three for centralized and two for sampling when no
retries are needed), and restore each selected worker's baseline free-GPU
count. The demo fails on an unreachable service, failed job, inconsistent
metrics, or unreleased GPU reservation. Run against an otherwise idle stack.
The Compose workers expose localhost ports 8001–8003 for direct resource
checks; the scheduler uses internal service names over Docker's network.

## Next work for a stronger distributed evaluation

1. Extend the existing reservation-race, worker-crash/restart, and ambiguous
   response tests to sustained overload, partitions, and scheduler crashes.
   Define durable recovery behavior and ownership after scheduler process loss.
2. Add a scheduler queue, cancellation, admission control, and more explicit
   fairness. Measure workload latency under concurrent producers.
3. Implement the SimPy gossip and periodic discovery policies over actual
   network messages. Quantify message counts, message bytes, accuracy of
   worker views, and retry rates under varying staleness.
4. Add persistent state and membership/heartbeats so processes can recover
   across container restarts. Define durable job and reservation IDs.
5. If needed for hardware evaluation, replace simulated sleeps with real GPU
   workloads and add physical GPU discovery. Keep simulated mode as a
   deterministic test fixture.
6. Benchmark policy choices on identical seeded job traces; report
   scheduling latency, completion time, throughput, per-GPU utilization,
   failed reservations, and network overhead with reproducible charts.
7. Complete the course report, UML diagrams, demonstration captures,
   presentation, and team contribution summary.

## Reconstruction note

The project history was reconstructed from preserved source and a record of
planned changes. Commit titles and dates may document the intended progression;
they should not be treated as proof that the original, lost Git objects were
recovered byte for byte.
