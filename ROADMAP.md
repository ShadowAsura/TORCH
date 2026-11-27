# TORCH roadmap

## Done

- **One implementation, two runtimes.** Every design is written once as
  sans-I/O state machines in `torchsched`. It runs in a deterministic
  simulator and as one HTTP process per node, with a compact JSON message
  encoding.
- **Workers.** GPU groups (a node can mix GPU types). Atomic two-phase
  reservations (reserve, then start) with leases, and idempotent grants.
  Versioned status, ordered by `(incarnation, version)`.
- **Schedulers.** Queued placement rounds with strict FIFO or backfill
  first fit. Ring forwarding of jobs that don't fit and retries on an
  interval. A ledger of assumed placements that prevents double booking
  between status updates. Rejections carry the worker's current status.
- **Designs.** Centralized, gossip (on a ring or a full mesh), periodic,
  and sampling.
- **Checked against the legacy code.** The centralized baseline matches
  legacy job by job. The workload generator and ring layouts match exactly.
  For every peer-to-peer design, backfill placement matches legacy
  scheduling delay over 100 paired seeds. See
  [experiments/README.md](experiments/README.md).
- **Deployment.** A Docker image, one cluster config and Compose file for
  every design, and an end-to-end demo that checks completions, released
  GPUs and failed messages.
- **Measurements.** Per-type message counts and encoded bytes, rejections,
  scheduling delay, and an execution log for utilization and overbooking
  checks.

## Next

1. **Failure handling.** The legacy simulator could crash and recover
   workers and drop status messages. That has not been ported yet; the
   gaps are marked with TODOs in `scheduler.py`. The work:
   - detecting crashed workers through incarnation numbers
   - timeouts for lost reserve replies
   - fault injection in the simulator
   - the three legacy demonstration scenarios: a stale-view conflict, a
     crash with restart elsewhere, and status loss bridged by forwarding
2. **SWIM membership.** It replaces static membership; the hooks are marked
   with TODOs in the membership role and messages.
3. **An experiment runner and parameter sweeps.** This replaces legacy
   `experiment.py --policy all`. It would run every design on the same
   seeded workloads, sweep staleness, interval, sample size and radius, and
   report delay, completion time, utilization, throughput, rejections and
   traffic with charts.
4. **Lower traffic.** `ForwardJob` lapping is about 70% of bytes and full
   `NodeSpec`s in status records about 25% (see the experiments README).
5. **Placement policies from the proposal.** Sia- and Beehive-style
   heterogeneity-aware placement, behind the `PlacementPolicy` interface.
6. **Cloud deployment and real GPU work** for the proposal's
   real-deployment validation. Keep simulated execution as the test
   fixture.
7. **Course deliverables.** The IEEE report with UML diagrams and
   demonstration snapshots, the recorded presentation and slides, and the
   division of work. Code and presentation are due December 3, the report
   December 4.
