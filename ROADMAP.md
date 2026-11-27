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
- **Failure handling.** Resending with acknowledgements for every request
  between nodes; worker heartbeats and incarnation checks to rerun jobs
  from crashed workers; restartable nodes (boot-time incarnations over
  HTTP); message loss and crash injection in the simulator; the three
  legacy demonstration scenarios (`tests/test_scenarios.py`); and
  `experiments/faults.py` to measure what faults cost. Faults are described
  by scenario files (crashes, random crashes, partitions, message loss) that
  run both in the simulator and in the Docker demo.

## Next

1. **SWIM membership.** It replaces static membership; the hooks are marked
   with TODOs in the membership role and messages. Until then, ring
   forwarding keeps handing jobs to a crashed node, and each handoff takes
   a full resend cycle (5 s) to be abandoned. A node that stays down
   blocks the ring for good: jobs that must pass it never run (about 4% of
   jobs with 2 permanent crashes; see the experiments README). SWIM's
   suspicion (indirect probes before declaring a node dead) should also
   replace the fixed heartbeat timeout, which is eager: about two missed
   heartbeats interrupt a run, so lost messages alone cause a few
   interruptions and duplicate runs (the 5% loss rows).
2. **Remaining failure gaps**:
   - jobs held by a crashed scheduler are lost; add a retry policy, for
     example client resubmission or replicated ownership
   - a duplicate run is not cancelled when the first run finishes
   - the centralized coordinator never learns about a worker whose first
     status report was lost; workers should re-report periodically
   - the sets of finished jobs and seen handoffs grow without bound
   - timeouts (1 s) are fixed and conservative next to a 20 ms network
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
   division of work.
