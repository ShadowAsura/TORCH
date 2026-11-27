# Experiments

## `sweep.py`

Runs every scheduling design on the same seeded workloads and writes a
self-contained experiment report. This is the replacement for the legacy
`experiment.py --policy all`: use it for new measurements, and use
`compare_with_legacy.py` only when checking against the recorded legacy
baseline.

```sh
python -m experiments.sweep --out results/baseline
python -m experiments.sweep --out results/sweeps --seeds 100 \
    --sweep max-staleness=1,3,6 --sweep interval=0.25,0.5,1 \
    --sweep sample-size=2,3,4 --sweep radius=1,2,3
```

The default compares centralized, global gossip, ring gossip, periodic and
sampling with both strict FIFO and backfill placement. Every case uses the
same workload for a given seed: centralized jobs enter through the
coordinator, while peer-to-peer jobs enter round-robin over the workers.
`--designs`, `--placement`, `--jobs`, `--arrival-rate`, cluster sizes,
network latency and protocol settings are configurable. Independent runs
use up to four worker processes by default; raise or lower that with
`--processes`.

Each `--sweep` is one factor at a time: only that parameter changes while
the rest stay at their baseline values. Parameters apply only where they
mean something: interval and staleness to gossip/periodic designs, sample
size to sampling, and radius to ring-topology designs. Repeat `--sweep` for
several parameters. Values equal to the baseline are not rerun.

The output directory contains:

| File | Contents |
| --- | --- |
| `results.json` | full configuration, every per-seed row, summaries and traffic by message type |
| `rows.csv` | one row per case and seed |
| `summary.csv` | means and 95% confidence intervals over seeds |
| `index.html` | report tables with the generated charts |
| `charts/` | dependency-free SVG bar charts for the baseline, traffic composition, and line charts for sweeps |

Metrics are mean scheduling delay (submission to first start), mean job
completion time (submission to first completion), makespan (first
submission to last completion), throughput (completions per makespan
second), GPU utilization (busy GPU-seconds over capacity GPU-seconds during
the makespan), rejected reservations, network messages and encoded traffic
bytes. Percentile columns add p95 delay and completion time. Runs are
deterministic: the same code and arguments produce the same metrics.

## `compare_with_legacy.py`

Runs a new design (strict FIFO and backfill placement, ring forwarding) on
seeded workloads and compares it with the legacy simulator's version of the
same design on the same workloads. It reports per-design means and per-seed
paired differences.

| `--design` | New code | Legacy |
| --- | --- | --- |
| `global-gossip` (default) | gossip on a full mesh | global gossip |
| `gossip` | gossip, ring with shortcuts | gossip, same ring |
| `periodic` | periodic, ring with shortcuts | periodic, same ring |
| `sampling` | sampling, ring with shortcuts | sampling, same ring |

```sh
python -m experiments.compare_with_legacy --seeds 100 --out results.json
python -m experiments.compare_with_legacy --design sampling --seeds 100
```

No legacy code runs here. Legacy results for seeds 0–99 are stored in
`data/legacy_<design>.json`, together with the settings and the commit
they came from. The script refuses settings that differ from the recorded
ones. The legacy code and its recorder, `experiments/record_legacy_baseline.py`,
were removed after `0abb0e6`. To record a different setup, check out
`0abb0e6` and run `python -m experiments.record_legacy_baseline` with the
same arguments.

Runs are deterministic: the same code and arguments give identical numbers,
so any change in the table below comes from a code change.

### Performance history

Use this to catch regressions: rerun the commands above after a change and
compare with the latest row. Add a row whenever a commit changes scheduling
behavior.

Setup (the script's defaults): seeds 0–99, 100 jobs per seed, arrival rate
5 jobs/s, 6 L40 + 6 A100 nodes with 2 GPUs each, 20 ms network latency,
gossip or periodic pushes every 0.5 s, 3 s staleness limit, 0.25 s retry
interval; ring radius 1 for every design except global gossip; for
sampling, sample size 2 (one random peer plus the node itself). Python
3.12.

#### Global gossip (`--design global-gossip`)

Means over 100 seeds. Delay is the mean scheduling delay (arrival to start)
in seconds, with a 95% confidence interval. Bytes are the encoded size of
all network messages (`torchsched.codec`). Bytes were first measured once
the encoder existed; the `77502f2` bytes come from the next commit, which
added the encoder without changing scheduling behavior.

| Version | Placement | Delay | 95% CI | Rejected | Messages | KB |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy (`distributed.py`, global gossip) | per job, no queue | 4.753 | [4.536, 4.969] | 760 | 13,753 | — |
| `807785d` gossip + ring forwarding | strict FIFO | 4.895 | [4.677, 5.114] | 1,072 | 13,445 | — |
| | backfill | 4.757 | [4.551, 4.963] | 981 | 13,085 | — |
| `77502f2` rejections carry worker status | strict FIFO | 4.886 | [4.631, 5.141] | 60 | 12,921 | 3,470 |
| | backfill | 4.638 | [4.411, 4.865] | 55 | 12,387 | 3,357 |
| `9f90a1d` acks for lost messages | strict FIFO | 4.886 | [4.631, 5.141] | 60 | 24,533 | 4,860 |
| | backfill | 4.638 | [4.411, 4.865] | 55 | 23,490 | 4,685 |
| `d70a125` crash detection, idle records stay fresh | strict FIFO | 4.954 | [4.699, 5.208] | 80 | 24,901 | 5,826 |
| | backfill | 4.639 | [4.414, 4.864] | 75 | 23,524 | 5,596 |

Paired differences against legacy (same workload per seed):

| Version | Placement | Delay vs legacy | 95% CI | Slower on |
| --- | --- | --- | --- | --- |
| `807785d` | strict FIFO | +0.142 s (+3.0%) | [+0.055, +0.229] | 60/100 seeds |
| | backfill | +0.004 s (+0.1%) | [−0.084, +0.092] | 50/100 seeds |
| `77502f2` | strict FIFO | +0.133 s (+2.8%) | [+0.016, +0.250] | 56/100 seeds |
| | backfill | −0.115 s (−2.4%) | [−0.197, −0.033] | 33/100 seeds |
| `d70a125` | strict FIFO | +0.201 s (+4.2%) | [+0.089, +0.313] | 62/100 seeds |
| | backfill | −0.114 s (−2.4%) | [−0.199, −0.029] | 37/100 seeds |

What changed between versions:

- **`807785d`**: with backfill, delay matches legacy. Strict FIFO is about
  3% slower because jobs wait behind a blocked queue head, which legacy
  never did since it placed every job independently.
- **Rejections carry worker status**: about 93% of rejections had been
  repeats, where a scheduler retried a worker using the same stale record
  that had just been rejected. Sending the worker's current status with
  each rejection corrects the view immediately. Rejections fell by about
  95%, messages by about 5%, and backfill became significantly faster than
  legacy.

Where the bytes go (seed 42, backfill, 12,888 messages, 3.5 MB):

| Message | Share of messages | Share of bytes | Average size |
| --- | --- | --- | --- |
| `ForwardJob` | 89% | 72% | 218 B |
| `StatusPush` | 9% | 26% | 770 B |
| everything else | 2% | 2% | 62–324 B |

Two likely improvements follow from this. Waiting jobs lap the ring every
retry interval, which on a full mesh adds traffic without helping
placement, because every scheduler already sees the same view. And each
status record carries its node's full `NodeSpec`.

#### Sampling (`--design sampling`)

| Version | Placement | Delay | 95% CI | Rejected | Messages | KB |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy (`distributed.py`, sampling) | per job, no queue | 4.850 | [4.627, 5.073] | 54 | 18,423 | — |
| `91dfae6` sampling ported | strict FIFO | 5.014 | [4.783, 5.245] | 6 | 9,664 | 1,827 |
| | backfill | 4.881 | [4.650, 5.111] | 7 | 9,544 | 1,800 |
| `9f90a1d` acks for lost messages | strict FIFO | 5.014 | [4.783, 5.245] | 6 | 15,569 | 2,526 |
| | backfill | 4.881 | [4.650, 5.111] | 7 | 15,312 | 2,483 |
| `d70a125` crash detection, idle records stay fresh | strict FIFO | 5.014 | [4.783, 5.245] | 6 | 15,750 | 2,538 |
| | backfill | 4.881 | [4.650, 5.111] | 7 | 15,519 | 2,496 |

| Version | Placement | Delay vs legacy | 95% CI | Slower on |
| --- | --- | --- | --- | --- |
| `91dfae6` | strict FIFO | +0.164 s (+3.4%) | [+0.079, +0.249] | 60/100 seeds |
| | backfill | +0.031 s (+0.6%) | [−0.059, +0.121] | 43/100 seeds |

With backfill, delay matches legacy, while messages are roughly halved and
rejections fall from 54 to 7. Legacy probed once per job at every
forwarding hop. The new scheduler probes once per placement round for all
of its waiting jobs, and the probe replies are fresh enough that few
reservations fail. As with gossip, strict FIFO costs about 3% in delay
because jobs wait behind a blocked queue head. Porting sampling did not
change any global-gossip number.

#### Gossip on a ring (`--design gossip`)

| Version | Placement | Delay | 95% CI | Rejected | Messages | KB |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy (`distributed.py`, gossip) | per job, no queue | 4.646 | [4.427, 4.865] | 208 | 12,935 | — |
| `0abb0e6` first recorded | strict FIFO | 4.820 | [4.571, 5.070] | 24 | 12,788 | 2,966 |
| | backfill | 4.583 | [4.354, 4.813] | 20 | 12,274 | 2,857 |
| `9f90a1d` acks for lost messages | strict FIFO | 4.820 | [4.571, 5.070] | 24 | 24,344 | 4,349 |
| | backfill | 4.583 | [4.354, 4.813] | 20 | 23,335 | 4,181 |
| `d70a125` crash detection, idle records stay fresh | strict FIFO | 4.869 | [4.614, 5.124] | 31 | 24,677 | 4,580 |
| | backfill | 4.584 | [4.358, 4.809] | 26 | 23,447 | 4,377 |

| Version | Placement | Delay vs legacy | 95% CI | Slower on |
| --- | --- | --- | --- | --- |
| `0abb0e6` | strict FIFO | +0.174 s (+3.7%) | [+0.078, +0.269] | 64/100 seeds |
| | backfill | −0.063 s (−1.4%) | [−0.152, +0.027] | 42/100 seeds |
| `d70a125` | strict FIFO | +0.223 s (+4.8%) | [+0.110, +0.335] | 62/100 seeds |
| | backfill | −0.063 s (−1.3%) | [−0.146, +0.021] | 44/100 seeds |

Ring gossip has been in the new code since `807785d`, but was first
compared with legacy here. The pattern matches global gossip: backfill is
as fast as legacy, and strict FIFO costs a few percent.

#### Periodic (`--design periodic`)

| Version | Placement | Delay | 95% CI | Rejected | Messages | KB |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy (`distributed.py`, periodic) | per job, no queue | 4.649 | [4.425, 4.873] | 154 | 14,879 | — |
| `0abb0e6` periodic ported | strict FIFO | 4.918 | [4.652, 5.183] | 41 | 15,062 | 3,174 |
| | backfill | 4.594 | [4.362, 4.827] | 30 | 14,329 | 3,021 |
| `9f90a1d` acks for lost messages | strict FIFO | 4.918 | [4.652, 5.183] | 41 | 26,814 | 4,580 |
| | backfill | 4.594 | [4.362, 4.827] | 30 | 25,405 | 4,346 |
| `d70a125` crash detection, idle records stay fresh | strict FIFO | 4.888 | [4.623, 5.152] | 44 | 26,844 | 4,566 |
| | backfill | 4.552 | [4.325, 4.780] | 35 | 25,368 | 4,324 |

| Version | Placement | Delay vs legacy | 95% CI | Slower on |
| --- | --- | --- | --- | --- |
| `0abb0e6` | strict FIFO | +0.269 s (+5.8%) | [+0.155, +0.382] | 63/100 seeds |
| | backfill | −0.055 s (−1.2%) | [−0.135, +0.026] | 38/100 seeds |
| `d70a125` | strict FIFO | +0.239 s (+5.1%) | [+0.128, +0.350] | 65/100 seeds |
| | backfill | −0.097 s (−2.1%) | [−0.174, −0.019] | 43/100 seeds |

With backfill, delay matches legacy. Strict FIFO costs more here (about
6%) than in the other designs. Porting periodic did not change any
global-gossip or sampling number.

#### Acknowledgements for lost messages (all designs)

The `9f90a1d` rows above add resending for lost messages. Every
`ForwardJob` is acknowledged with a `ForwardAck` and every `JobFinished`
with a `FinishedAck`, and `ForwardJob` gained a handoff id. Without
faults, nothing else changes: delay, its confidence interval, and
rejections are identical in every design, so the paired differences are
the same as in the previous rows. Messages rise by 60–90% and bytes by
35–40%, mostly from acknowledging `ForwardJob`, which was already most of
the traffic. Reducing forwarding laps (see above) would shrink both.

#### Crash detection and fresh idle records (all designs)

The `d70a125` rows add two things:

- **Heartbeats.** Workers send a `Heartbeat` to each scheduler with jobs
  running on them, about once a second. This adds roughly 200 messages
  per seed and changes no scheduling decision: sampling's delays and
  rejections are identical to the previous rows.
- **A gossip fix.** Gossip and periodic kept a received record only if
  its version was newer, so an idle node's later pushes (same version,
  later `observed_at`) were dropped, and the node fell out of every view
  `max_staleness` seconds after its last change. Records with the same
  version but a later `observed_at` now replace older ones, as in legacy.
  Views keep idle nodes, so status pushes carry more records (global
  gossip bytes +20%). Backfill delay is unchanged or better (periodic is
  now significantly faster than legacy). Strict FIFO is about 1% slower
  for both gossip designs, with more rejections; each change is well
  within its confidence interval, and the cause is not yet pinned down.

### Caveats

- Message counts are not directly comparable between legacy and the new
  code. Legacy counted its own message kinds, including probe and forward
  messages. The new code counts every message between nodes, including the
  two-phase `StartRequest`/`StartReply` and `JobFinished`. Compare messages
  between versions of the new code only.
- "Started" differs slightly. Legacy counted a job as started when its
  reservation was granted. The new code counts it at the `StartReply`, one
  round trip later for jobs placed on another node.
- The confidence intervals use a normal approximation and are not corrected
  for multiple comparisons.

## `faults.py`

Runs a design on the same seeded workloads with and without faults, so the
cost of the faults is paired per seed. Setup and arguments are as for
`compare_with_legacy.py`; placement is backfill unless `--strict`.

```sh
python -m experiments.faults --design gossip --scenario experiments/scenarios/loss-5pct.json
python -m experiments.faults --design gossip --scenario experiments/scenarios/two-crashes-5s.json
python -m experiments.faults --design sampling --scenario experiments/scenarios/two-crashes-permanent.json
python -m experiments.faults --design gossip --scenario deploy/scenarios/split-brain.json
```

The faults come from a scenario file (see `torchsched.scenarios` and the
main README): named crashes, random crashes, partitions and message loss.
Random crashes are picked per seed, uniformly among nodes and, by default,
in time while jobs arrive. `experiments/scenarios/` has one file per row of
the results below (`loss-5pct`, `two-crashes-5s`, `two-crashes-permanent`)
and `status-loss-5s`. They replace the earlier `--loss`, `--crashes`,
`--downtime` and `--status-loss-until` flags and draw exactly the same
faults, so they reproduce these numbers.

### Fault results

`acd6159`, 100 seeds, backfill. Columns are means per seed with
faults; delay and completion time are paired differences against the same
seed without faults, over jobs that completed in both runs. Lost jobs were
held by, or submitted to, a crashed scheduler. Wasted is GPU-seconds spent
on runs that were interrupted or duplicated. No job completed twice in any
run.

| Design | Faults | Completed | Lost | Interrupted | Abandoned | Wasted | Delay | Completion time | Messages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gossip | 5% loss | 100 | 0 | 0.1 | 0 | 1.7 | +0.52 s (+11%) | +0.53 s (+5%) | 23,447 → 11,995 |
| | 2 crashes, 5 s down | 92.1 | 7.9 | 0.8 | 23 | 11.3 | +0.62 s (+14%) | +0.73 s (+7%) | 23,447 → 17,185 |
| | 2 crashes, permanent | 83.1 | 12.8 | 0.8 | 8,533 | 11.1 | +10.2 s (+225%) | +10.4 s (+100%) | 23,447 → 132,105 |
| periodic | 5% loss | 100 | 0 | 0.4 | 0 | 3.3 | +0.58 s (+13%) | +0.60 s (+6%) | 25,368 → 13,712 |
| | 2 crashes, 5 s down | 92.2 | 7.8 | 1.1 | 24 | 11.2 | +0.68 s (+15%) | +0.80 s (+8%) | 25,368 → 19,259 |
| | 2 crashes, permanent | 83.3 | 12.8 | 1.2 | 8,147 | 11.2 | +9.8 s (+218%) | +10.1 s (+97%) | 25,368 → 282,091 |
| sampling | 5% loss | 100 | 0 | 0.2 | 0 | 1.7 | +1.44 s (+30%) | +1.45 s (+13%) | 15,519 → 6,244 |
| | 2 crashes, 5 s down | 91.9 | 8.1 | 0.9 | 7 | 11.4 | +0.56 s (+12%) | +0.67 s (+6%) | 15,519 → 11,195 |
| | 2 crashes, permanent | 83.3 | 12.7 | 0.8 | 7,721 | 10.8 | +15.6 s (+326%) | +15.8 s (+148%) | 15,519 → 42,228 |

What these show:

- **Lost messages slow jobs but lose none.** Each lost request costs a
  full 1 s timeout against 20 ms latency. Sampling suffers most: a lost
  probe stalls a whole placement round until the probe timeout. Messages
  fall, because waiting jobs lap the ring as fast as the network allows,
  and every lost `ForwardJob` or `ForwardAck` pauses that lap for a second.
- **Crashes lose the crashed schedulers' jobs** (about 4 per crash). Runs
  on crashed workers are detected and rerun elsewhere, costing about 11
  GPU-seconds per seed.
- **A node that stays down breaks the ring.** Ring forwarding keeps handing
  jobs to it and abandoning the handoff, so about 4% of jobs per seed
  never run (completed plus lost is under 100) and the run continues to
  the time limit. Fixing this needs membership to drop the dead node
  (SWIM, next on the roadmap).
