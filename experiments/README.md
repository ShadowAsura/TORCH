# Experiments

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
| _(next commit)_ acks for lost messages | strict FIFO | 4.886 | [4.631, 5.141] | 60 | 24,533 | 4,860 |
| | backfill | 4.638 | [4.411, 4.865] | 55 | 23,490 | 4,685 |

Paired differences against legacy (same workload per seed):

| Version | Placement | Delay vs legacy | 95% CI | Slower on |
| --- | --- | --- | --- | --- |
| `807785d` | strict FIFO | +0.142 s (+3.0%) | [+0.055, +0.229] | 60/100 seeds |
| | backfill | +0.004 s (+0.1%) | [−0.084, +0.092] | 50/100 seeds |
| `77502f2` | strict FIFO | +0.133 s (+2.8%) | [+0.016, +0.250] | 56/100 seeds |
| | backfill | −0.115 s (−2.4%) | [−0.197, −0.033] | 33/100 seeds |

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
| _(next commit)_ acks for lost messages | strict FIFO | 5.014 | [4.783, 5.245] | 6 | 15,569 | 2,526 |
| | backfill | 4.881 | [4.650, 5.111] | 7 | 15,312 | 2,483 |

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
| _(next commit)_ acks for lost messages | strict FIFO | 4.820 | [4.571, 5.070] | 24 | 24,344 | 4,349 |
| | backfill | 4.583 | [4.354, 4.813] | 20 | 23,335 | 4,181 |

| Version | Placement | Delay vs legacy | 95% CI | Slower on |
| --- | --- | --- | --- | --- |
| `0abb0e6` | strict FIFO | +0.174 s (+3.7%) | [+0.078, +0.269] | 64/100 seeds |
| | backfill | −0.063 s (−1.4%) | [−0.152, +0.027] | 42/100 seeds |

Ring gossip has been in the new code since `807785d`, but was first
compared with legacy here. The pattern matches global gossip: backfill is
as fast as legacy, and strict FIFO costs a few percent.

#### Periodic (`--design periodic`)

| Version | Placement | Delay | 95% CI | Rejected | Messages | KB |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy (`distributed.py`, periodic) | per job, no queue | 4.649 | [4.425, 4.873] | 154 | 14,879 | — |
| `0abb0e6` periodic ported | strict FIFO | 4.918 | [4.652, 5.183] | 41 | 15,062 | 3,174 |
| | backfill | 4.594 | [4.362, 4.827] | 30 | 14,329 | 3,021 |
| _(next commit)_ acks for lost messages | strict FIFO | 4.918 | [4.652, 5.183] | 41 | 26,814 | 4,580 |
| | backfill | 4.594 | [4.362, 4.827] | 30 | 25,405 | 4,346 |

| Version | Placement | Delay vs legacy | 95% CI | Slower on |
| --- | --- | --- | --- | --- |
| `0abb0e6` | strict FIFO | +0.269 s (+5.8%) | [+0.155, +0.382] | 63/100 seeds |
| | backfill | −0.055 s (−1.2%) | [−0.135, +0.026] | 38/100 seeds |

With backfill, delay matches legacy. Strict FIFO costs more here (about
6%) than in the other designs. Porting periodic did not change any
global-gossip or sampling number.

#### Acknowledgements for lost messages (all designs)

The _(next commit)_ rows above add resending for lost messages. Every
`ForwardJob` is acknowledged with a `ForwardAck` and every `JobFinished`
with a `FinishedAck`, and `ForwardJob` gained a handoff id. Without
faults, nothing else changes: delay, its confidence interval, and
rejections are identical in every design, so the paired differences are
the same as in the previous rows. Messages rise by 60–90% and bytes by
35–40%, mostly from acknowledging `ForwardJob`, which was already most of
the traffic. Reducing forwarding laps (see above) would shrink both.

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
