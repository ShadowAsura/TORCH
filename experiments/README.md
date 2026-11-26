# Experiments

## `compare_with_legacy.py`

Runs the new full-mesh gossip (strict FIFO and backfill placement, ring
forwarding) on seeded workloads and compares it with the legacy simulator's
global gossip on the same workloads. It reports per-design means and
per-seed paired differences.

```sh
python -m experiments.compare_with_legacy --seeds 100 --out results.json
```

No legacy code runs here. Legacy results for seeds 0–99 are stored in
[`data/legacy_global_gossip.json`](data/legacy_global_gossip.json), together
with the settings and the commit they came from. The script refuses
settings that differ from the recorded ones. To record a different setup,
run `python -m experiments.record_legacy_baseline` with the same arguments.
That needs the legacy modules, so after they are deleted, check out the
recorded commit first.

Runs are deterministic: the same code and arguments give identical numbers,
so any change in the table below comes from a code change.

### Performance history

Use this to catch regressions: rerun the command above after a change and
compare with the latest row. Add a row whenever a commit changes scheduling
behavior.

Setup (the script's defaults): seeds 0–99, 100 jobs per seed, arrival rate
5 jobs/s, 6 L40 + 6 A100 nodes with 2 GPUs each, 20 ms network latency,
gossip every 0.5 s, 3 s staleness limit, 0.25 s retry interval. Python 3.12.

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
