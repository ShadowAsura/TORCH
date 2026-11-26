# Experiments

## `compare_with_legacy.py`

Runs the legacy simulator's global gossip and the new full-mesh gossip
(strict FIFO and backfill placement, ring forwarding) on the same seeded
workloads, and reports per-design means and per-seed paired differences.

```sh
python -m experiments.compare_with_legacy --seeds 100 --out results.json
```

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
in seconds, with a 95% confidence interval.

| Version | Placement | Delay | 95% CI | Rejected | Messages |
| --- | --- | --- | --- | --- | --- |
| Legacy (`distributed.py`, global gossip) | per job, no queue | 4.753 | [4.536, 4.969] | 760 | 13,753 |
| `807785d` gossip + ring forwarding | strict FIFO | 4.895 | [4.677, 5.114] | 1,072 | 13,445 |
| | backfill | 4.757 | [4.551, 4.963] | 981 | 13,085 |
| _(next commit)_ rejections carry worker status | strict FIFO | 4.886 | [4.631, 5.141] | 60 | 12,921 |
| | backfill | 4.638 | [4.411, 4.865] | 55 | 12,387 |

Paired differences against legacy (same workload per seed):

| Version | Placement | Delay vs legacy | 95% CI | Slower on |
| --- | --- | --- | --- | --- |
| `807785d` | strict FIFO | +0.142 s (+3.0%) | [+0.055, +0.229] | 60/100 seeds |
| | backfill | +0.004 s (+0.1%) | [−0.084, +0.092] | 50/100 seeds |
| _(next commit)_ | strict FIFO | +0.133 s (+2.8%) | [+0.016, +0.250] | 56/100 seeds |
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
