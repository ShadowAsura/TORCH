# TORCH simulator

TORCH is a CPU-only SimPy prototype for the heterogeneous GPU cluster
scheduling proposal. It compares a centralized FIFO baseline with distributed
schedulers on synthetic L40 and A100 workloads.

## Run the networked scheduler

TORCH also includes a separate FastAPI implementation with one HTTP scheduler
and three independent HTTP workers. Every worker owns its GPU reservations;
the scheduler discovers free capacity over the network and starts jobs through
the selected worker's API. Work runs as timed Python coroutines, so the demo
requires **no physical GPUs** and does not execute CUDA workloads. The SimPy
experiments documented below are unchanged.

With Docker Desktop running, from the repository root:

```sh
docker compose up --build -d
docker compose ps
```

Compose builds a scheduler image and a worker image, then starts three
independent worker containers with different GPU settings.

The scheduler and worker HTTP APIs have no authentication. For local safety,
Compose binds all published ports to **127.0.0.1**, so only processes on the
Docker host can reach them through the published ports by default. Containers
still communicate over the private Compose network using their service names.

| Service | Host address | Simulated capacity |
| --- | --- | --- |
| Scheduler | 127.0.0.1:8000 | Dispatches jobs to the three workers |
| `worker-l40` | 127.0.0.1:8001 | 2 L40 GPUs, 48 GB each, performance 1.0 |
| `worker-a100` | 127.0.0.1:8002 | 4 A100 GPUs, 80 GB each, performance 2.0 |
| `worker-t4` | 127.0.0.1:8003 | 2 T4 GPUs, 16 GB each, performance 0.5 |

Compose waits for the three worker health checks before starting the scheduler.
Run the **real HTTP smoke demo** from your host after installing the Python
dependencies (Python 3.11 or newer):

```sh
python -m pip install -r requirements.txt
python demo.py --url http://localhost:8000
```

The demo calls the scheduler and all three workers over HTTP, submits one
`centralized` job and one `sampling` job, polls each job through completion,
reads its actual worker record, checks that reserved GPUs have been released,
and prints the workers queried and service metrics. It exits nonzero if any
step fails. For CI, use `python demo.py --json`; for a single policy, use
`python demo.py --policy sampling`. If Compose ports are customized, the demo
reads the corresponding `TORCH_*_PORT` environment values or accepts
`--worker NAME=URL` for each worker.

Useful endpoints and manual requests:

```sh
curl http://localhost:8000/health
curl http://localhost:8000/workers
curl http://localhost:8000/metrics
curl http://localhost:8001/resources
curl http://localhost:8002/metrics
```

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/jobs `
  -ContentType application/json `
  -Body '{"job_id":"manual-001","gpu_count":1,"vram_gb":16,"compatible_gpu_types":["L40","A100"],"work_units":8,"policy":"sampling"}'
Invoke-RestMethod http://localhost:8000/jobs/manual-001
```

The scheduler exposes `GET /health`, `GET /workers`, `POST /jobs`,
`GET /jobs/{job_id}`, and `GET /metrics`. Each worker exposes `GET /health`,
`GET /resources`, `POST /reservations`,
`POST /jobs/{job_id}/start`, `GET /jobs/{job_id}`, and `GET /metrics`.
Job states include `queued`, `reserved`, `running`, `completed`, and
`failed`. The scheduler supports the `centralized` and `sampling`
discovery policies; both rely on workers to enforce atomic reservations.
The networked implementation currently has fewer policies than the SimPy model.

Set `TORCH_POLICY` to choose the default scheduler policy,
`TORCH_SAMPLE_SIZE` to adjust random sampling, `TORCH_SEED` for repeatable
selection, and `TORCH_SIM_SECONDS_PER_WORK_UNIT` to adjust simulated execution
speed. Worker capacities and host ports have `TORCH_L40_*`, `TORCH_A100_*`,
and `TORCH_T4_*` overrides in `compose.yaml`. Each job may also explicitly
request its policy. To stop the network demo:

```sh
docker compose down
```

See [ROADMAP.md](ROADMAP.md) for current scope, validation, and future work.

## Original SimPy simulator

Run in the current cloud workspace:

```sh
cd /workspace/TORCH
/workspace/.torch-venv/bin/python simulator.py
```

For a fresh checkout, run from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python simulator.py
```

The scheduler sees all nodes, considers jobs in arrival order, and uses the first
compatible node with enough free GPUs. Each job reserves whole GPUs on one node
and specifies the minimum VRAM per GPU and runtime by GPU type. A job waits if
its eligible nodes are busy and is rejected if no node could ever fit it.

The demo reports scheduling delay (arrival to start), job completion time
(arrival to finish), and aggregate GPU utilization over the simulated run.

## Run a comparison

The runner writes configuration and metrics as JSON. Use `--policy all` to
give all policies the same workload in one run.

```sh
/workspace/.torch-venv/bin/python experiment.py --policy all --seed 42 --jobs 100 \
  --arrival-rate 5 --l40-nodes 6 --a100-nodes 6
```

In a fresh checkout, use `.venv/bin/python` from the setup above. The same
seed, job count, and arrival rate generate the same jobs across cluster
configurations. Arrivals have exponentially distributed interarrival times.
GPU runtimes and VRAM requirements are synthetic inputs. Use
`experiment.py --help` for cluster size, neighborhood radius, sample size,
update frequency, staleness limit, latency, and retry controls.

| Policy | How schedulers discover capacity |
| --- | --- |
| `centralized` | One FIFO queue sees the entire cluster immediately. |
| `gossip` | Each scheduler exchanges versioned snapshots with a random peer in its neighborhood and stores only neighborhood records. |
| `periodic` | Each worker sends its own status periodically to every neighbor. Schedulers ignore snapshots older than the configured limit. |
| `sampling` | Each scheduler probes a random subset of its neighborhood at scheduling time and keeps its own worker in the candidate set. |
| `global-gossip` | Schedulers can gossip with any peer and keep snapshots from across the cluster. This is an OpenTela-inspired comparison. |

Each worker owns its GPU reservations. Checking and reserving capacity is one
SimPy step at the worker, so competing remote requests cannot overbook it.
Schedulers start with the local worker and a bounded peer set from a ring
with seeded shortcuts. A job that cannot be placed can be forwarded around
the ring. It retries up to the configured round limit. A worker outage
interrupts active jobs, which can restart on another worker. The scheduler
role remains available during a GPU worker outage.

Reports include completion time, scheduling delay, utilization, throughput,
failed reservations, retries, control-plane message counts, and status
record counts. Record count approximates payload volume. Utilization uses
nominal cluster capacity and includes time spent on interrupted executions.
The idealized centralized baseline has no modeled network traffic and uses
strict FIFO. The distributed policies use the same local first-fit rule and
can place independent jobs concurrently. Use their results to isolate the
effect of changing the discovery policy.

## Demonstrate recovery

```sh
/workspace/.torch-venv/bin/python experiment.py --policy gossip --jobs 30 \
  --crash-node a100-1 --crash-at 7 --recover-at 12 --drop-status-until 3
/workspace/.torch-venv/bin/python -m unittest -v
```

The tests include three demonstration scenarios for the course report:

1. Stale capacity causes competing placement attempts. The worker rejects
   the conflicting request, and the second job starts after capacity frees up.
2. A GPU worker crashes during execution. Its interrupted job restarts on
   another worker.
3. A temporary loss of status messages hides remote capacity. Ring forwarding
   still places the job, and status delivery resumes after the loss window.

## Course submission status

- The prototype implements three algorithm categories listed in the rubric:
  gossip, resource discovery, and resource allocation. It exercises
  consistency and concurrency through versioned views and atomic reservations,
  and fault tolerance through simulated worker loss and job restart.
- The simulator uses fixed membership, modeled message delays, and trusted
  participants. SWIM membership, a full parameter sweep and analysis, and the
  proposal's real-deployment validation are still open.
- The team still needs the IEEE report with the required UML diagrams and
  demonstration snapshots, the recorded presentation and slides, and the
  division of work in this README. The rubric lists code and presentation
  submissions for December 3 and the report for December 4.
