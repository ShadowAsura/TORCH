"""CPU-only SimPy baseline for heterogeneous GPU scheduling."""

from collections import deque
from dataclasses import dataclass, field
from statistics import mean

import simpy


@dataclass(frozen=True)
class Node:
    name: str
    gpu_type: str
    gpu_count: int
    vram_gb: int


@dataclass(frozen=True)
class Job:
    name: str
    arrival: float
    gpu_count: int
    min_vram_gb: int
    runtime_by_gpu: dict[str, float]


@dataclass(frozen=True, eq=False)
class Reservation:
    gpu_count: int


class NodeResources:
    def __init__(self, node: Node):
        self.node = node
        self._free_gpus = node.gpu_count
        self._reservations: set[Reservation] = set()

    @property
    def free_gpus(self) -> int:
        return self._free_gpus

    def supports(self, job: Job) -> bool:
        return (
            0 < job.gpu_count <= self.node.gpu_count
            and job.min_vram_gb <= self.node.vram_gb
            and self.node.gpu_type in job.runtime_by_gpu
        )

    def try_reserve(self, job: Job) -> Reservation | None:
        """Check and reserve capacity without yielding to another SimPy process."""
        if not self.supports(job) or job.gpu_count > self._free_gpus:
            return None
        reservation = Reservation(job.gpu_count)
        self._reservations.add(reservation)
        self._free_gpus -= reservation.gpu_count
        return reservation

    def release(self, reservation: Reservation) -> None:
        self._reservations.remove(reservation)
        self._free_gpus += reservation.gpu_count


@dataclass(frozen=True)
class Placement:
    job: Job
    node: Node
    start: float
    finish: float


@dataclass
class Simulation:
    elapsed: float
    placements: list[Placement] = field(default_factory=list)
    rejected: list[Job] = field(default_factory=list)
    interrupted_gpu_seconds: float = 0


@dataclass(frozen=True)
class Metrics:
    mean_scheduling_delay: float
    mean_job_completion_time: float
    gpu_utilization: float


def summarize(nodes: list[Node], result: Simulation) -> Metrics:
    placements = result.placements
    total_capacity = result.elapsed * sum(node.gpu_count for node in nodes)
    busy_gpu_time = result.interrupted_gpu_seconds + sum(
        placement.job.gpu_count * (placement.finish - placement.start)
        for placement in placements
    )
    if not placements:
        return Metrics(0.0, 0.0, 0.0)
    return Metrics(
        mean(placement.start - placement.job.arrival for placement in placements),
        mean(placement.finish - placement.job.arrival for placement in placements),
        busy_gpu_time / total_capacity if total_capacity else 0.0,
    )


def simulate(nodes: list[Node], jobs: list[Job]) -> Simulation:
    env = simpy.Environment()
    resources = [NodeResources(node) for node in nodes]
    queue: deque[tuple[Job, list[NodeResources]]] = deque()
    result = Simulation(elapsed=0)

    def dispatch():
        while queue:
            job, eligible = queue[0]
            for resource in eligible:
                reservation = resource.try_reserve(job)
                if reservation is not None:
                    queue.popleft()
                    env.process(run(job, resource, reservation))
                    break
            else:
                return

    def run(job: Job, resource: NodeResources, reservation: Reservation):
        node = resource.node
        start = env.now
        yield env.timeout(job.runtime_by_gpu[node.gpu_type])
        result.placements.append(Placement(job, node, start, env.now))
        resource.release(reservation)
        dispatch()

    def arrive(job: Job):
        yield env.timeout(job.arrival)
        eligible = [resource for resource in resources if resource.supports(job)]
        if not eligible:
            result.rejected.append(job)
            return
        queue.append((job, eligible))
        dispatch()

    for job in jobs:
        env.process(arrive(job))
    env.run()
    result.elapsed = env.now
    return result


def main():
    nodes = [
        Node("node-1", "L40", gpu_count=2, vram_gb=48),
        Node("node-2", "A100", gpu_count=2, vram_gb=80),
    ]
    jobs = [
        Job("A", 0, 1, 24, {"L40": 6, "A100": 4}),
        Job("B", 0, 2, 64, {"A100": 5}),
        Job("C", 1, 2, 24, {"L40": 4, "A100": 3}),
        Job("D", 2, 1, 24, {"L40": 3, "A100": 2}),
    ]
    result = simulate(nodes, jobs)
    metrics = summarize(nodes, result)
    print(f"{'Job':<5} {'GPU':<6} {'Count':>5} {'Arrival':>8} {'Start':>8} {'Finish':>8}")
    for placement in result.placements:
        job = placement.job
        print(
            f"{job.name:<5} {placement.node.gpu_type:<6} {job.gpu_count:>5} "
            f"{job.arrival:>8g} {placement.start:>8g} {placement.finish:>8g}"
        )
    for job in result.rejected:
        print(f"{job.name} rejected (no compatible node)")
    print(f"Completed {len(result.placements)} jobs in {result.elapsed:g} simulated seconds")
    print(f"Mean scheduling delay {metrics.mean_scheduling_delay:.2f} s")
    print(f"Mean job completion time {metrics.mean_job_completion_time:.2f} s")
    print(f"GPU utilization {metrics.gpu_utilization:.1%}")


if __name__ == "__main__":
    main()
