"""
Plain data shared by every component and runtime.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

# arbitrary, but unique, identifiers. runtime-dependent
NodeId = str
JobId = str
ReservationId = str
# ie "a-100".
GpuType = str
# position of a GpuGroup in NodeSpec.gpus
GroupIndex = int


@dataclass(frozen=True, slots=True)
class GpuGroup:
    """
    Identical GPUs on one node that a single job may span.

    A node with mixed hardware has one group per kind of GPU. The same GPU
    type with different memory (A100 40 GB vs 80 GB) is a separate group.
    Two groups may even be identical, for example two NVLink islands of four
    GPUs each that a job should not span.
    """

    gpu_type: GpuType
    count: int
    vram_gb: float  # per GPU

    def __post_init__(self) -> None:
        if not self.gpu_type:
            raise ValueError("gpu_type must be nonempty")
        if self.count < 1:
            raise ValueError("count must be positive")
        if self.vram_gb <= 0:
            raise ValueError("vram_gb must be positive")


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """
    Static description of one node's hardware.
    """

    node_id: NodeId
    # may be empty. Normally means that the node isn't set up to be a worker
    gpus: tuple[GpuGroup, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be nonempty")
        # accept any sequence, but store a tuple so the spec stays immutable.
        object.__setattr__(self, "gpus", tuple(self.gpus))

    @property
    def gpu_count(self) -> int:
        return sum(group.count for group in self.gpus)


@dataclass(frozen=True, slots=True)
class Job:
    """
    A request for whole, homogeneous GPUs on one node.

    The job runs inside a single GpuGroup. `runtime_by_gpu` gives the
    execution time in seconds on each GPU type. Its keys are exactly the
    compatible GPU types, so heterogeneity-aware policies (for example,
    Sia-style) can compare speedups directly.
    """

    job_id: JobId
    gpu_count: int
    min_vram_gb: float
    runtime_by_gpu: Mapping[GpuType, float]

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must be nonempty")
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be positive")
        if self.min_vram_gb < 0:
            raise ValueError("min_vram_gb must be nonnegative")
        if not self.runtime_by_gpu or any(t <= 0 for t in self.runtime_by_gpu.values()):
            raise ValueError("runtime_by_gpu must be nonempty with positive runtimes")
        # freeze the mapping so a Job cannot change after it is sent.
        object.__setattr__(
            self, "runtime_by_gpu", MappingProxyType(dict(self.runtime_by_gpu))
        )

    def __hash__(self) -> int:
        # dataclass's hash would error on `runtime_by_gpu` since it's a mapping
        # as such, use a frozenset
        return hash(
            (
                self.job_id,
                self.gpu_count,
                self.min_vram_gb,
                frozenset(self.runtime_by_gpu.items()),
            )
        )

    def can_use(self, group: GpuGroup) -> bool:
        """
        Whether this group could ever run the job, ignoring current load.
        """
        return (
            group.gpu_type in self.runtime_by_gpu
            and self.gpu_count <= group.count
            and self.min_vram_gb <= group.vram_gb
        )

    def compatible_with(self, spec: NodeSpec) -> bool:
        """
        Whether any group on this node could ever run the job.
        """
        return any(self.can_use(group) for group in spec.gpus)


@dataclass(frozen=True, slots=True)
class Status:
    """
    One worker's resource state at a point in time.

    `free_gpus[i]` is the number of free GPUs in `spec.gpus[i]`.

    Records from the same node are ordered by `(incarnation, version)`.
    `version` increases on every change to the worker's allocations.
    `incarnation` increases every time the node restarts and loses its
    state, so a record from before a crash never overrides a newer one.
    `observed_at` is the worker's clock when it produced the record. It's
    used for staleness limits and assumes loosely synchronized clocks.
    """

    spec: NodeSpec
    free_gpus: tuple[int, ...]
    incarnation: int
    version: int
    observed_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "free_gpus", tuple(self.free_gpus))
        if len(self.free_gpus) != len(self.spec.gpus):
            raise ValueError("free_gpus must have one entry per GPU group")
        for free, group in zip(self.free_gpus, self.spec.gpus):
            if not 0 <= free <= group.count:
                raise ValueError("free_gpus entries must be within each group's count")

    @property
    def node_id(self) -> NodeId:
        return self.spec.node_id

    @property
    def total_free(self) -> int:
        return sum(self.free_gpus)

    def order_key(self) -> tuple[int, int]:
        return (self.incarnation, self.version)

    def fitting_groups(self, job: Job) -> tuple[GroupIndex, ...]:
        """
        Groups that could run the job now, according to this (possibly stale) record.
        """
        return tuple(
            index
            for index, (group, free) in enumerate(zip(self.spec.gpus, self.free_gpus))
            if job.can_use(group) and job.gpu_count <= free
        )

    def fits(self, job: Job) -> bool:
        return bool(self.fitting_groups(job))


@dataclass(frozen=True, slots=True)
class Assignment:
    """
    A placement decision: try `job_id` on one GPU group of `node_id`.
    """

    job_id: JobId
    node_id: NodeId
    group: GroupIndex
