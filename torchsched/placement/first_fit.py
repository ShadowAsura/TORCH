"""
First-fit placement, with or without backfilling.
"""

from __future__ import annotations

from collections.abc import Sequence

from torchsched.types.models import Assignment, Job, Status


class FirstFit:
    """
    Place jobs in queue order on the first GPU group with room.

    Candidates are tried in the order given, and groups in node order. With
    `backfill=False` this is strict FIFO: the first job that does not fit
    blocks every job behind it. With `backfill=True`, later jobs that fit
    are placed even when an earlier one does not.
    """

    def __init__(self, *, backfill: bool = False):
        self.backfill = backfill
        self.name = "first-fit-backfill" if backfill else "first-fit"

    def plan(
        self, pending: Sequence[Job], candidates: Sequence[Status], now: float
    ) -> list[Assignment]:
        # TODO: handle jobs that no node can ever run. Under strict FIFO such
        # a job blocks the queue forever; for now, callers must filter them
        # out before submitting.
        free = {status.node_id: list(status.free_gpus) for status in candidates}
        assignments: list[Assignment] = []
        for job in pending:
            assignment = self._first_fit(job, candidates, free)
            if assignment is not None:
                free[assignment.node_id][assignment.group] -= job.gpu_count
                assignments.append(assignment)
            elif not self.backfill:
                break
        return assignments

    @staticmethod
    def _first_fit(
        job: Job, candidates: Sequence[Status], free: dict[str, list[int]]
    ) -> Assignment | None:
        for status in candidates:
            for index, group in enumerate(status.spec.gpus):
                if job.can_use(group) and free[status.node_id][index] >= job.gpu_count:
                    return Assignment(job.job_id, status.node_id, index)
        return None
