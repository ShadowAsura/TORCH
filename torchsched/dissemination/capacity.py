"""
Helpers shared by dissemination implementations.
"""

from __future__ import annotations

from torchsched.types.models import Status


def gained_capacity(old: Status | None, new: Status) -> bool:
    """
    Whether `new` may offer capacity that `old` did not: the node is new or
    restarted, its hardware changed, or some group has more free GPUs.
    """
    if old is None or old.spec != new.spec or old.incarnation != new.incarnation:
        return True
    return any(n > o for n, o in zip(new.free_gpus, old.free_gpus))
