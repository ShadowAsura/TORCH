"""
Resend a message until it is acknowledged, a bounded number of times.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchsched.types.actions import Action, CancelTimer, Send, SetTimer
from torchsched.types.messages import Message
from torchsched.types.models import NodeId

# What an outstanding message is tracked by, for example ("reserve", request_id).
Key = tuple[str, str]


@dataclass
class _Outstanding:
    to: NodeId
    msg: Message
    sends: int = 1


class Retransmitter:
    """
    Tracks messages that need an acknowledgement, for one component.

    `send` sends a message and arms a timer `("resend", *key)`. The owning
    component routes that timer to `timed_out`, which sends the same
    message again, up to `max_sends` sends in all, then gives up. Call
    `acknowledged` when the reply or acknowledgement arrives.

    Resending the same message is only safe when the receiver handles
    duplicates, for example by request id.
    """

    def __init__(self, *, timeout: float, max_sends: int):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_sends < 1:
            raise ValueError("max_sends must be at least 1")
        self.timeout = timeout
        self.max_sends = max_sends
        self._outstanding: dict[Key, _Outstanding] = {}

    def __contains__(self, key: Key) -> bool:
        return key in self._outstanding

    def send(self, key: Key, to: NodeId, msg: Message) -> list[Action]:
        self._outstanding[key] = _Outstanding(to, msg)
        return [Send(to, msg), SetTimer(("resend", *key), self.timeout)]

    def acknowledged(self, key: Key) -> list[Action]:
        if self._outstanding.pop(key, None) is None:
            return []
        return [CancelTimer(("resend", *key))]

    def timed_out(self, key: Key) -> tuple[list[Action], _Outstanding | None]:
        """
        Resend the message for `key`, or give up on it.

        Returns the resend actions, or, after the last send, no actions and
        the abandoned entry (its `to` and `msg`). Unknown keys return
        `([], None)`.
        """
        entry = self._outstanding.get(key)
        if entry is None:
            return [], None
        if entry.sends >= self.max_sends:
            del self._outstanding[key]
            return [], entry
        entry.sends += 1
        return [Send(entry.to, entry.msg), SetTimer(("resend", *key), self.timeout)], None
