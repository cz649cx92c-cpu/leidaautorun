"""Small, ROS-independent helpers for identifying stale vehicle feedback."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeedbackFreshness:
    """Track when the bridge's monotonically increasing feedback sequence moves."""

    timeout_s: float = 2.0
    last_seq: int | None = None
    last_change_at: float | None = None

    def observe(self, seq: object, now: float) -> tuple[bool, int | None]:
        try:
            sequence = int(seq)
        except (TypeError, ValueError, OverflowError):
            sequence = None
        if sequence is not None and sequence != self.last_seq:
            self.last_seq = sequence
            self.last_change_at = now
        if self.last_change_at is None:
            return False, None
        age_ms = max(0, round((now - self.last_change_at) * 1000.0))
        return age_ms <= int(self.timeout_s * 1000), age_ms
