"""The GitLab guard: pause Panther while the shared service is unhealthy."""

from __future__ import annotations

from pydantic import BaseModel


class Guard(BaseModel):
    """Counts healthy/unhealthy checks and pauses after 3 bad, resumes after 5 good."""

    bad: int = 0  # consecutive unhealthy checks
    good: int = 0  # consecutive healthy checks
    paused: bool = False  # whether Panther is paused now

    def tick(self, healthy: bool) -> str | None:
        """Update the counters; return ``"pause"``, ``"resume"`` or ``None`` on a state change."""
        if healthy:
            self.good += 1
            self.bad = 0
        else:
            self.bad += 1
            self.good = 0
        if self.bad >= 3 and not self.paused:
            self.paused = True
            return "pause"
        if self.good >= 5 and self.paused:
            self.paused = False
            return "resume"
        return None
