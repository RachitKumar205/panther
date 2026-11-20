"""Resource claims: reserve a run's GPU/RAM, refuse when it does not fit (ADR 0026, bugs B9/B16)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    """One run's reserved GPU and RAM."""

    model_config = ConfigDict(frozen=True)

    id: str  # run id, e.g. "14b-b5"
    gpu: int  # bytes
    ram: int  # bytes


class Refused(Exception):
    """A claim does not fit beside the active runs."""


class Claims(BaseModel):
    """The set of active claims against configurable caps."""

    gpu_cap: int  # total GPU budget, bytes
    ram_cap: int  # total RAM budget, bytes
    active: dict[str, Claim] = Field(default_factory=dict)  # run id -> claim

    def fits(self, gpu: int, ram: int) -> bool:
        return self._used("gpu") + gpu <= self.gpu_cap and self._used("ram") + ram <= self.ram_cap

    def reserve(self, claim: Claim) -> None:
        if not self.fits(claim.gpu, claim.ram):
            raise Refused(f"{claim.id} does not fit beside {list(self.active)}")
        self.active[claim.id] = claim

    def drop(self, run_id: str) -> None:
        self.active.pop(run_id, None)

    def _used(self, field: str) -> int:
        return sum(getattr(claim, field) for claim in self.active.values())
