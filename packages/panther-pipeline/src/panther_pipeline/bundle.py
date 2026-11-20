"""The Pi bundle manifest (ADR 0007: a frozen external contract)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BundleManifest(BaseModel):
    """What the Pi needs to verify a winner."""

    model_config = ConfigDict(frozen=True)

    file: str  # the GGUF name
    sha256: str  # file hash
    budget_gib: int  # RAM budget
    predicted_ram_gib: float  # predicted peak RAM
    context: int  # context tokens
    kv_cache_type: str  # Q8_0 or F16
