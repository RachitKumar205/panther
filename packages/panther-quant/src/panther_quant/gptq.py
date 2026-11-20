"""Error-compensated (GPTQ) rounding (algorithm 04).

Quantize column-chunks and push each chunk's error onto the remaining columns, weighted by the
inverse-Cholesky of the Hessian, so the *output* error stays small.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch

Quantize = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
"""Quantize a chunk ``(rows, c)`` to ``(codes, dequantized)``."""


def gptq(w: torch.Tensor, h: torch.Tensor, quantize: Quantize, chunk: int = 256) -> torch.Tensor:
    """Compensated quantization of ``w``, returning the dequantized weights ``(rows, cols)``.

    Args:
        w: ``(rows, cols)`` weights.
        h: ``(cols, cols)`` Hessian ``E[x^T x]``.
        quantize: chunk ``(rows, c)`` -> ``(codes, dequantized)``.
        chunk: column block size (256).
    """
    w = w.clone()
    h = h.clone()
    dead = torch.diag(h) == 0  # columns with no activation get weight 0
    h[dead, dead], w[:, dead] = 1.0, 0.0
    h = h + 0.01 * torch.mean(torch.diag(h)) * torch.eye(h.shape[0], device=h.device)  # 1% damping
    chol = cast(torch.Tensor, torch.linalg.cholesky(h))  # type: ignore[reportUnknownMemberType]
    inv = torch.cholesky_inverse(chol)
    u = cast(torch.Tensor, torch.linalg.cholesky(inv, upper=True))  # type: ignore[reportUnknownMemberType]
    dequantized: list[torch.Tensor] = []
    for a in range(0, w.shape[1], chunk):
        b = min(a + chunk, w.shape[1])
        _, q = quantize(w[:, a:b].cpu().numpy())
        q = torch.tensor(q).to(w.device, w.dtype)
        err = cast(torch.Tensor, torch.linalg.solve_triangular(u[a:b, a:b], w[:, a:b] - q, upper=True, left=False))  # type: ignore[reportUnknownMemberType]
        w[:, b:] -= err @ u[a:b, b:]
        dequantized.append(q)
    return torch.cat(dequantized, dim=1)
