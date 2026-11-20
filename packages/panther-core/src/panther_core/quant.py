"""llama.cpp quantization types and exact byte accounting.

The 14 storage formats Panther searches over, plus the unquantized types used by the original
weights and by norms/biases. Every block size and byte size comes from :mod:`gguf`; nothing is
restated here.
"""

from __future__ import annotations

from enum import StrEnum

from gguf import GGML_QUANT_SIZES, GGMLQuantizationType


class QuantType(StrEnum):
    """A llama.cpp tensor type used by Panther."""

    # The 14 searchable storage formats, low to high bits per weight.
    IQ1_S = "IQ1_S"
    IQ1_M = "IQ1_M"
    IQ2_XXS = "IQ2_XXS"
    IQ2_XS = "IQ2_XS"
    IQ2_S = "IQ2_S"
    Q2_K = "Q2_K"
    IQ3_XXS = "IQ3_XXS"
    Q3_K = "Q3_K"
    IQ3_S = "IQ3_S"
    IQ4_XS = "IQ4_XS"
    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"

    # Unquantized types (the original weights, norms and biases).
    BF16 = "BF16"
    F16 = "F16"
    F32 = "F32"

    @property
    def ggml(self) -> GGMLQuantizationType:
        """The corresponding :class:`gguf.GGMLQuantizationType`."""
        return GGMLQuantizationType[self.value]

    @property
    def block(self) -> int:
        """Number of weights in one storage block."""
        return int(GGML_QUANT_SIZES[self.ggml][0])

    @property
    def type_size(self) -> int:
        """Bytes used by one storage block."""
        return int(GGML_QUANT_SIZES[self.ggml][1])

    @property
    def bits_per_weight(self) -> float:
        """Nominal bits per weight."""
        return 8 * self.type_size / self.block

    def nbytes(self, rows: int, cols: int) -> int:
        """Bytes of a ``rows x cols`` matrix stored in this type.

        Args:
            rows: Number of output rows.
            cols: Number of input columns.

        Raises:
            ValueError: If a dimension is negative or ``cols`` is not a multiple of the block size.
        """
        if rows < 0 or cols < 0:
            raise ValueError(f"negative matrix shape {rows}x{cols}")
        if cols % self.block:
            raise ValueError(f"{cols} columns do not fit {self.value} (block {self.block})")
        return rows * (cols // self.block) * self.type_size


SEARCH_FORMATS: tuple[QuantType, ...] = (
    QuantType.IQ1_S,
    QuantType.IQ1_M,
    QuantType.IQ2_XXS,
    QuantType.IQ2_XS,
    QuantType.IQ2_S,
    QuantType.Q2_K,
    QuantType.IQ3_XXS,
    QuantType.Q3_K,
    QuantType.IQ3_S,
    QuantType.IQ4_XS,
    QuantType.Q4_K,
    QuantType.Q5_K,
    QuantType.Q6_K,
    QuantType.Q8_0,
)
"""The 14 types Panther searches over, ordered by bits per weight."""

IMATRIX_FORMATS: frozenset[QuantType] = frozenset(
    {QuantType.IQ1_S, QuantType.IQ1_M, QuantType.IQ2_XXS, QuantType.IQ2_XS}
)
"""Types llama.cpp refuses for a matrix that has no importance-matrix data."""
