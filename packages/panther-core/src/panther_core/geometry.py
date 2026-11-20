"""Model geometry: the weight matrices of a decoder-only transformer.

The geometry is derived from a small, typed spec (a subset of a Hugging Face ``config.json``).
It is deliberately independent of pydantic so that the pure domain has no heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import dataclass

from panther_core.quant import QuantType


@dataclass(frozen=True, slots=True)
class Matrix:
    """One weight matrix of the model."""

    name: str
    """GGUF tensor name, e.g. ``blk.0.attn_q.weight``."""
    rows: int
    """Output features."""
    cols: int
    """Input features."""
    layer: int
    """Layer index, or ``-1`` for the token embedding and output head."""
    role: str
    """Matrix role, e.g. ``attn_q``, ``ffn_down``, ``token_embd``, ``output``."""

    def nbytes(self, quant: QuantType) -> int:
        """Bytes of this matrix stored in ``quant``."""
        return quant.nbytes(self.rows, self.cols)


@dataclass(frozen=True, slots=True)
class GeometrySpec:
    """The fields of a model config that determine its matrices and RAM footprint."""

    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: int | None = None
    tie_word_embeddings: bool = False
    model_type: str = ""
    imatrix_process_output: bool = False

    def __post_init__(self) -> None:
        for field in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive, got {getattr(self, field)}")
        if self.head_dim is None and self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by {self.num_attention_heads} heads"
            )

    @property
    def head_width(self) -> int:
        """Width of one attention head."""
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @property
    def attention_width(self) -> int:
        """Width of the query/output projection (``num_attention_heads * head_dim``)."""
        return self.num_attention_heads * self.head_width

    @property
    def kv_width(self) -> int:
        """Width of the key/value projections (``num_key_value_heads * head_dim``)."""
        return self.num_key_value_heads * self.head_width


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    """The matrices, parameter count and fixed (non-matrix) bytes of a model."""

    spec: GeometrySpec
    matrices: tuple[Matrix, ...]
    params: int
    fixed_bytes: int
    no_imatrix: frozenset[str]

    @classmethod
    def from_spec(cls, spec: GeometrySpec) -> ModelGeometry:
        """Build the geometry for ``spec``."""
        hidden = spec.hidden_size
        attention = spec.attention_width
        kv = spec.kv_width
        intermediate = spec.intermediate_size

        matrices: list[Matrix] = [Matrix("token_embd.weight", spec.vocab_size, hidden, -1, "token_embd")]
        if not spec.tie_word_embeddings:
            matrices.append(Matrix("output.weight", spec.vocab_size, hidden, -1, "output"))
        for layer in range(spec.num_hidden_layers):
            for role, rows, cols in (
                ("attn_q", attention, hidden),
                ("attn_k", kv, hidden),
                ("attn_v", kv, hidden),
                ("attn_output", hidden, attention),
                ("ffn_gate", intermediate, hidden),
                ("ffn_up", intermediate, hidden),
                ("ffn_down", hidden, intermediate),
            ):
                matrices.append(Matrix(f"blk.{layer}.{role}.weight", rows, cols, layer, role))

        params = sum(matrix.rows * matrix.cols for matrix in matrices)
        bias = (attention + 2 * kv) if spec.model_type == "qwen2" else 0
        fixed_bytes = 4 * (spec.num_hidden_layers * (2 * hidden + bias) + hidden)
        no_imatrix_roles: set[str] = {"token_embd"}
        if not spec.imatrix_process_output:
            no_imatrix_roles.add("output")
        return cls(
            spec=spec,
            matrices=tuple(matrices),
            params=params,
            fixed_bytes=fixed_bytes,
            no_imatrix=frozenset(matrix.name for matrix in matrices if matrix.role in no_imatrix_roles),
        )

    def allowed(
        self,
        formats: Sequence[QuantType],
        requires_imatrix: Set[QuantType],
    ) -> dict[str, tuple[QuantType, ...]]:
        """Every allowed type per matrix, low to high bits per weight.

        A matrix without importance-matrix data may not use a type that requires it.
        """
        ordered = sorted(formats, key=lambda quant: quant.bits_per_weight)
        return {
            matrix.name: tuple(
                quant
                for quant in ordered
                if not (matrix.name in self.no_imatrix and quant in requires_imatrix)
            )
            for matrix in self.matrices
        }
