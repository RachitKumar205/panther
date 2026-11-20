"""Model geometry: the weight matrices of a decoder-only transformer.

Derived from a small, typed spec (a subset of a Hugging Face ``config.json``).
"""

from __future__ import annotations

from collections.abc import Sequence, Set

from pydantic import BaseModel, ConfigDict, Field, model_validator

from panther_core.quant import QuantType


class Matrix(BaseModel):
    """One weight matrix of the model."""

    model_config = ConfigDict(frozen=True)

    name: str  # GGUF tensor name, e.g. blk.0.attn_q.weight
    rows: int  # output features
    cols: int  # input features
    layer: int  # layer index, or -1 for embedding / output head
    role: str  # attn_q, attn_k, ffn_down, token_embd, output

    def nbytes(self, quant: QuantType) -> int:
        return quant.nbytes(self.rows, self.cols)


class GeometrySpec(BaseModel):
    """The fields of a model config that determine its matrices and RAM footprint."""

    model_config = ConfigDict(frozen=True)

    num_hidden_layers: int = Field(gt=0)  # transformer blocks
    hidden_size: int = Field(gt=0)  # model width
    num_attention_heads: int = Field(gt=0)  # query / output heads
    num_key_value_heads: int = Field(gt=0)  # grouped-query KV heads
    intermediate_size: int = Field(gt=0)  # FFN width
    vocab_size: int = Field(gt=0)  # token vocabulary size
    head_dim: int | None = None  # per-head width; derived when None
    tie_word_embeddings: bool = False  # output shares the embedding
    model_type: str = ""  # e.g. qwen2, mistral (qwen2 has q/k/v biases)
    imatrix_process_output: bool = False  # output layer has importance data

    @model_validator(mode="after")
    def _check_divisible(self) -> GeometrySpec:
        if self.head_dim is None and self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by {self.num_attention_heads} heads"
            )
        return self

    @property
    def head_width(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @property
    def attention_width(self) -> int:
        return self.num_attention_heads * self.head_width

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_width


class ModelGeometry(BaseModel):
    """The matrices, parameter count and fixed (non-matrix) bytes of a model."""

    model_config = ConfigDict(frozen=True)

    spec: GeometrySpec  # the source config
    matrices: tuple[Matrix, ...]  # every weight matrix, in layer order
    params: int  # total parameters across all matrices
    fixed_bytes: int  # F32 norms and biases
    no_imatrix: frozenset[str]  # matrices with no importance data

    @classmethod
    def from_spec(cls, spec: GeometrySpec) -> ModelGeometry:
        hidden = spec.hidden_size
        attention = spec.attention_width
        kv = spec.kv_width
        intermediate = spec.intermediate_size

        matrices: list[Matrix] = [
            Matrix(name="token_embd.weight", rows=spec.vocab_size, cols=hidden, layer=-1, role="token_embd")
        ]
        if not spec.tie_word_embeddings:
            matrices.append(Matrix(name="output.weight", rows=spec.vocab_size, cols=hidden, layer=-1, role="output"))
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
                matrices.append(Matrix(name=f"blk.{layer}.{role}.weight", rows=rows, cols=cols, layer=layer, role=role))

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
            no_imatrix=frozenset(m.name for m in matrices if m.role in no_imatrix_roles),
        )

    def allowed(
        self,
        formats: Sequence[QuantType],
        requires_imatrix: Set[QuantType],
    ) -> dict[str, tuple[QuantType, ...]]:
        """Every allowed type per matrix, low to high bits per weight."""
        ordered = sorted(formats, key=lambda quant: quant.bits_per_weight)
        return {
            matrix.name: tuple(
                quant
                for quant in ordered
                if not (matrix.name in self.no_imatrix and quant in requires_imatrix)
            )
            for matrix in self.matrices
        }
