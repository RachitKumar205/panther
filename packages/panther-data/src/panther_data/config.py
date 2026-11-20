"""Typed configuration: models, geometry and search settings.

The pydantic models validate the YAML/JSON files; the domain is built from them via the pure
:mod:`panther_core` types, so this package is the only place that touches config files.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from panther_core.geometry import GeometrySpec
from panther_core.quant import QuantType
from panther_core.ram import RuntimeConfig


class ModelConfig(BaseModel):
    """A vendored Hugging Face ``config.json``, reduced to what the geometry needs."""

    model_config = ConfigDict(extra="ignore")

    model_type: str  # qwen2, mistral, ...
    hidden_size: int  # model width
    num_hidden_layers: int  # transformer blocks
    num_attention_heads: int  # query / output heads
    num_key_value_heads: int  # grouped-query KV heads
    intermediate_size: int  # FFN width
    vocab_size: int  # token vocabulary size
    head_dim: int | None = None  # per-head width; derived when None
    tie_word_embeddings: bool = False  # output shares the embedding

    def to_spec(self, imatrix_process_output: bool = False) -> GeometrySpec:
        return GeometrySpec(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            head_dim=self.head_dim,
            tie_word_embeddings=self.tie_word_embeddings,
            model_type=self.model_type,
            imatrix_process_output=imatrix_process_output,
        )


class ModelEntry(BaseModel):
    """One model in ``models.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str  # e.g. Qwen2.5-7B-Instruct
    repo: str  # Hugging Face repo id
    revision: str  # pinned commit SHA
    imatrix_process_output: bool = False  # collect importance for the output layer


class LlamaCpp(BaseModel):
    """The pinned llama.cpp build everything is measured with."""

    repo: str  # llama.cpp repository URL
    tag: str  # release tag
    commit: str  # exact commit everything is measured with


class ModelsConfig(BaseModel):
    """``models.yaml``."""

    models: list[ModelEntry]  # the models and their pinned revisions
    llama_cpp: LlamaCpp  # the pinned build


class ContestSpec(BaseModel):
    """The models searched at one budget."""

    model_config = ConfigDict(extra="forbid")

    searched: list[str]  # models planned at this budget


class SearchConfig(BaseModel):
    """``search.yaml``: the RAM model and the searchable formats."""

    model_config = ConfigDict(extra="forbid")

    context_tokens: int  # KV-cache context length
    kv_cache_type: str  # Q8_0 or F16
    overhead_gib: float  # fixed llama.cpp runtime overhead
    formats: list[QuantType]  # the 14 searchable storage types
    requires_imatrix: list[QuantType]  # types needing importance-matrix data
    contests: dict[int, ContestSpec]  # budget -> models searched

    @property
    def runtime(self) -> RuntimeConfig:
        return RuntimeConfig(
            context_tokens=self.context_tokens,
            kv_cache_type=self.kv_cache_type,
            overhead_gib=self.overhead_gib,
        )


def load_search_config(path: Path) -> SearchConfig:
    return SearchConfig.model_validate(yaml.safe_load(path.read_text()))


def load_models(path: Path) -> ModelsConfig:
    return ModelsConfig.model_validate(yaml.safe_load(path.read_text()))


def load_model_config(directory: Path, name: str) -> ModelConfig:
    return ModelConfig.model_validate(json.loads((directory / f"{name}.json").read_text()))
