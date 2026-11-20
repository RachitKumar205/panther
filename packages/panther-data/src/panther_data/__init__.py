"""Dataset splits, prompt building, config schemas and integrity checks."""

from __future__ import annotations

from panther_data.config import (
    ContestSpec,
    LlamaCpp,
    ModelConfig,
    ModelEntry,
    ModelsConfig,
    SearchConfig,
    load_model_config,
    load_models,
    load_search_config,
)
from panther_data.dataset import (
    ChatPrompt,
    Question,
    SplitName,
    SplitSpec,
    WebDocument,
    fill_to_budget,
    overlapping,
    partition_questions,
    split_prompts,
    words,
)
from panther_data.integrity import manifest_problems, question_overlap, sha256_file

__all__ = [
    "ChatPrompt",
    "ContestSpec",
    "LlamaCpp",
    "ModelConfig",
    "ModelEntry",
    "ModelsConfig",
    "Question",
    "SearchConfig",
    "SplitName",
    "SplitSpec",
    "WebDocument",
    "fill_to_budget",
    "load_model_config",
    "load_models",
    "load_search_config",
    "manifest_problems",
    "overlapping",
    "partition_questions",
    "question_overlap",
    "sha256_file",
    "split_prompts",
    "words",
]
