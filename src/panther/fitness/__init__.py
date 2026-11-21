"""Fitness: flips % + 10 × KL against the original, lower is better. See README.md."""

from panther.fitness.fitness import KL_WEIGHT, Fitness, Score

__all__ = ["KL_WEIGHT", "Fitness", "Score"]
