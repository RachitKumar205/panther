"""Small GA sets for the tiny model: a text, and question files in llama.cpp's formats (the data part will write the
real ones)."""

from __future__ import annotations

import random
import struct
from pathlib import Path


def write_text(path: Path, words: int = 3000, seed: int = 0) -> Path:
    rng = random.Random(seed)
    vocabulary = ["the", "of", "and", "to", "in", "is", "was", "that", "for", "it", "with", "as", "on", "be", "at",
                  "by", "this", "had", "not", "are", "but", "from", "or", "have"]
    path.write_text(" ".join(rng.choice(vocabulary) for _ in range(words)))
    return path


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<I", len(b)) + b


def write_multiple_choice(path: Path, count: int, seed: int = 0) -> Path:
    """llama.cpp's binary multiple-choice file: a count, an offset per question, then each question, its answers,
    which is right (one 1), and no second answer set."""
    rng = random.Random(seed)
    blobs = []
    for i in range(count):
        answers = [f" {i + k}" for k in range(4)]
        right = rng.randrange(4)
        blobs.append(_string(f"Question: what comes after {i}?\nAnswer:") + struct.pack("<I", 4)
                     + b"".join(map(_string, answers)) + struct.pack("<4i", *[int(k == right) for k in range(4)])
                     + struct.pack("<I", 0))
    offsets, position = [], 4 + 4 * count
    for b in blobs:
        offsets.append(position)
        position += len(b)
    path.write_bytes(struct.pack("<I", count) + struct.pack(f"<{count}I", *offsets) + b"".join(blobs))
    return path


def write_winogrande(path: Path, count: int, seed: int = 0) -> Path:
    """The CSV llama-perplexity --winogrande reads: index, sentence with one _, two options, the right one (1 or 2)."""
    rng = random.Random(seed)
    rows = [f"{i},The cat sat on the _ near door {i},mat,hat,{rng.choice((1, 2))}" for i in range(count)]
    path.write_text("\n".join(rows) + "\n\n")  # llama.cpp drops a last line that ends at the end of the file
    return path


def ga_set(folder: Path) -> tuple[Path, dict[str, Path]]:
    """A text and four tasks' question files."""
    folder.mkdir(parents=True, exist_ok=True)
    questions = {task: write_multiple_choice(folder / f"{task}.bin", 12, seed)
                 for seed, task in enumerate(("mmlu", "hellaswag", "arc_challenge"))}
    questions["winogrande"] = write_winogrande(folder / "winogrande.csv", 10)
    return write_text(folder / "text.txt"), questions
