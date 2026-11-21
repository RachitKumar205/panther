"""A model and its files: the download, the 16-bit original, the importance matrix and the 13 type copies."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import gguf
import torch
import yaml
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from panther.model.gptq import GPTQ, collect_hessians, input_of
from panther.model.quantizer import Quantizer

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("PANTHER_DATA", REPO / "data"))
CONVERT = Path(os.environ.get("PANTHER_CONVERT", "/opt/llama.cpp/convert_hf_to_gguf.py"))
GIB = 1024**3

# The 13 types a matrix may take, fewest bits first. IQ3_S is left out: the same size as Q3_K, measured the same.
TYPES = ("IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "Q2_K", "IQ3_XXS", "Q3_K",
         "IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q8_0")
# Types never given to the word table: it has no importance data (it is a lookup, it has no input). In those four
# type copies the word table is stored as Q8_0 instead.
NEEDS_IMPORTANCE = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS"}
# llama-quantize takes a preset name; with --pure these four presets are their type everywhere.
PRESET_NAME = {"Q3_K": "Q3_K_S", "IQ2_S": "IQ2_M", "Q4_K": "Q4_K_S", "Q5_K": "Q5_K_S"}
# GPTQ's calibration: 128 sequences of 512 tokens of the distill set's web text.
SEQUENCES, SEQUENCE_LENGTH = 128, 512


def bits(type_: str) -> float:
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[type_]]
    return 8 * size / block


@dataclass
class Model:
    """One model of configs/models.yaml, and where its files live."""

    name: str
    repo: str
    revision: str  # the Hugging Face commit it is pinned to
    dir: Path = field(init=False)
    hf: Path = field(init=False)  # the download: kept, so nothing is ever downloaded twice
    original: Path = field(init=False)  # the 16-bit GGUF: "the original" from the conversion on
    imatrix: Path = field(init=False)
    types: Path = field(init=False)  # the type copies: types/<TYPE>.gguf

    def __post_init__(self) -> None:
        self.dir = DATA / "models" / self.name
        self.hf = self.dir / "hf"
        self.original = self.dir / "original.gguf"
        self.imatrix = self.dir / "imatrix.gguf"
        self.types = self.dir / "types"

    @classmethod
    def load(cls, name: str) -> Model:
        """By full name (Qwen2.5-14B-Instruct) or size (14b)."""
        entries = yaml.safe_load((REPO / "configs/models.yaml").read_text())["models"]
        [entry] = [e for e in entries if name in (e["name"], e["name"].split("-")[1].lower())]
        return cls(**entry)

    def copy(self, type_: str) -> Path:
        """The whole model in type_."""
        return self.types / f"{type_}.gguf"

    # --- the steps; each is skipped if its file exists, and writes under a .part name renamed when complete ---

    def build(self, text: Path) -> None:
        """Every step. text: the distill set's web text."""
        self.download()
        self.convert()
        self.make_imatrix(text)
        self.make_type_copies(text)

    def download(self) -> None:
        """The weights at the pinned commit, once: everything after this runs from them without the network.
        Resumes a cut-off download."""
        done = self.hf / ".downloaded"
        if done.exists():
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self.require_disk(2 * HfApi().model_info(self.repo, revision=self.revision).safetensors.total)  # bf16
        snapshot_download(self.repo, revision=self.revision, local_dir=self.hf,
                          ignore_patterns=["consolidated*", "original/*", "*.pth"])  # Mistral's duplicate weights
        done.touch()

    def convert(self) -> None:
        """The 16-bit GGUF (bf16, as the weights are published), by llama.cpp's convert_hf_to_gguf.py."""
        if self.original.exists():
            return
        self.require_disk(sum(f.stat().st_size for f in self.hf.glob("*.safetensors")))  # bf16 in, bf16 out
        part = self.dir / "original.part.gguf"
        subprocess.run([sys.executable, CONVERT, self.hf, "--outtype", "bf16", "--outfile", part], check=True)
        os.replace(part, self.original)

    def make_imatrix(self, text: Path) -> None:
        """Each matrix's mean squared input per column, over text, by llama-imatrix. --process-output includes the
        output layer (llama-imatrix leaves it out by default), so it can take every type."""
        if self.imatrix.exists():
            return
        part = self.dir / "imatrix.part.gguf"  # llama-imatrix writes GGUF when the name ends in .gguf
        subprocess.run(["llama-imatrix", "-m", self.original, "-f", text, "-o", part, "-c", "512", "-ngl", "99",
                        "--process-output"], check=True)
        os.replace(part, self.imatrix)

    def make_type_copies(self, text: Path) -> None:
        """Every missing type copy: llama-quantize writes the whole model in the type, then GPTQ overwrites every
        layer matrix and the output layer with error-compensated codes. A stopped build's leftovers are redone."""
        todo = [t for t in TYPES if not self.copy(t).exists()]
        if not todo:
            return
        self.types.mkdir(exist_ok=True)
        for leftover in self.types.glob("*.part.gguf"):
            leftover.unlink()
        self.require_disk(int(sum(self.original.stat().st_size * bits(t) / 16 for t in todo)))  # from 16 bits
        parts = {t: self.types / f"{t}.part.gguf" for t in todo}

        for t, part in parts.items():
            word_table = ["--token-embedding-type", "q8_0"] if t in NEEDS_IMPORTANCE else []
            subprocess.run(["llama-quantize", "--pure", "--imatrix", self.imatrix, *word_table,
                            self.original, part, PRESET_NAME.get(t, t)], check=True)

        original = {x.name: x for x in gguf.GGUFReader(self.original).tensors}
        files = [gguf.GGUFReader(p, "r+") for p in parts.values()]
        copies = {t: {x.name: x for x in f.tensors} for t, f in zip(parts, files, strict=True)}
        # every matrix but the word table; a model whose output layer is the word table (tied) has no output.weight
        matrices = [n for n, x in original.items() if len(x.shape) == 2 and n != "token_embd.weight"]
        readers = defaultdict(list)  # transformers module -> the matrices reading its input
        for name in matrices:
            readers[input_of(name)].append(name)

        quantizer = Quantizer()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        done = set()
        for hessians in self.hessians(text, list(readers)):
            for module, h in hessians.items():
                gptq = GPTQ(h.to(device))
                for name in readers[module]:
                    w = torch.from_numpy(gguf.quants.dequantize(original[name].data, original[name].tensor_type))
                    for t, tensors in copies.items():
                        assert tensors[name].tensor_type.name == t
                        codes = gptq.round(w.to(device), t, quantizer)
                        tensors[name].data[...] = codes.reshape(tensors[name].data.shape)
                    done.add(name)
            del hessians  # free this group's H before the next group is collected (they share the GPU)
        assert done == set(matrices), f"no GPTQ codes for {sorted(set(matrices) - done)}"

        for f in files:
            f.data.flush()
        for t, part in parts.items():
            os.replace(part, self.copy(t))

    def hessians(self, text: Path, modules: list[str]) -> Iterator[dict[str, torch.Tensor]]:
        """H for each module's input: the original (the download, by transformers) reads the first 65,536 tokens of
        text. A group of modules at a time, as many H as half the free GPU memory holds."""
        ids = AutoTokenizer.from_pretrained(self.hf)(text.read_text(), return_tensors="pt").input_ids[0]
        ids = ids[: SEQUENCES * SEQUENCE_LENGTH].view(SEQUENCES, SEQUENCE_LENGTH)  # fails if the text is too short
        model = AutoModelForCausalLM.from_pretrained(self.hf, dtype=torch.bfloat16, device_map="cuda").eval()
        room = torch.cuda.mem_get_info()[0] // 2
        group, size = [], 0
        for name in modules:
            h_bytes = 4 * model.get_submodule(name).in_features ** 2  # float32, inputs x inputs
            if group and size + h_bytes > room:
                yield collect_hessians(model, ids, group)
                group, size = [], 0
            group.append(name)
            size += h_bytes
        yield collect_hessians(model, ids, group)

    def delete_type_copies(self) -> None:
        """Frees disk. build() makes them again from the kept download (no network)."""
        shutil.rmtree(self.types)

    def require_disk(self, need: int) -> None:
        free = shutil.disk_usage(self.dir).free
        if free < need + 20 * GIB:
            raise RuntimeError(f"needs {need / GIB:.0f} GiB + a 20 GiB margin of free disk, {free / GIB:.0f} GiB free")
