"""Block-AP (EfficientQAT's first stage) on a llama.cpp file: block by block, each layer matrix's weights are trained,
on the file's own grid, so the quantized block reproduces the original block; the new codes are written into a copy
of the file (same types, same size, same scales).

Ported from Rachit's blockap.py (origin/rachit), with his guarantees: codes always picked in 32 bits; every block
scored from the bytes that are saved; a block never saved worse than the file's."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import shutil
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from panther.distill.distill import MATRICES, permute, unpermute
from panther.model import Model
from panther.qat.grids import Grid

BATCH, EPOCHS = 2, 2  # 2 samples a step, every sample twice
RATE, LOWEST = 1e-3, 1 / 20  # a matrix's rate: 1e-3 × its weights' size (root mean square), cosine down to 1/20 of it
RECODE = 16  # steps between picking the codes again (the weights move by ~1e-5 a step)
CHECK, PATIENCE = 512, 2  # a validation check every 512 steps; stop after 2 without a better point


class QLinear(torch.nn.Module):
    """A layer matrix whose weight is its latent weights W (32-bit, starting at the original's) rounded to the file's
    grid under the file's scales; the gradient goes straight through to W. W is kept in the file's row order."""

    def __init__(self, linear: torch.nn.Linear, grid: Grid, raw: np.ndarray, heads: int | None) -> None:
        super().__init__()
        self.grid, self.heads, self.steps, self.codes = grid, heads, 0, None
        w = linear.weight.detach().float()
        self.W = torch.nn.Parameter(w if heads is None else permute(w, heads).contiguous())
        self.state = grid.unpack(raw, w.device)
        self.bias = None if linear.bias is None else linear.bias.detach().float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.codes is None or self.steps % RECODE == 0:
            self.codes = self.grid.encode(self.W, self.state)
        self.steps += 1
        w = self.grid.fake(self.W, self.state, self.codes)
        w = w if self.heads is None else unpermute(w, self.heads)
        return torch.nn.functional.linear(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    def saved(self) -> np.ndarray:
        """The matrix as the file's bytes, with codes picked now from the weights as they are."""
        return self.grid.pack({**self.state, **self.grid.encode(self.W, self.state)})


class BlockAP:
    """Block-AP for one model. tokens: the training samples (n × 2,048 token ids), validation: 64 held-back samples;
    both from the QAT text (the data part draws them). Work in data/models/<model>/qat/<start file>-<samples>/ (named
    by their content): the two hidden-state caches and each finished block's bytes, so a stopped run continues after
    its last finished block."""

    def __init__(self, model: Model, tokens: np.ndarray, validation: np.ndarray) -> None:
        self.model, self.tokens, self.validation = model, tokens, validation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self, start: Path, out: Path) -> list[dict]:
        """Trains every block of start (a file in our 13 types) against the original; out: a copy of start with the
        trained codes. Returns each block's validation error: before, after, and the file's."""
        net = AutoModelForCausalLM.from_pretrained(self.model.hf, dtype=torch.bfloat16, attn_implementation="sdpa")
        net.eval().requires_grad_(False)
        reader = gguf.GGUFReader(start)
        self.file = {t.name: t for t in reader.tensors}
        llama = reader.fields["general.architecture"].contents() == "llama"  # q/k rows reordered (Mistral)
        self.heads = {"self_attn.q_proj": net.config.num_attention_heads,
                      "self_attn.k_proj": net.config.num_key_value_heads} if llama else {}
        n, length, hidden = *self.tokens.shape, net.config.hidden_size
        assert n % BATCH == 0 and len(self.validation) % BATCH == 0
        with open(start, "rb") as f:
            file_id = hashlib.file_digest(f, "sha256").hexdigest()[:12]
        samples_id = hashlib.sha256(self.tokens.tobytes() + self.validation.tobytes()).hexdigest()[:12]
        self.dir = self.model.dir / "qat" / f"{file_id}-{samples_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model.require_disk(2 * n * length * hidden * 2)  # the two caches, 16 bits each
        self.cache = {k: np.lib.format.open_memmap(self.dir / f"{k}.npy", "w+", np.int16, (n, length, hidden))
                      for k in ("original", "quantized")}  # bf16 bits, as int16 (numpy has no bf16)
        positions = torch.arange(length, device=self.device)[None].expand(BATCH, length)
        self.rope = net.model.rotary_emb.to(self.device)(torch.zeros(1, device=self.device, dtype=torch.bfloat16),
                                                         positions)
        blocks = len(net.model.layers)
        finished = next((b for b in range(blocks) if not self.block_file(b).exists()), blocks)
        self.fill(net)
        for b in range(finished):  # resuming: the caches move on through the blocks already trained
            self.advance(net.model.layers[b].to(self.device), self.restored(net, b, dict(np.load(self.block_file(b)))))
            net.model.layers[b].cpu()
        log = [{"block": b, "resumed": True} for b in range(finished)]
        for b in range(finished, blocks):
            log.append(self.train(net, b))
        part = out.with_name(out.name + ".part")
        shutil.copy(start, part)
        writer = gguf.GGUFReader(part, "r+")
        tensors = {t.name: t for t in writer.tensors}
        for b in range(blocks):
            for name, raw in np.load(self.block_file(b)).items():
                tensors[name].data[...] = raw.reshape(tensors[name].data.shape)
        writer.data.flush()
        del writer, tensors
        os.replace(part, out)
        return log

    def fill(self, net: torch.nn.Module) -> None:
        """The caches start as every sample's input to block 0: the original's word table on the original's path,
        the start file's word table on the quantized path. The validation samples are kept in memory the same way."""
        table = self.file["token_embd.weight"]
        quantized_table = torch.from_numpy(gguf.quants.dequantize(table.data, table.tensor_type)).to(
            self.device, torch.bfloat16)
        embed = net.model.embed_tokens.to(self.device)

        def inputs(ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
            ids = torch.from_numpy(ids.astype(np.int64)).to(self.device)
            return embed(ids), quantized_table[ids]
        with torch.no_grad():
            for i in range(0, len(self.tokens), BATCH):
                x, xq = inputs(self.tokens[i:i + BATCH])
                self.put("original", i, x)
                self.put("quantized", i, xq)
            pairs = [inputs(self.validation[i:i + BATCH]) for i in range(0, len(self.validation), BATCH)]
        self.val_original = torch.cat([p[0] for p in pairs]).cpu()
        self.val_quantized = torch.cat([p[1] for p in pairs]).cpu()
        embed.cpu()

    def advance(self, original: torch.nn.Module, trained: torch.nn.Module) -> None:
        """The caches and validation samples move on a block: through the original block on the original's path,
        through the trained block (as saved) on the quantized path."""
        with torch.no_grad():
            for i in range(0, len(self.tokens), BATCH):
                self.put("original", i, self.forward(original, self.get("original", i)))
                self.put("quantized", i, self.forward(trained, self.get("quantized", i)))
            self.val_original = self.through(original, self.val_original)
            self.val_quantized = self.through(trained, self.val_quantized)

    def train(self, net: torch.nn.Module, b: int) -> dict:
        """Trains block b, saves its bytes, and moves the caches on to the next block's inputs."""
        original = net.model.layers[b].to(self.device)
        block = copy.deepcopy(original).float()
        matrices = {}
        for module, role in MATRICES.items():
            name = f"blk.{b}.{role}.weight"
            parent, attribute = module.rsplit(".", 1)
            t = self.file[name]
            matrices[name] = QLinear(block.get_submodule(module), Grid(t.tensor_type.name), np.asarray(t.data),
                                     self.heads.get(module))
            setattr(block.get_submodule(parent), attribute, matrices[name])
        steps = EPOCHS * len(self.tokens) // BATCH
        optimizer = torch.optim.AdamW([{"params": [q.W], "lr": RATE * q.W.detach().pow(2).mean().sqrt().item()}
                                       for q in matrices.values()], weight_decay=0)
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: LOWEST + (1 - LOWEST) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

        def snapshot() -> list[torch.Tensor]:
            return [q.W.detach().clone() for q in matrices.values()]

        file_block = self.restored(net, b, {})
        file_error = self.validate(original, file_block)
        before = self.validate(original, block, matrices)
        best, best_step, best_weights, step = before, 0, snapshot(), 0
        for _ in range(EPOCHS):
            for i in range(0, len(self.tokens), BATCH):
                x, xq = self.get("original", i), self.get("quantized", i)
                with torch.no_grad(), torch.autocast(self.device, torch.bfloat16):
                    target = self.forward(original, x)
                with torch.autocast(self.device, torch.bfloat16):
                    loss = torch.nn.functional.mse_loss(self.forward(block, xq).float(), target.float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                schedule.step()
                step += 1
                if step % CHECK == 0:
                    error = self.validate(original, block, matrices)
                    if error < best:
                        best, best_step, best_weights = error, step, snapshot()
                    if step - best_step >= PATIENCE * CHECK:
                        break
            if step - best_step >= PATIENCE * CHECK:
                break
        error = self.validate(original, block, matrices)  # the end of training is a check too
        if error < best:
            best, best_step, best_weights = error, step, snapshot()

        with torch.no_grad():  # back to the best point, then scored from the bytes that are saved
            for q, w in zip(matrices.values(), best_weights, strict=True):
                q.W.copy_(w)
        saved = {name: q.saved() for name, q in matrices.items()}
        trained = self.restored(net, b, saved)
        after = self.validate(original, trained)
        kept_file = after >= file_error
        if kept_file:  # training did not beat the block the file already has: that one stays
            saved = {name: np.asarray(self.file[name].data) for name in matrices}
            trained, after = file_block, file_error
        np.savez(self.dir / f"block-{b}.part.npz", **saved)
        os.replace(self.dir / f"block-{b}.part.npz", self.block_file(b))

        self.advance(original, trained)
        net.model.layers[b] = original.cpu()
        del block, optimizer, trained, file_block
        torch.cuda.empty_cache()
        return {"block": b, "before": before, "after": after, "file": file_error, "kept_file": kept_file,
                "best_step": best_step}

    def restored(self, net: torch.nn.Module, b: int, saved: dict[str, np.ndarray]) -> torch.nn.Module:
        """Block b with its matrices decoded from saved bytes (the start file's where none are saved), in bf16."""
        block = copy.deepcopy(net.model.layers[b]).to(self.device)
        for module, role in MATRICES.items():
            name = f"blk.{b}.{role}.weight"
            t = self.file[name]
            values = gguf.quants.dequantize(saved.get(name, np.asarray(t.data)), t.tensor_type)
            if module in self.heads:
                values = unpermute(values, self.heads[module])
            weight = block.get_submodule(module).weight
            weight.data = torch.from_numpy(np.ascontiguousarray(values)).to(weight.device, weight.dtype)
        return block

    def validate(self, original: torch.nn.Module, block: torch.nn.Module, matrices: dict | None = None) -> float:
        """The block's mean squared error against the original block on the validation samples. For the block in
        training, its codes are picked afresh first: what a save from here would hold."""
        for q in (matrices or {}).values():
            q.codes = None
        errors = []
        with torch.no_grad(), torch.autocast(self.device, torch.bfloat16):
            for i in range(0, len(self.val_original), BATCH):
                target = self.forward(original, self.val_original[i:i + BATCH].to(self.device))
                got = self.forward(block, self.val_quantized[i:i + BATCH].to(self.device))
                errors.append(torch.nn.functional.mse_loss(got.float(), target.float()).item())
        return float(np.mean(errors))

    def through(self, block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """x (on the CPU) through a block, a batch at a time."""
        with torch.no_grad(), torch.autocast(self.device, torch.bfloat16):
            return torch.cat([self.forward(block, x[i:i + BATCH].to(self.device)).cpu()
                              for i in range(0, len(x), BATCH)])

    def forward(self, block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = block(x, attention_mask=None, position_embeddings=self.rope)
        return y[0] if isinstance(y, tuple) else y

    def get(self, cache: str, i: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.cache[cache][i:i + BATCH])).view(torch.bfloat16).to(self.device)

    def put(self, cache: str, i: int, x: torch.Tensor) -> None:
        self.cache[cache][i:i + len(x)] = x.to(torch.bfloat16).view(torch.int16).cpu().numpy()

    def block_file(self, b: int) -> Path:
        return self.dir / f"block-{b}.npz"
