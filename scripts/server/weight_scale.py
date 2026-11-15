"""The size of a model's layer weights by block (root mean square per matrix, from the 16-bit file): what a
learning rate that is an absolute step per weight has to be judged against.

  ./panther exec /opt/venv/bin/python scripts/server/weight_scale.py Mistral-22B-Instruct [blocks...]
"""

import sys

import gguf
import numpy as np

import pipeline as P

KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


def main() -> int:
    model, want = sys.argv[1], [int(b) for b in sys.argv[2:]]
    ts = {t.name: t for t in gguf.GGUFReader(P.copy_of(model, "BF16")).tensors}
    n = 1 + max(int(k.split(".")[1]) for k in ts if k.startswith("blk."))
    print("block " + " ".join(f"{k:>11}" for k in KINDS))
    for b in want or range(n):
        row = []
        for k in KINDS:
            t = ts[f"blk.{b}.{k}.weight"]
            w = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
            row.append(float(np.sqrt(np.mean(w * w))))
        print(f"{b:5d} " + " ".join(f"{v:11.5f}" for v in row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
