"""Before a day-long EfficientQAT run on a new model: Block-AP on its first block(s) with a few samples, everything
in the container's /tmp (nothing kept). Checks that the model loads, that the file's row order is found, and how
each block's validation error moves over the whole (short) run: early stopping is off, so the curve is complete.

  ./panther exec /opt/venv/bin/python scripts/server/qat_trial.py 7 Mistral-22B-Instruct [samples=512] [blocks=2] [tokens=512] [weights lr, relative]

Needs the model's token samples (pipeline.py qatdata) and its winner at that budget (used as the starting file).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import pipeline as P
from panther import blockap


def main() -> int:
    P.BUDGET, model = int(sys.argv[1]), sys.argv[2]
    n, blocks = (int(sys.argv[3]) if len(sys.argv) > 3 else 512), (int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    length = int(sys.argv[5]) if len(sys.argv) > 5 else 512  # 512 x 512 x 2 copies fits the container's /tmp
    t = np.load(P.out(model, "qat", "tokens.npz"))
    win = json.loads(P.out(f"budget-{P.BUDGET}", f"winner-{model}{P.SUFFIX}.json").read_text())
    blockap.CHECK = max(2, n // 8)  # validation checks every n/8 steps (512 in a real run), so the trial shows them
    blockap.PATIENCE = 10**6  # no early stop: the whole curve
    if len(sys.argv) > 6:  # a (relative) learning rate for the weights other than the run's
        blockap.LR_W = float(sys.argv[6])
    t0 = time.time()
    blockap.train(P.copy_of(model, "BF16"), Path(win["file"]), t["block"][:n, :length], t["val"][:8, :length], Path("/tmp/trial.gguf"),
          Path("/tmp/trial-cache"), lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True), blocks=blocks)
    print(f"trial done in {(time.time() - t0) / 60:.1f} min ({n} samples x {length} tokens, {blocks} block(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
