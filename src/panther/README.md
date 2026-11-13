# src/panther

| File | What |
|---|---|
| `model.py` | A model's matrices (from its vendored `config.json`), bytes per type, the RAM model, the budget's type menu |
| `llama.py` | llama.cpp runs (KL, perplexity, multiple choice, WinoGrande, GSM8K via llama-server), output parsing and checks, one JSON record per run in `data/results/<model>/` |
| `stats.py` | Accuracy, flips against the original, paired bootstrap, WIN/TIE/LOSE |
| `rounding.py` | Error-compensated rounding (GPTQ) into llama.cpp's types, written into the type copies in place |
| `distill.py` | Distillation of a finished file: one multiplier per block scale and the norm weights, trained to match the original's next-word probabilities, written back into the GGUF |
| `solver.py` | CP-SAT: one type per matrix inside a byte window, least predicted loss |
