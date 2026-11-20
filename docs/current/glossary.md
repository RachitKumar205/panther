# Glossary

| Term | Meaning |
|---|---|
| **type** | one of llama.cpp's 14 storage formats (IQ1_S … Q8_0, 1.56–8.5 bits/weight) |
| **type copy** | the whole model stored once in a single type |
| **genome** | one type assigned per weight matrix; its file is assembled from the type copies |
| **preset** | llama.cpp's ready-made mix; the competitor |
| **contest** | one model + one RAM budget, written `<size>@<budget>` (e.g. `14b@5`) |
| **score** | `flips% + 10 × KL`, lower is better |
| **relative score** | `ours ÷ preset × 100` (100 = the preset's damage, 0 = none, >100 = worse) |
| **flips** | share of benchmark questions whose right/wrong differs from the original |
| **KL** | mean KL divergence per token from the original's next-word probabilities |
| **same top word %** | share of positions where the file's most likely next word equals the original's |
| **train / select / test** | the three disjoint data splits (build/train, search/guard, verdict) |
| **selection / held-out** | the two disjoint benchmark-question sets |
| **claim** | a run's GPU and RAM reservation for its whole life |
| **store** | a model's 16-bit original, importance matrix and 14 type copies |
| **distillation** | training per-block scale multipliers and norms against the original's top-64 |
| **rounding** | error-compensated (GPTQ) quantization into llama.cpp's own types |
