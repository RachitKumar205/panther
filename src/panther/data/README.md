# The data (`panther.data`)

Part 0 of the pipeline. Every text and question the pipeline reads, built once from pinned sources. The definition
is `configs/dataset.yaml`; the dataset lives in `data/dataset/<digest of that file>/`, so a changed definition builds
a new dataset and never reuses an old one.

## The four sets

They never share a web document, a chat prompt or a question.

| Set | Used for | Holds |
|---|---|---|
| **GA set** | the search's fitness (part 4) | its text; the search's 900 questions (MMLU 300, HellaSwag 200, ARC 200, WinoGrande 200) |
| **winner set** | choosing the preset (part 2) and the winner (part 7) | its text; MMLU 1,000, HellaSwag 800, ARC 800, WinoGrande 800 |
| **distill set** | everything built or trained: the importance matrix and GPTQ (part 1, its web text), distillation (part 5) | text |
| **verdict set** | the verdict only, once per contest (part 7) | its text; MMLU 2,000, HellaSwag 2,000, ARC-Challenge all (1,172), WinoGrande all (1,267, less any llama.cpp cannot read), GSM8K 500 |

Plus **QAT's documents** (part 6): FineWeb-Edu text in no set.

A set's **text** is, for each model, that model's own greedy answers to the set's chat prompts (in its chat template),
then the set's web text. Nothing is trained on the GA, winner or verdict sets, and nothing chooses anything with the
verdict set.

## The sources (pinned to Hugging Face commits)

| Source | Revision | Used for |
|---|---|---|
| `HuggingFaceFW/fineweb-edu`, `sample/10BT/000_00000.parquet` | `87f09149` | web text |
| `OpenAssistant/oasst1` (Apache-2.0) | `fdf72ae0` | chat prompts |
| `cais/mmlu` (dev, validation, test) | `c30699e8` | questions; dev: the 5 worked examples per subject |
| `Rowan/hellaswag` (train, validation) | `218ec52e` | questions |
| `allenai/ai2_arc` ARC-Challenge (train, validation, test) | `210d026f` | questions |
| `allenai/winogrande` winogrande_xl (train, validation) | `01e74176` | questions |
| `openai/gsm8k` main (train, test) | `740312ad` | the verdict's GSM8K; 5 worked examples from train |
| `Qwen/Qwen2.5-1.5B-Instruct` tokenizer | `989aa798` | counting web tokens |

`manifest.json` records every downloaded file's SHA-256.

## How each part is made (`Dataset().build()`, network, no GPU)

1. **Questions.** For each benchmark:
   - **verdict**: a seeded sample of the official test split (all of ARC's and WinoGrande's);
   - **GA, then winner**: from the non-test splits (MMLU validation; HellaSwag train; ARC train + validation; WinoGrande
     train), one record per question, skipping any whose normalized text is in the **full** test split or the few-shot
     examples, in a seeded shuffle: the GA takes the first ones, the winner the next (capped at what is left);
   - WinoGrande questions llama.cpp's CSV reader cannot hold (a comma in an option, a double quote) are dropped;
   - few-shot: MMLU dev (5 per subject, inside every MMLU prompt); GSM8K: 5 worked examples from train.
   Each set's questions are written in llama.cpp's formats: a binary file for MMLU, HellaSwag and ARC (lm-eval's
   prompts; MMLU 5-shot), a CSV for WinoGrande, and JSON lines for GSM8K. **Every file is read back as llama.cpp
   reads it** and must give the same questions and answers.
2. **Web text.** Every FineWeb-Edu document is hashed (its id with the seed) into one of 5 shards: the four sets and
   QAT's. A set takes its shard's documents in hash order, whole, until its token target (distill 300k, GA 100k,
   winner 100k, verdict 150k, QAT 8.5M = (4,096 + 64) windows of 2,048), skipping documents outside 128–8,192 tokens
   and **documents that leak a benchmark question**: one holding 12 words in a row of any question (any set's, few-shot
   included), or the whole question of a question of 8 to 11 words. Questions under 8 words are too common to mean a
   leak ("Which of these is true?") and are not looked for. Because a document's set comes from its own id's hash,
   dropping one never moves another between sets.
3. **Chat prompts.** OpenAssistant's English first prompter messages, not deleted, review passed, 20–1,000
   characters; deduplicated; sorted, then a seeded shuffle; any of the first 1,200 sharing 8 words in a row with any
   benchmark question is dropped; the first 600 clean ones are cut into distill 300, GA 150, winner 150, verdict 150.
4. **The check** (fails the build loudly): no question in two sets (by normalized content; the few-shot examples may
   be in the verdict's test split, as MMLU's dev questions can be, but never in the GA's or the winner's); no chat
   prompt in two sets; no web document in two sets.
5. **The manifest**: counts, tokens, the sources' file hashes, and everything dropped and why.

Written in a `.part` folder and renamed when complete; skipped if done.

## One model's sets (`Dataset().for_model(model)`, GPU, no network)

- **Chat answers**: the original (the 16-bit file from part 1) answers every prompt by `llama-server` in its own chat
  template (`--jinja`), greedy, up to 256 tokens, 8 at once; the whole exchange is kept.
- **Each set's text**: its chat answers, then its web text: `distill.txt`, `ga.txt`, `winner.txt`, `verdict.txt`.
- **QAT's samples**: the QAT web text in the model's own tokens, cut into windows of 2,048: the first n (default
  1,024) for training, the next 64 for validation.

Made once, in `data/models/<model>/sets/<dataset>/`.

## On disk

```
data/dataset/<digest>/
  manifest.json
  web/{distill,ga,winner,verdict,qat}.txt, .ids.json      documents joined by a blank line; their ids
  prompts/{distill,ga,winner,verdict}.json
  questions/{ga,winner,verdict}/{mmlu,hellaswag,arc_challenge}.bin, winogrande.csv, <task>.jsonl (the records)
  questions/gsm8k-verdict.jsonl, gsm8k-examples.jsonl
data/models/<model>/sets/<digest>/
  chat.json, distill.txt, ga.txt, winner.txt, verdict.txt, qat-<n>.npz
```

## Decisions

| | Decision | Why | Before |
|---|---|---|---|
| 1 | The builders exist again, in code, from pinned sources | the data can be rebuilt and checked; on master they were deleted (87289e1) and the files existed only on the server | built once by scripts since removed |
| 2 | Four sets, with a **winner set** | choosing the preset and the winner on the search's questions would reward luck | train / select / test |
| 3 | The verdict's MMLU (2,000), HellaSwag (2,000) and GSM8K (500) are samples | a fast verdict | Rachit's `poc_test` subset (MMLU 1,000, HellaSwag 2,000, GSM8K 300) |
| 4 | Web documents leaking a benchmark question dropped (Lalit's check), with its two fixes: the questions' runs indexed once (it re-scanned per document), and stems under 8 words not looked for (a short common question dropped every document containing it) | no question's text can be learnt from the web text | none (Lalit's rewrite: the check, without the fixes) |
| 5 | Documents placed by their own id's hash | dropping one moves no other | Lalit's shuffle: one drop reshuffled most documents |
| 6 | The selection questions skip anything in the **full** test split | stronger than skipping only the sampled verdict questions | the same |
| 7 | Every llama.cpp file read back before it is used | a question llama.cpp would read differently never reaches a measurement | the same, in the removed builder |
| 8 | Each model answers the chat prompts itself | the text matches the model being measured | the same |

## Using it

```python
from panther.data import Dataset

ds = Dataset()
ds.build()                       # once, with the network
m.build(ds.web_distill)          # part 1
sets = ds.for_model(m)           # once per model, on the GPU
sets.ga.text, sets.ga.questions  # the search's fitness; sets.winner, sets.verdict alike
sets.distill, sets.gsm8k_questions, sets.gsm8k_examples, sets.qat_tokens, sets.qat_validation
```

## Files

| File | What |
|---|---|
| `dataset.py` | `Dataset` (`build`, `for_model`, and its steps: `questions`, `web`, `prompts`, `check`), `Hub` (the pinned sources), `ModelSets`, `Set` |
| `questions.py` | records from each source's rows, how each question is asked, llama.cpp's binary multiple-choice and WinoGrande CSV formats with their readers |

## Tests (`tests/data/`)

Built from tiny synthetic sources shaped like the real ones (no network):

- the sets' sizes; the verdict takes every question of a small test split; unreadable WinoGrande rows dropped; GSM8K's
  questions and examples;
- no question in two sets; web documents fill their sets and none is in two;
- documents leaking a benchmark question are dropped (planted in every 10th document) and counted; a short common
  stem drops nothing, a 9-word one found whole does; dropping documents moves no other between sets;
- chat prompts sharing 8 words with a question are dropped (one planted), none in two sets;
- every llama.cpp file reads back as written; the MMLU prompt holds its subject's 5 worked examples; WinoGrande rows
  round-trip and an unwritable one is refused;
- the check catches a question planted in two sets; a changed definition is a new dataset; a built one is not rebuilt;
- one model's sets on the tiny model (real `llama-server` in chat mode): its chat answers, then the web text; every
  question file; QAT's windows; made once.
