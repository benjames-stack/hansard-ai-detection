# Run guide

Step-by-step for running the AI-detection pipeline against a Hansard-like
corpus. Assumes you've completed the Quick start in the README
(Postgres running, migrations applied, `db_config.py` filled in,
`ANTHROPIC_API_KEY` exported, Modal CLI authenticated where relevant).

The detectors are independent — run any subset.

---

## 1. Sanity-check the corpus

```sh
.venv/bin/python scripts/research_ai_detection.py volume
```

Read-only report on how many speeches are in scope, the
pre/post-ChatGPT split, median speeches per MP per quarter, and
per-MP baseline coverage. Writes to `reports/ai_detection_volume_<date>.md`.

---

## 2. Stylometric feature extraction

CPU-only, resumable. Use `tmux` or `nohup` so an SSH disconnect doesn't
kill it.

```sh
.venv/bin/python scripts/research_ai_detection.py stylo-features
```

Streams progress every ~5000 speeches. Sanity check after:

```sh
psql -d $YOUR_DB -c \
    "SELECT COUNT(*), COUNT(stylo_features) FROM public.speech_ai_scores;"
```

The two numbers should match.

---

## 3. Binoculars on Modal

### Dump speeches

```sh
.venv/bin/python scripts/research_ai_detection.py dump-speeches \
    --output /tmp/speeches.jsonl
```

### Smoke test (always)

```sh
modal run scripts/research_ai_binoculars_modal.py \
    --input-file /tmp/speeches.jsonl \
    --output-file /tmp/scores_smoke.jsonl \
    --limit 1000
```

First run downloads ~14 GB of Falcon weights to a Modal Volume
(one-time). Subsequent runs skip it. Manually spot-check 10 flagged
and 10 unflagged speeches before the full run.

### Full run

```sh
modal run --detach scripts/research_ai_binoculars_modal.py \
    --input-file /tmp/speeches.jsonl \
    --output-file /tmp/scores.jsonl
```

### Load scores

```sh
.venv/bin/python scripts/research_ai_detection.py load-binoculars \
    --input /tmp/scores.jsonl
```

---

## 4. Fast-DetectGPT on Modal

Same pattern, lighter weight (~half the GPU time of Binoculars):

```sh
modal run --detach scripts/research_ai_fast_detect_gpt_modal.py \
    --input-file /tmp/speeches.jsonl \
    --output-file /tmp/scores_fdgpt.jsonl

.venv/bin/python scripts/research_ai_detection.py load-fdgpt \
    --input /tmp/scores_fdgpt.jsonl
```

---

## 5. Per-MP stylometric change-point

```sh
.venv/bin/python scripts/research_ai_detection.py change-point
```

Writes `reports/ai_detection_per_mp_<date>.csv`. Note this analysis
is sensitive to election-driven regime changes — a chamber-wide flip
from opposition to government produces a stylistic shift indistinguishable
from AI adoption at the per-MP level. Read the caveat comments in
`research_ai_detection.py` before relying on the output.

---

## 6. Claude LLM-as-judge

Submit a Batch API job via the `judge` subcommand. Three target column
prefixes corresponding to three models:

| `--save-as` | Model | Use |
|---|---|---|
| `llm_judge` | Claude Haiku 4.5 | Cheap exploratory run on a sample |
| `sonnet_judge` | Claude Sonnet 4.6 | Full-corpus production run |
| `opus_judge` | Claude Opus 4.7 | Small-sample high-confidence check |

Full post-Nov-2022 corpus on Sonnet:

```sh
.venv/bin/python scripts/research_ai_detection.py judge \
    --model claude-sonnet-4-6 \
    --save-as sonnet_judge \
    --full-corpus \
    --date-from 2022-11-30 \
    --yes
```

Batch IDs are saved to `reports/judge_batch_ids_<date>.txt`. If your
shell dies mid-poll, resume with:

```sh
.venv/bin/python scripts/research_ai_detection.py judge \
    --save-as sonnet_judge \
    --batch-id reports/judge_batch_ids_<date>.txt
```

---

## 7. Methodology validation

```sh
.venv/bin/python scripts/research_ai_calibrate.py --n 50
```

Generates N synthetic AI speeches with Claude on real Hansard topics
and scores both human and AI sets with the production judge prompt.
Prints histograms, sensitivity, and false-positive rate.

---

## 8. Report

```sh
.venv/bin/python scripts/research_ai_detection.py report
```

Writes a Markdown summary with the cross-detector comparison and
per-MP change-point details to `reports/ai_detection_<date>.md`.
