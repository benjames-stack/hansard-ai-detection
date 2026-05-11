# hansard-ai-detection

Detection pipeline for evaluating AI authorship in the Australian
federal parliamentary record (Hansard).

The pipeline reads speech text from a Postgres database, scores each
speech with up to four independent detection methods, and writes the
scores back to a companion table for downstream analysis. The
scripts here are the analysis tooling only — the speech corpus
itself lives in a separate database, sourced from
[checkhansard.com.au](https://checkhansard.com.au).

## What's here

```
hansard-ai-detection/
├── README.md                  This file.
├── LICENSE                    MIT.
├── requirements.txt           Python dependencies.
├── db_config.example.py       Postgres connection-config template.
├── docs/
│   └── RUNBOOK.md             Step-by-step run guide.
├── migrations/
│   └── *.sql                  Schema migrations defining the
│                              speech_ai_scores table.
└── scripts/
    ├── research_ai_detection.py        Main driver (CLI subcommands
    │                                   for each stage of the pipeline).
    ├── research_ai_binoculars.py       Binoculars wrapper for a rented
    │                                   GPU host.
    ├── research_ai_binoculars_modal.py Binoculars wrapper for Modal.
    ├── research_ai_fast_detect_gpt_modal.py
    │                                   Fast-DetectGPT on Modal.
    └── research_ai_calibrate.py        Methodology sanity check: scores
                                        N known-AI speeches and N
                                        known-human speeches with the
                                        same judge.
```

## Detection methods

The pipeline applies four detection methods. Each is independent and
writes its scores to a different column on the `speech_ai_scores`
table. Any subset can be run.

1. **Binoculars** (Hans et al., ICML 2024). Statistical perplexity
   ratio between Falcon-7B base and Falcon-7B-instruct. Runs on a GPU
   via Modal or a rented host.

2. **Fast-DetectGPT** (Bao et al., 2024). Probability-curvature
   detector using GPT-Neo-2.7B. Runs on a GPU via Modal.

3. **Stylometric change-point per MP.** ~130 stylometric features per
   speech (TTR, MTLD, function-word frequencies, AI-tell phrase
   counts, etc.), aggregated to per-MP-per-quarter means, with
   Mahalanobis distance from each MP's pre-Nov-2022 baseline.
   `ruptures.Pelt` for breakpoint dating. CPU only.

4. **Claude as LLM-as-judge** via the Anthropic Batch API. Each
   speech is read individually and scored 0–10 with a structured
   JSON verdict. Three models supported via the `--save-as` flag:
   `llm_judge` (Haiku 4.5), `sonnet_judge` (Sonnet 4.6), `opus_judge`
   (Opus 4.7).

A calibration test (`scripts/research_ai_calibrate.py`) generates
synthetic AI speeches with Claude on real Hansard topics and scores
them against guaranteed-human pre-ChatGPT speeches, reporting
sensitivity and false-positive rate for the judge prompt.

## Database schema

The pipeline expects a `public.speeches` table with at minimum these
columns:

| Column | Type | Notes |
|---|---|---|
| `id` | bigint primary key | |
| `date` | date | sitting date |
| `chamber` | text | e.g. 'House of Reps' or 'Senate' |
| `source_type` | text | e.g. 'chamber' or 'committee' |
| `speech_text` | text | the speech body |
| `debate_title` | text | e.g. 'BILLS', 'CONDOLENCES' |
| `word_count` | int | |
| `name_id` | text | speaker identifier |
| `jurisdiction` | text | filtered to 'commonwealth' in queries |

A companion `public.members` table with `(name_id, display_name, party)`
is used by the report subcommand for joins.

The `speech_ai_scores` table that the pipeline writes to is created
by the migrations in `migrations/`.

## Quick start

```sh
git clone https://github.com/<your-username>/hansard-ai-detection.git
cd hansard-ai-detection

python -m venv .venv
.venv/bin/pip install -r requirements.txt

cp db_config.example.py db_config.py
# Edit db_config.py with your Postgres credentials.

export ANTHROPIC_API_KEY='sk-ant-api03-...'

# Apply migrations (one-time)
for m in migrations/*.sql; do
    psql -d your_database -f "$m"
done
```

## Running the pipeline

Each detector is independent and writes to its own column. Run any
subset.

**Binoculars:**

```sh
python scripts/research_ai_detection.py dump-speeches --output speeches.jsonl
modal run scripts/research_ai_binoculars_modal.py \
    --input-file speeches.jsonl --output-file scores.jsonl
python scripts/research_ai_detection.py load-binoculars --input scores.jsonl
```

**Fast-DetectGPT** (same shape, different Modal app and load
subcommand):

```sh
modal run scripts/research_ai_fast_detect_gpt_modal.py \
    --input-file speeches.jsonl --output-file scores_fdgpt.jsonl
python scripts/research_ai_detection.py load-fdgpt --input scores_fdgpt.jsonl
```

**Stylometric features + per-MP change-point:**

```sh
python scripts/research_ai_detection.py stylo-features
python scripts/research_ai_detection.py change-point
```

**Claude LLM-as-judge** (example: Sonnet 4.6 on the full post-Nov-2022
corpus):

```sh
python scripts/research_ai_detection.py judge \
    --model claude-sonnet-4-6 \
    --save-as sonnet_judge \
    --full-corpus --date-from 2022-11-30 --yes
```

**Methodology validation:**

```sh
python scripts/research_ai_calibrate.py --n 50
```

**Build the summary report:**

```sh
python scripts/research_ai_detection.py report
```

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for the full step-by-step.

## Caveats

- A flag indicates statistical or stylistic properties associated with
  AI generation. It does not tell you whether a speech was wholly
  AI-written, partially AI-polished, or merely AI-structured.
- Parliamentary prose is at the edge of what current detectors handle
  well. The older statistical detectors (Binoculars, Fast-DetectGPT)
  produce noisier results than the LLM-as-judge approach on this
  register.
- The scripts are tuned for the Australian federal Hansard schema. They
  will need adaptation if applied to a different corpus.

## Requirements

- Python 3.11+
- PostgreSQL 14+
- An Anthropic API key (for the LLM-as-judge components)
- A [Modal](https://modal.com) account (for the GPU-based detectors)

## License

MIT. See [LICENSE](LICENSE).
