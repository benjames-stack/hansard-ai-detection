-- speech_ai_scores_fdgpt_v1.sql
--
-- Adds Fast-DetectGPT score columns to public.speech_ai_scores. Used by
-- the `load-fdgpt` subcommand of scripts/research_ai_detection.py to ingest
-- output from scripts/research_ai_fast_detect_gpt_modal.py.
--
-- Independent open-source second opinion alongside Binoculars (Phase A v1).
-- Columns:
--   * fdgpt_score      — raw Fast-DetectGPT analytic-sampling-discrepancy
--                        score (higher → more AI-like; signs are inverted
--                        relative to Binoculars).
--   * fdgpt_is_ai      — convenience boolean using the paper's default
--                        threshold (>0.0). Re-derivable from the raw score.
--
-- Run on VPS:
--   sudo -u postgres psql -d <your_database> \
--     -f migrations/speech_ai_scores_fdgpt_v1.sql
--
-- Safe to re-run.

ALTER TABLE public.speech_ai_scores
    ADD COLUMN IF NOT EXISTS fdgpt_score REAL    NULL,
    ADD COLUMN IF NOT EXISTS fdgpt_is_ai BOOLEAN NULL;

CREATE INDEX IF NOT EXISTS speech_ai_scores_fdgpt_flagged_idx
    ON public.speech_ai_scores (speech_id)
    WHERE fdgpt_is_ai = TRUE;

GRANT SELECT, INSERT, UPDATE ON public.speech_ai_scores TO <your-readonly-user>;
