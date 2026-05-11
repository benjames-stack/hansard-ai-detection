-- speech_ai_scores_sonnet_judge_v1.sql
--
-- Adds Sonnet 4.6 columns to public.speech_ai_scores. Used by the
-- `judge` subcommand with --save-as sonnet_judge.
--
-- Run on VPS:
--   sudo -u postgres psql -d <your_database> \
--     -f migrations/speech_ai_scores_sonnet_judge_v1.sql
--
-- Safe to re-run.

ALTER TABLE public.speech_ai_scores
    ADD COLUMN IF NOT EXISTS sonnet_judge_score      SMALLINT NULL,
    ADD COLUMN IF NOT EXISTS sonnet_judge_confidence TEXT     NULL,
    ADD COLUMN IF NOT EXISTS sonnet_judge_reason     TEXT     NULL;

CREATE INDEX IF NOT EXISTS speech_ai_scores_sonnet_judge_high_idx
    ON public.speech_ai_scores (speech_id)
    WHERE sonnet_judge_score >= 7;

GRANT SELECT, INSERT, UPDATE ON public.speech_ai_scores TO <your-readonly-user>;
