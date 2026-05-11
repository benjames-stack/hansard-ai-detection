-- speech_ai_scores_opus_judge_v1.sql
--
-- Adds a second LLM-as-judge slot for Claude Opus 4.7 (the most capable
-- model available at time of writing). Used by the `judge` subcommand of
-- scripts/research_ai_detection.py with --save-as opus_judge.
--
-- Keeps the existing Haiku data in llm_judge_* untouched; the report
-- renderer can show both side by side for an "even the strongest model
-- agrees" comparison in the analysis.
--
-- Run on VPS:
--   sudo -u postgres psql -d <your_database> \
--     -f migrations/speech_ai_scores_opus_judge_v1.sql
--
-- Safe to re-run.

ALTER TABLE public.speech_ai_scores
    ADD COLUMN IF NOT EXISTS opus_judge_score      SMALLINT NULL,
    ADD COLUMN IF NOT EXISTS opus_judge_confidence TEXT     NULL,
    ADD COLUMN IF NOT EXISTS opus_judge_reason     TEXT     NULL;

CREATE INDEX IF NOT EXISTS speech_ai_scores_opus_judge_high_idx
    ON public.speech_ai_scores (speech_id)
    WHERE opus_judge_score >= 7;

GRANT SELECT, INSERT, UPDATE ON public.speech_ai_scores TO <your-readonly-user>;
