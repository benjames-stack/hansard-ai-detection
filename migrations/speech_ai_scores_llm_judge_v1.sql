-- speech_ai_scores_llm_judge_v1.sql
--
-- Adds LLM-as-judge columns to public.speech_ai_scores for the AI-drafted
-- speech detection research analysis. Used by the `judge` subcommand of
-- scripts/research_ai_detection.py — Claude (Haiku 4.5 by default) reads each
-- speech via the Anthropic Batch API and emits a structured 0-10 verdict.
--
-- Columns:
--   * llm_judge_score        — 0-10 ordinal: 0-2 clearly human, 8-10 likely AI.
--   * llm_judge_confidence   — 'low' | 'medium' | 'high'
--   * llm_judge_reason       — short free-text rationale (<= ~100 chars)
--   * llm_judge_model        — pinned model identifier (e.g.
--                              'claude-haiku-4-5-20251001').
--
-- Run on VPS:
--   sudo -u postgres psql -d <your_database> \
--     -f migrations/speech_ai_scores_llm_judge_v1.sql
--
-- Safe to re-run.

ALTER TABLE public.speech_ai_scores
    ADD COLUMN IF NOT EXISTS llm_judge_score      SMALLINT NULL,
    ADD COLUMN IF NOT EXISTS llm_judge_confidence TEXT     NULL,
    ADD COLUMN IF NOT EXISTS llm_judge_reason     TEXT     NULL,
    ADD COLUMN IF NOT EXISTS llm_judge_model      TEXT     NULL;

-- Partial index for fast aggregation of LLM-flagged speeches in Phase C.
-- Threshold ≥ 7 = "possibly AI-assisted or likely AI-drafted".
CREATE INDEX IF NOT EXISTS speech_ai_scores_llm_judge_high_idx
    ON public.speech_ai_scores (speech_id)
    WHERE llm_judge_score >= 7;

GRANT SELECT, INSERT, UPDATE ON public.speech_ai_scores TO <your-readonly-user>;
