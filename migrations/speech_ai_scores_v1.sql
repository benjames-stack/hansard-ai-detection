-- speech_ai_scores_v1.sql
--
-- Intermediate-results table for the one-off AI-drafted speech detection
-- research analysis (see scripts/research_ai_detection.py).
--
-- Holds two complementary signals per speech so we can iterate on aggregation
-- and thresholds without re-running the expensive scoring passes:
--
--   * binoculars_score   — raw Binoculars detector score (lower => more
--                          AI-like). Pinned model versions in model_version.
--   * is_ai_at_low_fpr   — convenience boolean using the paper's low-FPR
--                          threshold (~0.901). Re-derivable from the raw score.
--   * stylo_features     — per-speech stylometric feature vector (TTR, MTLD,
--                          Yule's K, mean sentence length, function-word
--                          frequencies, em-dash density, AI-tell phrase counts,
--                          etc.). Stored as JSONB so the feature set can grow
--                          without a schema change.
--   * ai_phrase_count    — denormalised count of AI-tell regex hits, for fast
--                          aggregate queries that don't need the full vector.
--   * model_version      — pinned identifier for reproducibility, e.g.
--                          "binoculars@<commit-sha>+falcon-7b@<rev>".
--
-- No FK on speech_id: this lets us land scores from historical.speeches too if
-- the analysis later expands. Single-int speech-id namespace is enforced by
-- HISTORICAL_ID_OFFSET (see db_config.py).
--
-- Run on VPS:
--   sudo -u postgres psql -d <your_database> -f migrations/speech_ai_scores_v1.sql
--
-- Safe to re-run.

CREATE TABLE IF NOT EXISTS public.speech_ai_scores (
    speech_id         BIGINT      PRIMARY KEY,
    binoculars_score  REAL        NULL,
    is_ai_at_low_fpr  BOOLEAN     NULL,
    stylo_features    JSONB       NULL,
    ai_phrase_count   INTEGER     NULL,
    run_at            TIMESTAMPTZ DEFAULT now(),
    model_version     TEXT        NULL
);

COMMENT ON TABLE public.speech_ai_scores IS
    'Per-speech AI-likelihood and stylometric features for the one-off '
    'AI-detection research analysis. Not exposed via any public route. '
    'See scripts/research_ai_detection.py.';

-- Partial index for fast "flagged speeches" aggregation in Phase C.
CREATE INDEX IF NOT EXISTS speech_ai_scores_low_fpr_idx
    ON public.speech_ai_scores (speech_id)
    WHERE is_ai_at_low_fpr = TRUE;
