#!/usr/bin/env python3
"""Driver for the one-off AI-drafted speech detection research analysis.

This is a research/analysis script — output is a markdown report plus CSVs
under reports/, NOT a productionised feature. See
the project README for the full plan.

Subcommands:

  volume          Step 0: read-only volume sanity-check report.
  stylo-features  Phase B step 1: extract per-speech stylometric features
                  and write them to public.speech_ai_scores.stylo_features.
                  Resumable; only processes speeches not already scored.
  change-point    Phase B step 2: aggregate features per (MP, quarter), fit
                  per-MP baseline, flag MPs with sustained post-Nov-2022
                  shift. Outputs CSV under reports/.
  report          Phase C: cross-check Binoculars + stylometric, render
                  charts and final markdown report.

Phase A (Binoculars on rented GPU) is not part of this driver — see
scripts/research_ai_binoculars.py for the GPU-host wrapper. Once that's run
externally and JSONL scores are scp'd back, this driver loads them via the
`load-binoculars` subcommand.

Usage (read-only volume report; safe to run repeatedly):

    GR_DB_PASSWORD=… GR_DB_NAME=… GR_DB_USER=… \\
    META_DB_PASSWORD=… META_DB_NAME=… META_DB_USER=… \\
        .venv/bin/python scripts/research_ai_detection.py volume

Phase B feature extraction (writes to DB):

    GR_DB_STATEMENT_TIMEOUT_MS=0 \\
    GR_DB_PASSWORD=… GR_DB_NAME=… GR_DB_USER=… \\
        .venv/bin/python scripts/research_ai_detection.py stylo-features
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_config import DB_CONFIG  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

# Scope of the analysis. Federal only; substantive speeches only.
JURISDICTION = "commonwealth"
MIN_WORD_COUNT = 200
START_DATE = "2018-01-01"  # baseline window starts here; ChatGPT launch Nov 2022
CHATGPT_LAUNCH = "2022-11-30"
MIN_SPEECHES_PER_QUARTER = 5  # per-MP-quarter inclusion threshold for Phase B

# Stylometric feature-extractor version. Bump when feature definitions change
# so we know to re-extract. Stored in speech_ai_scores.model_version alongside
# Binoculars version when both phases have run.
STYLO_VERSION = "stylo-v1"

# Resumable batch sizing for Phase B feature extraction.
STYLO_FETCH_BATCH = 2000      # rows fetched from DB per batch
STYLO_WRITE_BATCH = 500       # rows written per execute_values call
STYLO_WORKERS = max(1, (os.cpu_count() or 4) - 1)
STYLO_LOG_INTERVAL = 5000     # progress log every N speeches


def connect():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        print(
            "ERROR: could not connect to Postgres.\n"
            f"  host={DB_CONFIG['host']} port={DB_CONFIG['port']} "
            f"dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']}\n"
            f"  underlying error: {exc}".rstrip(),
            file=sys.stderr,
        )
        sys.exit(2)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def fetchone(cur, sql, params=None):
    cur.execute(sql, params or ())
    return cur.fetchone()


def fetchall(cur, sql, params=None):
    cur.execute(sql, params or ())
    return cur.fetchall()


def _f(n):
    if n is None:
        return "-"
    if isinstance(n, int):
        return f"{n:,}"
    if isinstance(n, float):
        return f"{n:,.1f}"
    return str(n)


def section_scope(now_utc) -> str:
    return (
        f"Generated {now_utc:%Y-%m-%d %H:%M UTC} against "
        f"`{DB_CONFIG['dbname']}` at `{DB_CONFIG['host']}:{DB_CONFIG['port']}` "
        "(read-only).\n\n"
        f"**Scope:** `public.speeches WHERE jurisdiction='{JURISDICTION}' "
        f"AND word_count > {MIN_WORD_COUNT} AND date >= '{START_DATE}'`. "
        f"ChatGPT launched {CHATGPT_LAUNCH}; speeches before that act as the "
        "per-MP stylometric baseline; speeches after are what's being scored."
    )


def section_by_year(cur) -> tuple[str, int]:
    rows = fetchall(
        cur,
        """
        SELECT EXTRACT(YEAR FROM date)::int AS yr,
               COUNT(*)::bigint AS n_speeches,
               COUNT(DISTINCT name_id)::bigint AS n_mps,
               AVG(word_count)::float AS mean_words,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY word_count)::float AS median_words,
               SUM(word_count)::bigint AS total_words
        FROM public.speeches
        WHERE jurisdiction = %s
          AND word_count > %s
          AND date >= %s
        GROUP BY 1
        ORDER BY 1
        """,
        (JURISDICTION, MIN_WORD_COUNT, START_DATE),
    )
    lines = [
        "| Year | Speeches | Distinct MPs | Mean words | Median words | Total words |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    grand_n = 0
    grand_words = 0
    for yr, n, mps, mean_w, med_w, total_w in rows:
        grand_n += n or 0
        grand_words += total_w or 0
        lines.append(
            f"| {yr} | {_f(n)} | {_f(mps)} | {_f(mean_w)} | {_f(med_w)} | {_f(total_w)} |"
        )
    lines.append(
        f"| **Total** | **{_f(grand_n)}** | — | — | — | **{_f(grand_words)}** |"
    )
    return "\n".join(lines), grand_n


def section_pre_post_split(cur) -> str:
    """Eligible counts split at ChatGPT launch — drives Phase A GPU sizing."""
    row = fetchone(
        cur,
        """
        SELECT
          COUNT(*) FILTER (WHERE date < %s)::bigint AS pre,
          COUNT(*) FILTER (WHERE date >= %s)::bigint AS post,
          SUM(word_count) FILTER (WHERE date < %s)::bigint AS pre_words,
          SUM(word_count) FILTER (WHERE date >= %s)::bigint AS post_words
        FROM public.speeches
        WHERE jurisdiction = %s
          AND word_count > %s
          AND date >= %s
        """,
        (
            CHATGPT_LAUNCH,
            CHATGPT_LAUNCH,
            CHATGPT_LAUNCH,
            CHATGPT_LAUNCH,
            JURISDICTION,
            MIN_WORD_COUNT,
            START_DATE,
        ),
    )
    pre, post, pre_words, post_words = row
    total = (pre or 0) + (post or 0)
    total_words = (pre_words or 0) + (post_words or 0)
    # Rough Voyage/HF tokenisation: ~1.3 tokens/word for English.
    est_tokens = int(total_words * 1.3) if total_words else 0
    # Binoculars runs each text through TWO models; doubles effective token volume.
    est_binoc_tokens = est_tokens * 2

    lines = [
        "| Window | Speeches | Words | Notes |",
        "| --- | ---: | ---: | --- |",
        f"| Pre-ChatGPT (`{START_DATE}` → `{CHATGPT_LAUNCH}`) | {_f(pre)} | {_f(pre_words)} | per-MP stylometric baseline |",
        f"| Post-ChatGPT (`{CHATGPT_LAUNCH}` → today) | {_f(post)} | {_f(post_words)} | what we're scoring |",
        f"| **Total** | **{_f(total)}** | **{_f(total_words)}** | |",
        "",
        "**GPU sizing estimate (Phase A — Binoculars):**",
        "",
        f"- ~{_f(est_tokens)} word-tokens of source text (≈1.3 tokens/word).",
        f"- Binoculars runs each speech through two ~7B models, so effective "
        f"throughput is over ~{_f(est_binoc_tokens)} tokens.",
        "- At ~30–60 speeches/sec on an A6000, full corpus is ~"
        f"{(total or 0) / 45 / 3600:.1f}–{(total or 0) / 30 / 3600:.1f} GPU-hrs.",
    ]
    return "\n".join(lines)


def section_per_mp_quarter(cur) -> str:
    """Distribution of speeches-per-MP-per-quarter — drives the
    MIN_SPEECHES_PER_QUARTER threshold for Phase B change-point detection."""
    rows = fetchall(
        cur,
        """
        WITH per_quarter AS (
            SELECT name_id,
                   date_trunc('quarter', date) AS q,
                   COUNT(*)::bigint AS n
            FROM public.speeches
            WHERE jurisdiction = %s
              AND word_count > %s
              AND date >= %s
              AND name_id IS NOT NULL
            GROUP BY 1, 2
        )
        SELECT COUNT(*)::bigint AS n_buckets,
               MIN(n)::bigint AS min_n,
               percentile_cont(0.50) WITHIN GROUP (ORDER BY n)::float AS p50,
               percentile_cont(0.90) WITHIN GROUP (ORDER BY n)::float AS p90,
               percentile_cont(0.99) WITHIN GROUP (ORDER BY n)::float AS p99,
               MAX(n)::bigint AS max_n,
               AVG(n)::float AS mean_n,
               COUNT(*) FILTER (WHERE n >= %s)::bigint AS buckets_above_threshold
        FROM per_quarter
        """,
        (JURISDICTION, MIN_WORD_COUNT, START_DATE, MIN_SPEECHES_PER_QUARTER),
    )
    n_buckets, min_n, p50, p90, p99, max_n, mean_n, above = rows[0]

    pct_above = (above / n_buckets * 100) if n_buckets else 0.0

    lines = [
        f"_Each row = one (MP, year-quarter) pair across the full {START_DATE}+ window._",
        "",
        f"- Total (MP, quarter) buckets: **{_f(n_buckets)}**",
        f"- Min / Mean / Max speeches per bucket: {_f(min_n)} / {_f(mean_n)} / {_f(max_n)}",
        f"- p50 / p90 / p99: **{_f(p50)}** / {_f(p90)} / {_f(p99)}",
        f"- Buckets with ≥ {MIN_SPEECHES_PER_QUARTER} speeches "
        f"(usable for Phase B): **{_f(above)} ({pct_above:.1f}%)**",
        "",
        f"_Threshold sanity-check: if median bucket size is well below "
        f"{MIN_SPEECHES_PER_QUARTER}, raise it to ~3 and document the caveat. "
        f"Backbenchers will have unstable Mahalanobis distance with too few samples._",
    ]
    return "\n".join(lines)


def section_baseline_coverage(cur) -> str:
    """How many MPs have enough pre-ChatGPT data to compute a stylometric
    baseline? Plan requires ≥4 quarters with ≥5 speeches each."""
    rows = fetchall(
        cur,
        """
        WITH per_quarter AS (
            SELECT name_id,
                   date_trunc('quarter', date) AS q,
                   COUNT(*)::bigint AS n
            FROM public.speeches
            WHERE jurisdiction = %s
              AND word_count > %s
              AND date >= %s
              AND date < %s
              AND name_id IS NOT NULL
            GROUP BY 1, 2
        ),
        usable_quarters AS (
            SELECT name_id, COUNT(*)::int AS n_quarters
            FROM per_quarter
            WHERE n >= %s
            GROUP BY 1
        )
        SELECT
          (SELECT COUNT(DISTINCT name_id) FROM per_quarter)::bigint AS mps_with_any_pre,
          (SELECT COUNT(*) FROM usable_quarters WHERE n_quarters >= 4)::bigint AS mps_with_baseline,
          (SELECT COUNT(*) FROM usable_quarters WHERE n_quarters >= 8)::bigint AS mps_with_strong_baseline
        """,
        (
            JURISDICTION,
            MIN_WORD_COUNT,
            START_DATE,
            CHATGPT_LAUNCH,
            MIN_SPEECHES_PER_QUARTER,
        ),
    )
    any_pre, baseline, strong = rows[0]
    lines = [
        "| Cohort | MPs |",
        "| --- | ---: |",
        f"| Any pre-ChatGPT speeches in scope | {_f(any_pre)} |",
        f"| ≥4 usable quarters (≥{MIN_SPEECHES_PER_QUARTER} speeches each) — eligible for change-point | **{_f(baseline)}** |",
        f"| ≥8 usable quarters — strong baseline | {_f(strong)} |",
        "",
        "_MPs without a sufficient baseline are excluded from Phase B "
        "change-point flagging but are still included in Phase A "
        "(Binoculars) corpus-wide aggregates._",
    ]
    return "\n".join(lines)


def section_party_breakdown(cur) -> str:
    rows = fetchall(
        cur,
        """
        SELECT COALESCE(m.party, '(unknown)') AS party,
               COUNT(*)::bigint AS n_speeches,
               COUNT(DISTINCT s.name_id)::bigint AS n_mps
        FROM public.speeches s
        LEFT JOIN public.members m ON m.name_id = s.name_id
        WHERE s.jurisdiction = %s
          AND s.word_count > %s
          AND s.date >= %s
        GROUP BY 1
        ORDER BY n_speeches DESC
        LIMIT 20
        """,
        (JURISDICTION, MIN_WORD_COUNT, START_DATE),
    )
    lines = [
        "_Top 20 by speech count. Party labels are raw `members.party`; "
        "Phase C will normalise via `trends.NORMALISED_PARTY_TO_CODES`._",
        "",
        "| Party | Speeches | MPs |",
        "| --- | ---: | ---: |",
    ]
    for party, n, mps in rows:
        lines.append(f"| {party} | {_f(n)} | {_f(mps)} |")
    return "\n".join(lines)


def generate_volume_report(cur) -> str:
    now_utc = fetchone(cur, "SELECT now() AT TIME ZONE 'UTC'")[0]

    out = []
    out.append(f"# AI-detection volume sanity-check — {date.today().isoformat()}")
    out.append("")
    out.append(section_scope(now_utc))
    out.append("")

    out.append("## Eligible speeches by year")
    out.append("")
    by_year_md, grand_n = section_by_year(cur)
    out.append(by_year_md)
    out.append("")

    out.append("## Pre/post ChatGPT split")
    out.append("")
    out.append(section_pre_post_split(cur))
    out.append("")

    out.append("## Speeches per MP per quarter")
    out.append("")
    out.append(section_per_mp_quarter(cur))
    out.append("")

    out.append("## Per-MP stylometric baseline coverage")
    out.append("")
    out.append(section_baseline_coverage(cur))
    out.append("")

    out.append("## Party breakdown (top 20)")
    out.append("")
    out.append(section_party_breakdown(cur))
    out.append("")

    out.append("---")
    out.append("")
    out.append(
        f"_Total in-scope speeches: **{_f(grand_n)}**. "
        "Phase C (final report) is not yet implemented — see plan at "
        "the methodology section in the README._"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Phase B step 1 — stylometric feature extraction
# ---------------------------------------------------------------------------
#
# Runs on CPU on the VPS. For each in-scope speech compute ~120 numeric
# features and store as JSONB in public.speech_ai_scores.stylo_features.
# Resumable: speeches that already have stylo_features set are skipped.
#
# Tokenisation is deliberately stdlib-only (regex). NLTK adds 200MB+ of data
# files for marginal accuracy gains over a regex sentence splitter on
# parliamentary register, which is well-formed prose.

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")

# Standard top-100 English function-word list (subset of NLTK stopwords +
# common pronouns/aux verbs). Used as a 100-dim per-speech style vector —
# function-word frequencies are the canonical stylometric primitive (see
# Mosteller & Wallace 1964; Federalist Papers).
FUNCTION_WORDS = (
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "it",
    "for", "on", "with", "as", "be", "by", "this", "are", "have", "but",
    "not", "they", "or", "had", "from", "at", "we", "an", "which", "their",
    "you", "he", "she", "his", "her", "i", "my", "me", "our", "us",
    "would", "will", "can", "could", "should", "may", "might", "must", "shall",
    "do", "does", "did", "has", "been", "being", "were",
    "if", "when", "while", "because", "although", "though", "since", "until",
    "before", "after", "where", "why", "how", "what", "who", "whose",
    "all", "any", "some", "no", "more", "most", "other", "such", "only", "own",
    "same", "so", "than", "too", "very", "just", "now", "then", "there",
    "here", "up", "out", "into", "about", "over", "under", "between", "through",
)
assert len(FUNCTION_WORDS) == 100, f"expected 100 function words, got {len(FUNCTION_WORDS)}"

# Subordinating conjunctions — fraction of these tokens / total tokens is a
# classic syntactic-complexity proxy.
SUBORDINATING_CONJUNCTIONS = frozenset({
    "because", "although", "while", "when", "since", "if", "unless", "until",
    "after", "before", "though", "whereas", "wherever", "whenever", "whether",
    "that", "which", "who", "as",
})

# AI-tell phrases. Counted with case-insensitive whole-word regex. Each
# label becomes a feature key `tell_<label>`. The list is intentionally
# narrow — high-precision tells, not stylistic guesses. "I hope this helps"
# is extremely rare in Hansard but a near-perfect ChatGPT signature when
# it slips through.
AI_TELL_PATTERNS = [
    ("delve", re.compile(r"\bdelv(?:e|ed|ing|es)\b", re.I)),
    ("tapestry", re.compile(r"\btapestry\b", re.I)),
    ("realm", re.compile(r"\brealm\b", re.I)),
    ("navigate", re.compile(r"\bnavigat(?:e|ed|ing|es)\b(?:\s+the\s+\w+)?", re.I)),
    ("ever_evolving", re.compile(r"\bever[\s-]evolving\b", re.I)),
    ("in_conclusion", re.compile(r"\bin\s+conclusion\b", re.I)),
    ("important_to_note", re.compile(r"\bit\s+is\s+important\s+to\s+note\b", re.I)),
    ("robust", re.compile(r"\brobust\b", re.I)),
    ("leverage", re.compile(r"\bleverag(?:e|ed|ing|es)\b", re.I)),
    ("paramount", re.compile(r"\bparamount\b", re.I)),
    ("unwavering", re.compile(r"\bunwavering\b", re.I)),
    ("foster", re.compile(r"\bfoster(?:s|ed|ing)?\b", re.I)),
    ("holistic", re.compile(r"\bholistic\b", re.I)),
    ("multifaceted", re.compile(r"\bmultifacet(?:ed|s)?\b", re.I)),
    ("hope_this_helps", re.compile(r"\bI\s+hope\s+this\s+helps\b", re.I)),
    ("furthermore", re.compile(r"\bfurthermore\b", re.I)),
    ("moreover", re.compile(r"\bmoreover\b", re.I)),
    ("crucial", re.compile(r"\bcrucial\b", re.I)),
]

# Acknowledgement-of-country boilerplate — strip before tokenisation since
# it's identical across thousands of speeches and would skew TTR/MTLD low.
ACK_OF_COUNTRY_RE = re.compile(
    r"I\s+(?:would\s+like\s+to\s+|wish\s+to\s+)?acknowledge\s+the\s+"
    r"traditional\s+(?:owners|custodians)[^.]*\.",
    re.I,
)


def _strip_boilerplate(text: str) -> str:
    """Remove acknowledgements of country and trim. Conservative — only
    obvious boilerplate; doesn't try to strip block quotes (see plan
    caveats)."""
    return ACK_OF_COUNTRY_RE.sub("", text).strip()


def _ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).
    More robust than TTR to text length. Walks the token stream and counts
    "factors": each factor ends when running TTR drops to `threshold`. MTLD
    = total tokens / number of factors."""
    if len(tokens) < 50:
        return 0.0  # MTLD is undefined / unstable on very short texts.

    def _walk(seq):
        types = set()
        token_count = 0
        factors = 0.0
        for tok in seq:
            types.add(tok)
            token_count += 1
            ttr = len(types) / token_count
            if ttr <= threshold:
                factors += 1
                types.clear()
                token_count = 0
        if token_count > 0:
            # Partial factor — interpolate.
            ttr = len(types) / token_count if token_count else 1.0
            factors += (1 - ttr) / (1 - threshold) if ttr < 1 else 0.0
        return len(seq) / factors if factors > 0 else float(len(seq))

    return (_walk(tokens) + _walk(list(reversed(tokens)))) / 2


def _yules_k(tokens: list[str]) -> float:
    """Yule's K: vocabulary-richness measure independent of text length.
    K = 10000 * (sum(V_i * i^2) - N) / N^2  where V_i = number of types
    appearing exactly i times. Lower K = richer vocabulary."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    freq_of_freq = Counter(counts.values())
    n = len(tokens)
    s = sum(v * (i ** 2) for i, v in freq_of_freq.items())
    return 10000.0 * (s - n) / (n * n) if n else 0.0


def compute_stylo_features(speech_text: str) -> dict:
    """Return a flat dict of ~120 numeric stylometric features.

    All features are normalised (per-token or per-1000-words) so they're
    comparable across speeches of different lengths.
    """
    text = _strip_boilerplate(speech_text or "")
    tokens_raw = WORD_RE.findall(text)
    tokens = [t.lower() for t in tokens_raw]
    n_tokens = len(tokens)

    # Sentences. Empty filter handles trailing/multiple punctuation.
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    n_sentences = max(1, len(sentences))

    counts = Counter(tokens)

    feats: dict[str, float] = {}

    feats["n_tokens"] = float(n_tokens)
    feats["n_sentences"] = float(n_sentences)

    # Lexical diversity.
    feats["ttr"] = _ttr(tokens)
    feats["mtld"] = _mtld(tokens)
    feats["yules_k"] = _yules_k(tokens)

    # Word/sentence length.
    feats["mean_word_length"] = (
        sum(len(t) for t in tokens) / n_tokens if n_tokens else 0.0
    )
    feats["mean_sentence_length"] = n_tokens / n_sentences

    # Syntactic / function-word.
    feats["subord_conj_frac"] = (
        sum(counts[w] for w in SUBORDINATING_CONJUNCTIONS) / n_tokens
        if n_tokens else 0.0
    )

    # Punctuation (per 1000 chars / per 1000 tokens, depending on what's stable).
    text_len = max(1, len(text))
    n_words_for_per_k = max(1, n_tokens) / 1000.0
    feats["em_dash_per_1k"] = text.count("—") / n_words_for_per_k
    feats["en_dash_per_1k"] = text.count("–") / n_words_for_per_k
    feats["semicolon_per_1k"] = text.count(";") / n_words_for_per_k
    feats["colon_per_1k"] = text.count(":") / n_words_for_per_k
    feats["paren_open_per_1k"] = text.count("(") / n_words_for_per_k
    feats["comma_per_sentence"] = text.count(",") / n_sentences
    # Quotes are tricky (curly vs straight); collapse all variants.
    n_quote_chars = (
        text.count('"') + text.count('“') + text.count('”')
        + text.count("'") + text.count('‘') + text.count('’')
    )
    feats["quote_chars_per_1k"] = n_quote_chars / n_words_for_per_k

    # Function-word frequencies — 100 features, each is freq per total
    # tokens. Stable cross-text stylometric primitive.
    for fw in FUNCTION_WORDS:
        feats[f"fw_{fw}"] = counts.get(fw, 0) / n_tokens if n_tokens else 0.0

    # AI-tell regex hits (per 1000 words).
    total_tells = 0
    for label, pat in AI_TELL_PATTERNS:
        n_hits = len(pat.findall(text))
        total_tells += n_hits
        feats[f"tell_{label}"] = n_hits / n_words_for_per_k
    feats["tell_total"] = float(total_tells)

    # Sentence-length variance (clipped) — formal/AI text often has low
    # variance compared to extemporaneous speech.
    if len(sentences) >= 3:
        slens = [len(WORD_RE.findall(s)) for s in sentences]
        mean = sum(slens) / len(slens)
        var = sum((x - mean) ** 2 for x in slens) / len(slens)
        feats["sentence_length_variance"] = var
    else:
        feats["sentence_length_variance"] = 0.0

    return feats


def _worker_compute(args: tuple[int, str]) -> tuple[int, dict, int]:
    """Multiprocessing worker: compute features for one speech.
    Returns (speech_id, features_dict, tell_total). Top-level for picklability.
    """
    speech_id, speech_text = args
    feats = compute_stylo_features(speech_text)
    return speech_id, feats, int(feats.get("tell_total", 0))


def cmd_stylo_features(args) -> int:
    """Phase B step 1: compute and persist per-speech stylometric features."""
    if os.environ.get("GR_DB_STATEMENT_TIMEOUT_MS", "30000") != "0":
        print(
            "WARNING: GR_DB_STATEMENT_TIMEOUT_MS is not 0. The 30s default "
            "may abort large COUNT/SELECT queries on the speech corpus. "
            "Re-run with `GR_DB_STATEMENT_TIMEOUT_MS=0`.",
            file=sys.stderr,
        )

    # Two separate connections: a server-side cursor on the read connection
    # would be invalidated by every commit on a shared connection. Splitting
    # them keeps the long-running SELECT alive while the write side commits
    # batches independently.
    read_conn = psycopg2.connect(**DB_CONFIG)
    # Server-side cursors require an open transaction, so we explicitly do
    # NOT autocommit here. read-only protects against accidental writes.
    # We never commit/rollback this connection during the run; it's
    # implicitly rolled back at .close() (which is a no-op for read-only).
    read_conn.set_session(readonly=True, autocommit=False)
    write_conn = psycopg2.connect(**DB_CONFIG)
    write_conn.set_session(autocommit=False)
    try:
        with read_conn.cursor() as cur:
            # How many speeches still need features?
            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.speeches s
                LEFT JOIN public.speech_ai_scores a ON a.speech_id = s.id
                WHERE s.jurisdiction = %s
                  AND s.word_count > %s
                  AND s.date >= %s
                  AND (a.stylo_features IS NULL)
                """,
                (JURISDICTION, MIN_WORD_COUNT, START_DATE),
            )
            (todo,) = cur.fetchone()
        print(
            f"[stylo-features] {todo:,} speeches need feature extraction "
            f"(workers={STYLO_WORKERS}, fetch_batch={STYLO_FETCH_BATCH}, "
            f"write_batch={STYLO_WRITE_BATCH})",
            file=sys.stderr,
        )
        if todo == 0:
            print("[stylo-features] nothing to do.", file=sys.stderr)
            return 0
        if args.limit:
            print(
                f"[stylo-features] --limit {args.limit} — capping this run.",
                file=sys.stderr,
            )

        # Server-side cursor on the autocommit read connection. Survives all
        # writes because the read connection is never used for INSERTs and
        # has no explicit transaction boundary that would close it.
        fetch_cur = read_conn.cursor(name="stylo_features_fetch")
        fetch_cur.itersize = STYLO_FETCH_BATCH
        fetch_cur.execute(
            """
            SELECT s.id, s.speech_text
            FROM public.speeches s
            LEFT JOIN public.speech_ai_scores a ON a.speech_id = s.id
            WHERE s.jurisdiction = %s
              AND s.word_count > %s
              AND s.date >= %s
              AND a.stylo_features IS NULL
            ORDER BY s.id
            """ + (f" LIMIT {int(args.limit)}" if args.limit else ""),
            (JURISDICTION, MIN_WORD_COUNT, START_DATE),
        )

        write_cur = write_conn.cursor()
        pool = mp.Pool(STYLO_WORKERS) if STYLO_WORKERS > 1 else None
        processed = 0
        t0 = time.time()
        buf: list[tuple] = []

        def _flush():
            nonlocal buf
            if not buf:
                return
            psycopg2.extras.execute_values(
                write_cur,
                """
                INSERT INTO public.speech_ai_scores
                    (speech_id, stylo_features, ai_phrase_count, model_version, run_at)
                VALUES %s
                ON CONFLICT (speech_id) DO UPDATE SET
                    stylo_features  = EXCLUDED.stylo_features,
                    ai_phrase_count = EXCLUDED.ai_phrase_count,
                    model_version   = CASE
                        WHEN public.speech_ai_scores.model_version IS NULL THEN EXCLUDED.model_version
                        WHEN public.speech_ai_scores.model_version LIKE '%%' || EXCLUDED.model_version || '%%' THEN public.speech_ai_scores.model_version
                        ELSE public.speech_ai_scores.model_version || '+' || EXCLUDED.model_version
                    END,
                    run_at = EXCLUDED.run_at
                """,
                buf,
                template="(%s, %s, %s, %s, now())",
                page_size=STYLO_WRITE_BATCH,
            )
            write_conn.commit()
            buf = []

        try:
            batch_in: list[tuple[int, str]] = []
            for row in fetch_cur:
                batch_in.append((row[0], row[1]))
                if len(batch_in) >= STYLO_WRITE_BATCH:
                    if pool is not None:
                        results = pool.map(_worker_compute, batch_in)
                    else:
                        results = [_worker_compute(x) for x in batch_in]
                    for sid, feats, tells in results:
                        buf.append((
                            sid,
                            psycopg2.extras.Json(feats),
                            tells,
                            STYLO_VERSION,
                        ))
                    _flush()
                    processed += len(batch_in)
                    batch_in = []
                    if processed % STYLO_LOG_INTERVAL < STYLO_WRITE_BATCH:
                        elapsed = time.time() - t0
                        rate = processed / elapsed if elapsed else 0
                        eta = (todo - processed) / rate if rate else 0
                        print(
                            f"[stylo-features] {processed:,}/{todo:,} "
                            f"({rate:.0f} sp/s, eta {eta/60:.0f} min)",
                            file=sys.stderr,
                        )
            # Drain.
            if batch_in:
                if pool is not None:
                    results = pool.map(_worker_compute, batch_in)
                else:
                    results = [_worker_compute(x) for x in batch_in]
                for sid, feats, tells in results:
                    buf.append((
                        sid,
                        psycopg2.extras.Json(feats),
                        tells,
                        STYLO_VERSION,
                    ))
                _flush()
                processed += len(batch_in)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            fetch_cur.close()
            write_cur.close()

        elapsed = time.time() - t0
        print(
            f"[stylo-features] done: {processed:,} speeches in "
            f"{elapsed/60:.1f} min ({processed/max(1,elapsed):.0f}/s).",
            file=sys.stderr,
        )
        return 0
    finally:
        read_conn.close()
        write_conn.close()


# ---------------------------------------------------------------------------
# Phase B step 2 — per-MP change-point detection
# ---------------------------------------------------------------------------
#
# Aggregates per-speech features into per-(MP, quarter) means; for each MP
# with sufficient pre-Nov-2022 baseline computes the Mahalanobis distance of
# each post-ChatGPT quarter from the baseline; flags MPs whose post-2022
# quarters are sustained outliers; records change-point dates via ruptures.
#
# Mahalanobis on the full ~135-dim feature vector would over-fit at small N.
# We use a curated low-dim CORE feature set that covers the main stylometric
# axes and the AI-tells. The full vector is still in the DB for ad-hoc
# exploration in Phase C.

CORE_FEATURES = (
    # Lexical diversity.
    "ttr",
    "mtld",
    "yules_k",
    # Word/sentence shape.
    "mean_word_length",
    "mean_sentence_length",
    # NOTE: sentence_length_variance was previously included but removed
    # in the v1.1 tightening — it has a small baseline variance (Hansard
    # editing flattens within-speech sentence lengths) and is right-skewed,
    # so quarter-mean Mahalanobis distance blew up to +59σ / +175σ on real
    # data, dominating every flag and producing 174/257 (68%) flag rate.
    # Syntax / punctuation.
    "subord_conj_frac",
    "em_dash_per_1k",
    "semicolon_per_1k",
    "colon_per_1k",
    "comma_per_sentence",
    # AI-tell denormalised total (per 1k tokens — derived in aggregator).
    "tell_per_1k",
    # Most-stable function-word frequencies (these shift detectably with
    # LLM-drafting in published stylometric studies).
    "fw_the",
    "fw_of",
    "fw_and",
)

# Aggregation/flagging thresholds (tightened in v1.1 after first run on real
# data flagged 68% of MPs — too noisy to be useful):
MIN_BASELINE_QUARTERS = 4
# Originally the rule was "max post quarter > χ²₉₉ AND last-3 mean >
# baseline_p95". With ~21 quarters/MP that's a 1% per-quarter false-positive
# floor, so over a multi-year post window we expect many MPs to trip it
# spuriously. Tightened to require:
#   1. ≥ MIN_POST_QUARTERS_OVER_THRESHOLD post-2022 quarters above χ²₉₉
#      (rejects single-quarter outliers from portfolio change / cabinet
#      reshuffle, etc.)
#   2. last-3-quarter mean distance > baseline_p99 (was p95)
MIN_POST_QUARTERS_OVER_THRESHOLD = 2
BASELINE_PERCENTILE = 99  # was 95
# ruptures.Pelt RBF penalty for change-point on the Mahalanobis-distance
# series. Tuned via synthetic test across shift magnitudes:
#   pen=3 detects shifts at any magnitude; stable signals stay clean;
#         gradual drift produces one (spurious but interpretable) breakpoint.
#   pen=5 saturates — RBF kernel under-detects high signal-to-noise shifts.
MAHAL_PEN = 3.0


def cmd_change_point(args) -> int:
    """Phase B step 2: per-MP Mahalanobis change-point flagging.

    Output: reports/ai_detection_per_mp_<date>.csv plus a summary printed to
    stdout. Reads from public.speech_ai_scores (joined to public.speeches +
    public.members); writes nothing to the DB.
    """
    # Lazy imports — these are research-only deps from requirements-research.txt.
    try:
        import numpy as np
        from scipy.stats import chi2
        import ruptures as rpt
    except ImportError as exc:
        print(
            f"ERROR: missing research dependency ({exc.name}). Install with:\n"
            "  pip install -r requirements-research.txt",
            file=sys.stderr,
        )
        return 2

    import csv

    if os.environ.get("GR_DB_STATEMENT_TIMEOUT_MS", "30000") != "0":
        print(
            "WARNING: GR_DB_STATEMENT_TIMEOUT_MS is not 0. The big SELECT "
            "below scans all in-scope feature rows. Re-run with "
            "`GR_DB_STATEMENT_TIMEOUT_MS=0` if the query times out.",
            file=sys.stderr,
        )

    conn = psycopg2.connect(**DB_CONFIG)
    # Server-side cursor below requires an open transaction; do not
    # autocommit. Read-only protects against accidental writes.
    conn.set_session(readonly=True, autocommit=False)
    try:
        # Pull every (name_id, quarter, features) row in scope. With ~450k
        # speeches and ~135 features each, the JSONB payload is big but
        # tractable (~1-2 GB raw); a server-side cursor avoids buffering.
        fetch_cur = conn.cursor(name="changepoint_fetch")
        fetch_cur.itersize = 5000
        fetch_cur.execute(
            """
            SELECT s.name_id,
                   date_trunc('quarter', s.date)::date AS quarter,
                   a.stylo_features
            FROM public.speeches s
            JOIN public.speech_ai_scores a ON a.speech_id = s.id
            WHERE s.jurisdiction = %s
              AND s.word_count > %s
              AND s.date >= %s
              AND s.name_id IS NOT NULL
              AND a.stylo_features IS NOT NULL
            """,
            (JURISDICTION, MIN_WORD_COUNT, START_DATE),
        )

        # nested dict: name_id -> quarter -> list[features_dict]
        per_mp: dict[str, dict] = {}
        n_rows = 0
        for name_id, quarter, feats in fetch_cur:
            n_rows += 1
            per_mp.setdefault(name_id, {}).setdefault(quarter, []).append(feats)
        fetch_cur.close()
        print(
            f"[change-point] loaded {n_rows:,} feature rows for "
            f"{len(per_mp):,} MPs.",
            file=sys.stderr,
        )

        # Pull party / display_name for join into output.
        members_cur = conn.cursor()
        members_cur.execute(
            """
            SELECT name_id, display_name, party
            FROM public.members
            WHERE jurisdiction = %s
            """,
            (JURISDICTION,),
        )
        members = {r[0]: (r[1], r[2]) for r in members_cur}
        members_cur.close()
    finally:
        conn.close()

    # ---- Per-MP analysis -------------------------------------------------
    chatgpt_q = _quarter_floor(CHATGPT_LAUNCH)

    rows_out = []
    n_flagged = 0
    n_with_baseline = 0
    n_skipped_small_baseline = 0

    for name_id, quarters in per_mp.items():
        # Aggregate each quarter's per-speech features into a single mean
        # vector. Drop quarters with < MIN_SPEECHES_PER_QUARTER speeches.
        quarter_means: dict = {}
        quarter_counts: dict = {}
        for q, speeches in quarters.items():
            if len(speeches) < MIN_SPEECHES_PER_QUARTER:
                continue
            vec = _aggregate_quarter(speeches)
            quarter_means[q] = vec
            quarter_counts[q] = len(speeches)

        if not quarter_means:
            continue

        sorted_qs = sorted(quarter_means.keys())
        baseline_qs = [q for q in sorted_qs if q < chatgpt_q]
        post_qs = [q for q in sorted_qs if q >= chatgpt_q]

        if len(baseline_qs) < MIN_BASELINE_QUARTERS:
            n_skipped_small_baseline += 1
            rows_out.append({
                "name_id": name_id,
                "display_name": (members.get(name_id) or (None, None))[0] or "",
                "party": (members.get(name_id) or (None, None))[1] or "",
                "n_baseline_quarters": len(baseline_qs),
                "n_post_quarters": len(post_qs),
                "n_post_speeches": sum(quarter_counts.get(q, 0) for q in post_qs),
                "max_distance": "",
                "last3_mean_distance": "",
                "baseline_pct_distance": "",
                "n_post_quarters_over_threshold": "",
                "flagged": False,
                "flag_reason": "insufficient_baseline",
                "first_breakpoint_q": "",
                "top_shifted_features": "",
            })
            continue

        n_with_baseline += 1

        baseline_mat = np.array([
            [quarter_means[q].get(f, 0.0) for f in CORE_FEATURES]
            for q in baseline_qs
        ], dtype=float)
        post_mat = np.array([
            [quarter_means[q].get(f, 0.0) for f in CORE_FEATURES]
            for q in post_qs
        ], dtype=float) if post_qs else np.zeros((0, len(CORE_FEATURES)))

        baseline_mean = baseline_mat.mean(axis=0)
        # Regularised covariance: shrink toward identity so we can invert
        # even when the matrix is rank-deficient at small N.
        cov = np.cov(baseline_mat, rowvar=False) if len(baseline_qs) > 1 else (
            np.eye(len(CORE_FEATURES))
        )
        cov_reg = cov + 1e-3 * np.eye(len(CORE_FEATURES))
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_reg)

        def _mahal(vec):
            d = vec - baseline_mean
            return float(np.sqrt(max(0.0, d @ cov_inv @ d)))

        baseline_distances = np.array([_mahal(v) for v in baseline_mat])
        post_distances = np.array([_mahal(v) for v in post_mat])

        baseline_pct = float(
            np.percentile(baseline_distances, BASELINE_PERCENTILE)
        ) if len(baseline_distances) else 0.0

        # Chi-square cutoff with k = len(CORE_FEATURES) degrees of freedom.
        # We compare squared Mahalanobis to chi2.ppf(0.99, k); take sqrt for
        # the linear distance threshold.
        chi2_99 = float(np.sqrt(chi2.ppf(0.99, df=len(CORE_FEATURES))))

        max_dist = float(post_distances.max()) if len(post_distances) else 0.0
        last3_mean = float(post_distances[-3:].mean()) if len(post_distances) >= 3 else (
            float(post_distances.mean()) if len(post_distances) else 0.0
        )
        n_over = int((post_distances > chi2_99).sum()) if len(post_distances) else 0

        # Flag only if (a) at least MIN_POST_QUARTERS_OVER_THRESHOLD post-
        # 2022 quarters exceed χ²₉₉, and (b) the mean of the last 3 quarters
        # exceeds the per-MP baseline at BASELINE_PERCENTILE — both
        # conditions reject one-off portfolio/staffer-change outliers.
        flagged = (
            len(post_distances) >= 3
            and n_over >= MIN_POST_QUARTERS_OVER_THRESHOLD
            and last3_mean > baseline_pct
        )
        if flagged:
            n_flagged += 1

        # Run ruptures.Pelt on the full per-MP distance time series (baseline
        # + post). Detected breakpoints are quarter indices; we report the
        # date of the first breakpoint that lands in or after Q4 2022.
        all_distances = np.concatenate([baseline_distances, post_distances])
        bkps = []
        if len(all_distances) >= 4:
            try:
                algo = rpt.Pelt(model="rbf").fit(all_distances.reshape(-1, 1))
                bkps = algo.predict(pen=MAHAL_PEN)
            except Exception:
                bkps = []
        all_qs = baseline_qs + post_qs
        first_break_q = ""
        for bp in bkps:
            # ruptures breakpoints are 1-indexed end-of-segment; convert to
            # the quarter that begins the new segment.
            if bp >= len(all_qs):
                continue
            q = all_qs[bp]
            if q >= chatgpt_q:
                first_break_q = q.isoformat() if hasattr(q, "isoformat") else str(q)
                break

        # Top 3 features that shifted most (mean post – mean baseline,
        # normalised by baseline std). Useful for caveats and for cross-
        # checking whether the shift is plausibly AI-related vs. something
        # else (e.g. portfolio change).
        top_shifted = ""
        if len(post_qs) > 0:
            post_mean = post_mat.mean(axis=0)
            baseline_std = baseline_mat.std(axis=0)
            baseline_std = np.where(baseline_std < 1e-6, 1e-6, baseline_std)
            z_shift = (post_mean - baseline_mean) / baseline_std
            top_idx = np.argsort(-np.abs(z_shift))[:3]
            top_shifted = "; ".join(
                f"{CORE_FEATURES[i]}({z_shift[i]:+.2f}σ)" for i in top_idx
            )

        rows_out.append({
            "name_id": name_id,
            "display_name": (members.get(name_id) or (None, None))[0] or "",
            "party": (members.get(name_id) or (None, None))[1] or "",
            "n_baseline_quarters": len(baseline_qs),
            "n_post_quarters": len(post_qs),
            "n_post_speeches": sum(quarter_counts.get(q, 0) for q in post_qs),
            "max_distance": f"{max_dist:.3f}",
            "last3_mean_distance": f"{last3_mean:.3f}",
            "baseline_pct_distance": f"{baseline_pct:.3f}",
            "n_post_quarters_over_threshold": n_over,
            "flagged": flagged,
            "flag_reason": (
                "sustained_shift" if flagged
                else ("transient_outlier" if max_dist > chi2_99 else "no_shift")
            ),
            "first_breakpoint_q": first_break_q,
            "top_shifted_features": top_shifted,
        })

    # ---- Output ----------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"ai_detection_per_mp_{date.today().isoformat()}.csv"
    fieldnames = [
        "name_id", "display_name", "party",
        "n_baseline_quarters", "n_post_quarters", "n_post_speeches",
        "max_distance", "last3_mean_distance", "baseline_pct_distance",
        "n_post_quarters_over_threshold",
        "flagged", "flag_reason", "first_breakpoint_q", "top_shifted_features",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # Sort: flagged first, then by max_distance desc.
        def _sort_key(r):
            try:
                d = -float(r["max_distance"]) if r["max_distance"] else 0.0
            except ValueError:
                d = 0.0
            return (not r["flagged"], d)
        for r in sorted(rows_out, key=_sort_key):
            w.writerow(r)

    print(
        f"\n[change-point] {len(per_mp)} MPs scanned, "
        f"{n_with_baseline} had a sufficient baseline, "
        f"{n_skipped_small_baseline} skipped (<{MIN_BASELINE_QUARTERS} baseline quarters), "
        f"{n_flagged} flagged for sustained post-Nov-2022 shift.\n"
        f"wrote {out_path}",
        file=sys.stderr,
    )
    return 0


def _quarter_floor(date_str: str):
    """Return the date object representing the start of the calendar quarter
    containing date_str. Postgres returns date_trunc('quarter', ...) as a
    date; we must compare against the same shape."""
    from datetime import date as _date
    y, m, d = (int(p) for p in date_str.split("-"))
    qm = ((m - 1) // 3) * 3 + 1
    return _date(y, qm, 1)


def _aggregate_quarter(speeches: list[dict]) -> dict:
    """Collapse a list of per-speech feature dicts into a quarter-mean vector.
    Computes per-1k-words `tell_per_1k` from raw `tell_total` + `n_tokens`
    so the change-point feature is comparable across quarters."""
    n = len(speeches)
    if n == 0:
        return {}
    keys = set()
    for s in speeches:
        keys.update(s.keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = [s.get(k, 0.0) or 0.0 for s in speeches]
        out[k] = sum(vals) / n
    # Derived: tell_per_1k = mean across speeches of (tells / words * 1000).
    per_speech_tpk = []
    for s in speeches:
        tt = float(s.get("tell_total", 0) or 0)
        nt = float(s.get("n_tokens", 0) or 0)
        if nt > 0:
            per_speech_tpk.append(tt / nt * 1000.0)
        else:
            per_speech_tpk.append(0.0)
    out["tell_per_1k"] = sum(per_speech_tpk) / n if per_speech_tpk else 0.0
    return out


# ---------------------------------------------------------------------------
# Phase C — cross-check + final report
# ---------------------------------------------------------------------------
#
# Pulls together Phase A (Binoculars per-speech scores) and Phase B
# (per-MP stylometric change-point) into a single markdown report under
# reports/. Runs read-only against the DB; outputs:
#
#   reports/ai_detection_<date>.md                         — main report
#   reports/ai_detection_<date>_quarterly.csv              — corpus aggregates
#   reports/ai_detection_<date>_per_party.csv              — by-party trend
#   reports/ai_detection_<date>_figs/binoculars_quarterly.png
#   reports/ai_detection_<date>_figs/binoculars_by_party.png
#   reports/ai_detection_<date>_figs/mp_heatmap.png
#
# Gracefully degrades if Phase A hasn't been run (Binoculars sections are
# replaced with "Phase A not yet run"). Same for Phase B → stylometric
# sections are skipped.

# Top-N parties to show in the per-party chart (longest tail dropped).
REPORT_TOP_PARTIES = 5
REPORT_HEATMAP_TOP_MPS = 30


def cmd_report(args) -> int:
    """Phase C: cross-check Binoculars + stylometric, render the final report."""
    try:
        import numpy as np
        from scipy.stats import chi2
        import ruptures as rpt
    except ImportError as exc:
        print(
            f"ERROR: missing research dependency ({exc.name}). Install with:\n"
            "  pip install -r requirements-research.txt",
            file=sys.stderr,
        )
        return 2

    try:
        import matplotlib
        matplotlib.use("Agg")  # Headless backend; no display required.
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(
            f"ERROR: matplotlib not installed ({exc.name}). Install with:\n"
            "  pip install -r requirements-research.txt",
            file=sys.stderr,
        )
        return 2

    import csv

    today = date.today().isoformat()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figs_dir = REPORTS_DIR / f"ai_detection_{today}_figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("GR_DB_STATEMENT_TIMEOUT_MS", "30000") != "0":
        print(
            "WARNING: GR_DB_STATEMENT_TIMEOUT_MS is not 0. The corpus-wide "
            "JOIN below scans every in-scope speech. Re-run with "
            "`GR_DB_STATEMENT_TIMEOUT_MS=0` if it times out.",
            file=sys.stderr,
        )

    conn = psycopg2.connect(**DB_CONFIG)
    # Server-side cursor (report_stylo_fetch) below needs a transaction.
    conn.set_session(readonly=True, autocommit=False)
    try:
        # Counts gate which sections we render.
        with conn.cursor() as c:
            c.execute(
                """
                SELECT
                  COUNT(*) FILTER (WHERE a.binoculars_score IS NOT NULL),
                  COUNT(*) FILTER (WHERE a.stylo_features IS NOT NULL),
                  COUNT(*) FILTER (WHERE a.fdgpt_score IS NOT NULL),
                  COUNT(*) FILTER (WHERE a.llm_judge_score IS NOT NULL),
                  COUNT(*)
                FROM public.speeches s
                JOIN public.speech_ai_scores a ON a.speech_id = s.id
                WHERE s.jurisdiction = %s
                  AND s.word_count > %s
                  AND s.date >= %s
                """,
                (JURISDICTION, MIN_WORD_COUNT, START_DATE),
            )
            (n_with_binoc, n_with_stylo, n_with_fdgpt,
             n_with_judge, n_total_scored) = c.fetchone()
        have_binoc = (n_with_binoc or 0) > 0
        have_fdgpt = (n_with_fdgpt or 0) > 0
        have_judge = (n_with_judge or 0) > 0
        have_stylo = (n_with_stylo or 0) > 0
        print(
            f"[report] {n_total_scored:,} scored: "
            f"binoc={n_with_binoc:,}, stylo={n_with_stylo:,}, "
            f"fdgpt={n_with_fdgpt:,}, judge={n_with_judge:,}.",
            file=sys.stderr,
        )

        # ---- Detector pre/post-Nov-2022 aggregates ---------------------
        # For each detector, return (pre_total, pre_flagged, post_total,
        # post_flagged, mean_score_pre, mean_score_post). For FDGPT we
        # also calibrate a threshold by taking the pre-Nov-2022 95th
        # percentile of fdgpt_score — that's a corpus-specific FPR floor
        # that tells us "how many post-2022 speeches exceed the pre-2022
        # noise level".
        detector_compare = {}
        if have_binoc:
            with conn.cursor() as c:
                c.execute(
                    """
                    SELECT
                      COUNT(*) FILTER (WHERE s.date <  %s) AS pre_total,
                      COUNT(*) FILTER (WHERE s.date <  %s AND a.is_ai_at_low_fpr) AS pre_flagged,
                      COUNT(*) FILTER (WHERE s.date >= %s) AS post_total,
                      COUNT(*) FILTER (WHERE s.date >= %s AND a.is_ai_at_low_fpr) AS post_flagged
                    FROM public.speeches s
                    JOIN public.speech_ai_scores a ON a.speech_id = s.id
                    WHERE s.jurisdiction = %s
                      AND s.word_count > %s
                      AND s.date >= %s
                      AND a.binoculars_score IS NOT NULL
                    """,
                    (CHATGPT_LAUNCH, CHATGPT_LAUNCH, CHATGPT_LAUNCH, CHATGPT_LAUNCH,
                     JURISDICTION, MIN_WORD_COUNT, START_DATE),
                )
                detector_compare["binoculars"] = c.fetchone()
        if have_judge:
            with conn.cursor() as c:
                # Two thresholds for the judge: ≥6 (possibly AI-assisted)
                # and ≥8 (likely AI-drafted).
                c.execute(
                    """
                    SELECT
                      COUNT(*) FILTER (WHERE s.date <  %s) AS pre_total,
                      COUNT(*) FILTER (WHERE s.date <  %s AND a.llm_judge_score >= 6) AS pre_6,
                      COUNT(*) FILTER (WHERE s.date <  %s AND a.llm_judge_score >= 8) AS pre_8,
                      COUNT(*) FILTER (WHERE s.date >= %s) AS post_total,
                      COUNT(*) FILTER (WHERE s.date >= %s AND a.llm_judge_score >= 6) AS post_6,
                      COUNT(*) FILTER (WHERE s.date >= %s AND a.llm_judge_score >= 8) AS post_8
                    FROM public.speeches s
                    JOIN public.speech_ai_scores a ON a.speech_id = s.id
                    WHERE s.jurisdiction = %s
                      AND s.word_count > %s
                      AND s.date >= %s
                      AND a.llm_judge_score IS NOT NULL
                    """,
                    (CHATGPT_LAUNCH, CHATGPT_LAUNCH, CHATGPT_LAUNCH,
                     CHATGPT_LAUNCH, CHATGPT_LAUNCH, CHATGPT_LAUNCH,
                     JURISDICTION, MIN_WORD_COUNT, START_DATE),
                )
                detector_compare["judge"] = c.fetchone()

        # FDGPT calibrated threshold: pre-Nov-2022 95th percentile.
        fdgpt_calibrated_threshold = None
        if have_fdgpt:
            with conn.cursor() as c:
                c.execute(
                    """
                    SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY a.fdgpt_score)
                    FROM public.speeches s
                    JOIN public.speech_ai_scores a ON a.speech_id = s.id
                    WHERE s.jurisdiction = %s
                      AND s.word_count > %s
                      AND s.date >= %s
                      AND s.date < %s
                      AND a.fdgpt_score IS NOT NULL
                    """,
                    (JURISDICTION, MIN_WORD_COUNT, START_DATE, CHATGPT_LAUNCH),
                )
                (fdgpt_calibrated_threshold,) = c.fetchone()
                fdgpt_calibrated_threshold = (
                    float(fdgpt_calibrated_threshold)
                    if fdgpt_calibrated_threshold is not None else 0.0
                )
            with conn.cursor() as c:
                c.execute(
                    """
                    SELECT
                      COUNT(*) FILTER (WHERE s.date <  %s) AS pre_total,
                      COUNT(*) FILTER (WHERE s.date <  %s AND a.fdgpt_score > %s) AS pre_flagged_calib,
                      COUNT(*) FILTER (WHERE s.date >= %s) AS post_total,
                      COUNT(*) FILTER (WHERE s.date >= %s AND a.fdgpt_score > %s) AS post_flagged_calib,
                      AVG(a.fdgpt_score) FILTER (WHERE s.date <  %s) AS mean_pre,
                      AVG(a.fdgpt_score) FILTER (WHERE s.date >= %s) AS mean_post
                    FROM public.speeches s
                    JOIN public.speech_ai_scores a ON a.speech_id = s.id
                    WHERE s.jurisdiction = %s
                      AND s.word_count > %s
                      AND s.date >= %s
                      AND a.fdgpt_score IS NOT NULL
                    """,
                    (CHATGPT_LAUNCH,
                     CHATGPT_LAUNCH, fdgpt_calibrated_threshold,
                     CHATGPT_LAUNCH,
                     CHATGPT_LAUNCH, fdgpt_calibrated_threshold,
                     CHATGPT_LAUNCH, CHATGPT_LAUNCH,
                     JURISDICTION, MIN_WORD_COUNT, START_DATE),
                )
                detector_compare["fdgpt"] = c.fetchone()

        # Cross-detector agreement: how many speeches with all 3 per-speech
        # signals (binoc/fdgpt/judge — stylo is per-MP, not per-speech)
        # are flagged by N detectors out of 3, post-Nov-2022.
        cross_agreement = None
        if have_binoc and have_fdgpt and have_judge:
            with conn.cursor() as c:
                c.execute(
                    """
                    WITH labelled AS (
                      SELECT
                        a.speech_id,
                        (a.is_ai_at_low_fpr)::int AS f_binoc,
                        (a.fdgpt_score > %s)::int AS f_fdgpt,
                        (a.llm_judge_score >= 6)::int AS f_judge6,
                        (a.llm_judge_score >= 8)::int AS f_judge8
                      FROM public.speeches s
                      JOIN public.speech_ai_scores a ON a.speech_id = s.id
                      WHERE s.jurisdiction = %s
                        AND s.word_count > %s
                        AND s.date >= %s
                        AND a.binoculars_score IS NOT NULL
                        AND a.fdgpt_score IS NOT NULL
                        AND a.llm_judge_score IS NOT NULL
                    )
                    SELECT
                      COUNT(*) AS n_speeches,
                      SUM(f_binoc) AS n_binoc,
                      SUM(f_fdgpt) AS n_fdgpt,
                      SUM(f_judge6) AS n_judge6,
                      SUM(f_judge8) AS n_judge8,
                      COUNT(*) FILTER (
                        WHERE f_binoc + f_fdgpt + f_judge8 = 3
                      ) AS all_three_strict,
                      COUNT(*) FILTER (
                        WHERE f_binoc + f_fdgpt + f_judge6 >= 2
                      ) AS two_of_three_loose,
                      COUNT(*) FILTER (
                        WHERE f_binoc + f_fdgpt + f_judge8 >= 2
                      ) AS two_of_three_strict
                    FROM labelled
                    """,
                    (fdgpt_calibrated_threshold, JURISDICTION, MIN_WORD_COUNT, START_DATE),
                )
                cross_agreement = c.fetchone()

        # ---- Quarterly aggregates (Binoculars) --------------------------
        quarterly_rows = []
        if have_binoc:
            with conn.cursor() as c:
                c.execute(
                    """
                    SELECT date_trunc('quarter', s.date)::date AS q,
                           COUNT(*)::bigint AS n_total,
                           COUNT(a.binoculars_score)::bigint AS n_scored,
                           COUNT(*) FILTER (WHERE a.is_ai_at_low_fpr)::bigint AS n_flagged,
                           AVG(a.binoculars_score)::float AS mean_score
                    FROM public.speeches s
                    LEFT JOIN public.speech_ai_scores a ON a.speech_id = s.id
                    WHERE s.jurisdiction = %s
                      AND s.word_count > %s
                      AND s.date >= %s
                    GROUP BY 1
                    ORDER BY 1
                    """,
                    (JURISDICTION, MIN_WORD_COUNT, START_DATE),
                )
                quarterly_rows = c.fetchall()

        # ---- By-party quarterly aggregates ------------------------------
        party_rows = []
        top_parties: list[str] = []
        if have_binoc:
            with conn.cursor() as c:
                c.execute(
                    """
                    SELECT COALESCE(m.party, '(unknown)') AS party,
                           COUNT(*)::bigint AS n
                    FROM public.speeches s
                    LEFT JOIN public.members m ON m.name_id = s.name_id
                    WHERE s.jurisdiction = %s
                      AND s.word_count > %s
                      AND s.date >= %s
                    GROUP BY 1
                    ORDER BY n DESC
                    LIMIT %s
                    """,
                    (JURISDICTION, MIN_WORD_COUNT, START_DATE, REPORT_TOP_PARTIES),
                )
                top_parties = [r[0] for r in c.fetchall()]
            with conn.cursor() as c:
                c.execute(
                    """
                    SELECT date_trunc('quarter', s.date)::date AS q,
                           COALESCE(m.party, '(unknown)') AS party,
                           COUNT(*)::bigint AS n_total,
                           COUNT(*) FILTER (WHERE a.is_ai_at_low_fpr)::bigint AS n_flagged
                    FROM public.speeches s
                    LEFT JOIN public.members m ON m.name_id = s.name_id
                    LEFT JOIN public.speech_ai_scores a ON a.speech_id = s.id
                    WHERE s.jurisdiction = %s
                      AND s.word_count > %s
                      AND s.date >= %s
                      AND COALESCE(m.party, '(unknown)') = ANY(%s)
                    GROUP BY 1, 2
                    ORDER BY 1, 2
                    """,
                    (JURISDICTION, MIN_WORD_COUNT, START_DATE, top_parties),
                )
                party_rows = c.fetchall()

        # ---- Stylometric per-MP analysis --------------------------------
        per_mp_results: dict = {}
        members: dict = {}
        if have_stylo:
            members_cur = conn.cursor()
            members_cur.execute(
                """
                SELECT name_id, display_name, party
                FROM public.members
                WHERE jurisdiction = %s
                """,
                (JURISDICTION,),
            )
            members = {r[0]: (r[1], r[2]) for r in members_cur}
            members_cur.close()

            fetch_cur = conn.cursor(name="report_stylo_fetch")
            fetch_cur.itersize = 5000
            fetch_cur.execute(
                """
                SELECT s.name_id,
                       date_trunc('quarter', s.date)::date AS quarter,
                       a.stylo_features
                FROM public.speeches s
                JOIN public.speech_ai_scores a ON a.speech_id = s.id
                WHERE s.jurisdiction = %s
                  AND s.word_count > %s
                  AND s.date >= %s
                  AND s.name_id IS NOT NULL
                  AND a.stylo_features IS NOT NULL
                """,
                (JURISDICTION, MIN_WORD_COUNT, START_DATE),
            )
            per_mp: dict[str, dict] = {}
            for name_id, quarter, feats in fetch_cur:
                per_mp.setdefault(name_id, {}).setdefault(quarter, []).append(feats)
            fetch_cur.close()
            per_mp_results = _analyze_all_mps(per_mp, np, chi2, rpt)

        # ---- Cross-check: Binoculars stats for flagged MPs --------------
        # For each MP flagged by stylometric change-point, compute fraction
        # of their post-2022 speeches that Binoculars also flagged.
        cross_check: dict = {}
        if have_binoc and per_mp_results:
            flagged_ids = [
                nid for nid, r in per_mp_results.items() if r.get("flagged")
            ]
            if flagged_ids:
                with conn.cursor() as c:
                    c.execute(
                        """
                        SELECT s.name_id,
                               COUNT(*) FILTER (WHERE a.binoculars_score IS NOT NULL)::bigint AS n_scored,
                               COUNT(*) FILTER (WHERE a.is_ai_at_low_fpr)::bigint AS n_flagged,
                               AVG(a.binoculars_score)::float AS mean_score
                        FROM public.speeches s
                        LEFT JOIN public.speech_ai_scores a ON a.speech_id = s.id
                        WHERE s.jurisdiction = %s
                          AND s.word_count > %s
                          AND s.date >= %s
                          AND s.name_id = ANY(%s)
                        GROUP BY 1
                        """,
                        (JURISDICTION, MIN_WORD_COUNT, CHATGPT_LAUNCH, flagged_ids),
                    )
                    for nid, n_scored, n_flag, mean_score in c.fetchall():
                        cross_check[nid] = {
                            "n_scored": n_scored or 0,
                            "n_flagged": n_flag or 0,
                            "mean_score": mean_score,
                        }
    finally:
        conn.close()

    # =====================================================================
    # Render charts
    # =====================================================================
    chart_paths: dict[str, str] = {}

    if quarterly_rows:
        qs = [r[0] for r in quarterly_rows]
        pcts = [
            (r[3] / r[2] * 100.0) if r[2] else 0.0
            for r in quarterly_rows
        ]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(qs, pcts, marker="o", linewidth=1.5, color="#c0392b")
        ax.axvline(_quarter_floor(CHATGPT_LAUNCH), color="grey", linestyle="--",
                   linewidth=1, label="ChatGPT launch")
        # Pre-ChatGPT baseline band.
        pre = [p for q, p in zip(qs, pcts) if q < _quarter_floor(CHATGPT_LAUNCH)]
        if pre:
            import statistics as _stat
            mean = _stat.mean(pre)
            std = _stat.pstdev(pre) if len(pre) > 1 else 0.0
            ax.axhspan(max(0, mean - std), mean + std, alpha=0.12, color="grey",
                       label=f"pre-ChatGPT baseline ±1σ ({mean:.1f}%)")
        ax.set_ylabel("% of speeches flagged by Binoculars (low-FPR)")
        ax.set_xlabel("Quarter")
        ax.set_title("Binoculars-flagged % over time, federal commonwealth speeches")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = figs_dir / "binoculars_quarterly.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        chart_paths["binoculars_quarterly"] = path.name

    if party_rows:
        # party -> {q: pct}
        by_party: dict[str, dict] = {}
        for q, party, n_total, n_flag in party_rows:
            by_party.setdefault(party, {})[q] = (n_flag / n_total * 100.0) if n_total else 0.0
        fig, ax = plt.subplots(figsize=(9, 4))
        for party in top_parties:
            series = sorted(by_party.get(party, {}).items())
            if not series:
                continue
            ax.plot(
                [s[0] for s in series], [s[1] for s in series],
                marker=".", label=party, linewidth=1.4,
            )
        ax.axvline(_quarter_floor(CHATGPT_LAUNCH), color="grey", linestyle="--",
                   linewidth=1)
        ax.set_ylabel("% Binoculars-flagged")
        ax.set_xlabel("Quarter")
        ax.set_title("Binoculars-flagged % by party, federal commonwealth speeches")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = figs_dir / "binoculars_by_party.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        chart_paths["binoculars_by_party"] = path.name

    if per_mp_results:
        # Top-N MPs by post-Nov-2022 speech count, with sufficient baseline.
        prolific = sorted(
            (
                (nid, r) for nid, r in per_mp_results.items()
                if r.get("n_post_speeches") and r.get("baseline_qs")
            ),
            key=lambda x: -x[1]["n_post_speeches"],
        )[:REPORT_HEATMAP_TOP_MPS]
        if prolific:
            all_qs_set = set()
            for _, r in prolific:
                all_qs_set.update(r["per_quarter_distance"].keys())
            all_qs_sorted = sorted(all_qs_set)
            mat = np.zeros((len(prolific), len(all_qs_sorted)))
            mat[:] = np.nan
            for i, (_, r) in enumerate(prolific):
                for j, q in enumerate(all_qs_sorted):
                    if q in r["per_quarter_distance"]:
                        mat[i, j] = r["per_quarter_distance"][q]
            labels = [
                (members.get(nid) or (None, None))[0] or nid
                for nid, _ in prolific
            ]
            fig, ax = plt.subplots(figsize=(min(14, 1.2 + 0.45 * len(all_qs_sorted)),
                                           min(12, 0.4 * len(prolific) + 1.5)))
            im = ax.imshow(mat, aspect="auto", cmap="OrRd", interpolation="nearest")
            ax.set_yticks(range(len(prolific)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xticks(range(len(all_qs_sorted)))
            ax.set_xticklabels([q.isoformat() for q in all_qs_sorted],
                              rotation=70, fontsize=7)
            chatgpt_q = _quarter_floor(CHATGPT_LAUNCH)
            for j, q in enumerate(all_qs_sorted):
                if q == chatgpt_q:
                    ax.axvline(j - 0.5, color="black", linestyle="--", linewidth=1)
                    break
            ax.set_title(
                f"Per-MP Mahalanobis distance from pre-Nov-2022 baseline "
                f"(top {len(prolific)} most-prolific post-2022)",
                fontsize=10,
            )
            cb = fig.colorbar(im, ax=ax, fraction=0.025)
            cb.set_label("Mahalanobis distance (post-baseline)", fontsize=8)
            fig.tight_layout()
            path = figs_dir / "mp_heatmap.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            chart_paths["mp_heatmap"] = path.name

    # =====================================================================
    # Write CSVs
    # =====================================================================
    if quarterly_rows:
        path = REPORTS_DIR / f"ai_detection_{today}_quarterly.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["quarter", "n_total", "n_scored", "n_flagged",
                        "pct_flagged", "mean_score"])
            for q, n_total, n_scored, n_flag, mean_score in quarterly_rows:
                pct = (n_flag / n_total * 100.0) if n_total else 0.0
                w.writerow([q.isoformat(), n_total, n_scored, n_flag,
                            f"{pct:.2f}", f"{mean_score:.4f}" if mean_score else ""])

    if party_rows:
        path = REPORTS_DIR / f"ai_detection_{today}_per_party.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["quarter", "party", "n_total", "n_flagged", "pct_flagged"])
            for q, party, n_total, n_flag in party_rows:
                pct = (n_flag / n_total * 100.0) if n_total else 0.0
                w.writerow([q.isoformat(), party, n_total, n_flag, f"{pct:.2f}"])

    # =====================================================================
    # Write the markdown report
    # =====================================================================
    md = _render_report_markdown(
        today=today,
        n_with_binoc=n_with_binoc or 0,
        n_with_stylo=n_with_stylo or 0,
        n_with_fdgpt=n_with_fdgpt or 0,
        n_with_judge=n_with_judge or 0,
        n_total_scored=n_total_scored or 0,
        quarterly_rows=quarterly_rows,
        per_mp_results=per_mp_results,
        cross_check=cross_check,
        members=members,
        chart_paths=chart_paths,
        detector_compare=detector_compare,
        fdgpt_calibrated_threshold=fdgpt_calibrated_threshold,
        cross_agreement=cross_agreement,
    )
    out_path = REPORTS_DIR / f"ai_detection_{today}.md"
    out_path.write_text(md)
    print(f"\n[report] wrote {out_path}", file=sys.stderr)
    if chart_paths:
        for k, v in chart_paths.items():
            print(f"  fig: {figs_dir / v}", file=sys.stderr)
    return 0


def _analyze_all_mps(per_mp, np, chi2, rpt) -> dict:
    """Run the per-MP Mahalanobis + change-point analysis used by both
    cmd_change_point and cmd_report. Returns a dict
    name_id -> result dict including per-quarter distances (the heatmap
    uses these; cmd_change_point ignores them)."""
    chatgpt_q = _quarter_floor(CHATGPT_LAUNCH)
    chi2_99 = float(np.sqrt(chi2.ppf(0.99, df=len(CORE_FEATURES))))
    out: dict = {}

    for name_id, quarters in per_mp.items():
        quarter_means: dict = {}
        quarter_counts: dict = {}
        for q, speeches in quarters.items():
            if len(speeches) < MIN_SPEECHES_PER_QUARTER:
                continue
            quarter_means[q] = _aggregate_quarter(speeches)
            quarter_counts[q] = len(speeches)
        if not quarter_means:
            continue
        sorted_qs = sorted(quarter_means.keys())
        baseline_qs = [q for q in sorted_qs if q < chatgpt_q]
        post_qs = [q for q in sorted_qs if q >= chatgpt_q]

        result = {
            "n_baseline_quarters": len(baseline_qs),
            "n_post_quarters": len(post_qs),
            "n_post_speeches": sum(quarter_counts.get(q, 0) for q in post_qs),
            "baseline_qs": baseline_qs,
            "post_qs": post_qs,
            "per_quarter_distance": {},
            "flagged": False,
            "flag_reason": "",
            "max_distance": None,
            "last3_mean_distance": None,
            "baseline_pct_distance": None,
            "n_post_quarters_over_threshold": 0,
            "first_breakpoint_q": "",
            "top_shifted_features": "",
        }

        if len(baseline_qs) < MIN_BASELINE_QUARTERS:
            result["flag_reason"] = "insufficient_baseline"
            out[name_id] = result
            continue

        baseline_mat = np.array(
            [[quarter_means[q].get(f, 0.0) for f in CORE_FEATURES] for q in baseline_qs],
            dtype=float,
        )
        post_mat = (
            np.array(
                [[quarter_means[q].get(f, 0.0) for f in CORE_FEATURES] for q in post_qs],
                dtype=float,
            )
            if post_qs else np.zeros((0, len(CORE_FEATURES)))
        )
        baseline_mean = baseline_mat.mean(axis=0)
        cov = (
            np.cov(baseline_mat, rowvar=False)
            if len(baseline_qs) > 1
            else np.eye(len(CORE_FEATURES))
        )
        cov_reg = cov + 1e-3 * np.eye(len(CORE_FEATURES))
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_reg)

        def _mahal(vec):
            d = vec - baseline_mean
            return float(np.sqrt(max(0.0, d @ cov_inv @ d)))

        baseline_distances = np.array([_mahal(v) for v in baseline_mat])
        post_distances = np.array([_mahal(v) for v in post_mat])

        for q, d in zip(baseline_qs, baseline_distances):
            result["per_quarter_distance"][q] = float(d)
        for q, d in zip(post_qs, post_distances):
            result["per_quarter_distance"][q] = float(d)

        baseline_pct = (
            float(np.percentile(baseline_distances, BASELINE_PERCENTILE))
            if len(baseline_distances) else 0.0
        )
        max_dist = float(post_distances.max()) if len(post_distances) else 0.0
        last3_mean = (
            float(post_distances[-3:].mean())
            if len(post_distances) >= 3
            else (float(post_distances.mean()) if len(post_distances) else 0.0)
        )
        n_over = int((post_distances > chi2_99).sum()) if len(post_distances) else 0
        flagged = (
            len(post_distances) >= 3
            and n_over >= MIN_POST_QUARTERS_OVER_THRESHOLD
            and last3_mean > baseline_pct
        )
        all_distances = np.concatenate([baseline_distances, post_distances])
        bkps = []
        if len(all_distances) >= 4:
            try:
                bkps = rpt.Pelt(model="rbf").fit(
                    all_distances.reshape(-1, 1)
                ).predict(pen=MAHAL_PEN)
            except Exception:
                bkps = []
        all_qs = baseline_qs + post_qs
        first_break_q = ""
        for bp in bkps:
            if bp >= len(all_qs):
                continue
            q = all_qs[bp]
            if q >= chatgpt_q:
                first_break_q = q.isoformat() if hasattr(q, "isoformat") else str(q)
                break

        top_shifted = ""
        if len(post_qs) > 0:
            post_mean = post_mat.mean(axis=0)
            baseline_std = baseline_mat.std(axis=0)
            baseline_std = np.where(baseline_std < 1e-6, 1e-6, baseline_std)
            z_shift = (post_mean - baseline_mean) / baseline_std
            top_idx = np.argsort(-np.abs(z_shift))[:3]
            top_shifted = "; ".join(
                f"{CORE_FEATURES[i]}({z_shift[i]:+.2f}σ)" for i in top_idx
            )

        result.update({
            "max_distance": max_dist,
            "last3_mean_distance": last3_mean,
            "baseline_pct_distance": baseline_pct,
            "n_post_quarters_over_threshold": n_over,
            "flagged": flagged,
            "flag_reason": (
                "sustained_shift" if flagged
                else ("transient_outlier" if max_dist > chi2_99 else "no_shift")
            ),
            "first_breakpoint_q": first_break_q,
            "top_shifted_features": top_shifted,
        })
        out[name_id] = result
    return out


def _render_report_markdown(
    today,
    n_with_binoc,
    n_with_stylo,
    n_total_scored,
    quarterly_rows,
    per_mp_results,
    cross_check,
    members,
    chart_paths,
    n_with_fdgpt=0,
    n_with_judge=0,
    detector_compare=None,
    fdgpt_calibrated_threshold=None,
    cross_agreement=None,
) -> str:
    have_binoc = n_with_binoc > 0
    have_stylo = n_with_stylo > 0
    have_fdgpt = n_with_fdgpt > 0
    have_judge = n_with_judge > 0
    detector_compare = detector_compare or {}

    lines: list[str] = []
    lines.append(f"# AI-drafted speech detection — {today}")
    lines.append("")
    lines.append(
        "_One-off research analysis estimating what fraction of federal "
        "Australian Hansard speeches between Nov 2022 (ChatGPT launch) and "
        "today were drafted with AI assistance, broken down by MP, party "
        "and time. NOT a productionised feature; per-speech scores are not "
        "exposed via any public route._"
    )
    lines.append("")

    # ---- Caveats up front (per plan) ------------------------------------
    lines.append("## ⚠ Caveats (read first)")
    lines.append("")
    lines.append(
        "- **\"Drafted with AI\" ≠ \"written by AI\".** Detectors flag both. "
        "This analysis cannot tell whether an MP edited an AI draft, or "
        "fed an AI a human draft for polishing — both produce the same "
        "signal."
    )
    lines.append(
        "- **Hansard editorial smoothing.** Official Hansard is lightly "
        "grammar-edited; this is itself a stylistic shift that pre-dates "
        "GPT. Pre-Nov-2022 baseline change-points act as a control — if "
        "we see baseline shifts of similar magnitude, the post-2022 signal "
        "is not as clean as it looks."
    )
    lines.append(
        "- **Detector drift.** Binoculars was trained on Falcon-7B; speeches "
        "may have been drafted with GPT-4o / Claude / Gemini whose token "
        "distributions differ. The score is an \"AI-likeness\" proxy, not "
        "a probability."
    )
    lines.append(
        "- **Speeches are staffer-drafted by default.** Pre-Nov-2022 base"
        "line change-points within an MP's history are routinely produced by "
        "staffer turnover and portfolio changes. Mahalanobis flagging "
        "cannot distinguish a new chief-of-staff from an MP starting to "
        "use ChatGPT."
    )
    lines.append(
        "- **Block quotes are not stripped.** Hansard markup doesn't "
        "preserve quote-block indentation. Speeches that read large "
        "passages of correspondence will have anomalous stylometric "
        "features unrelated to AI use."
    )
    lines.append(
        "- **Backbencher noise.** MPs with few speeches per quarter have "
        "unstable Mahalanobis distances; flag with care. We require ≥4 "
        "pre-Nov-2022 quarters of ≥5 substantive speeches each — MPs "
        "without that baseline are excluded from change-point flagging."
    )
    lines.append(
        "- **Paraphrasing defeats both methods.** Anyone who runs the AI "
        "draft through a thesaurus or prompts the model to \"write in a "
        "natural Australian parliamentary style\" can defeat both "
        "Binoculars and stylometric detection. We measure a floor, not "
        "a ceiling."
    )
    lines.append(
        "- **Pre-registered thresholds.** Binoculars threshold = paper's "
        f"low-FPR cutoff {BINOC_LOW_FPR_THRESHOLD}; Mahalanobis cutoff = "
        "χ²₉₉(15). Both fixed before any per-MP results were inspected. "
        "Document any post-hoc changes."
    )
    lines.append("")

    # ---- Headline -------------------------------------------------------
    lines.append("## Headline")
    lines.append("")
    if not (have_binoc or have_stylo):
        lines.append(
            "**Neither phase has produced data yet.** Run "
            "`scripts/research_ai_detection.py stylo-features` for Phase B, "
            "and the Binoculars dump → GPU run → load workflow for Phase A."
        )
        lines.append("")
        return "\n".join(lines)

    if have_binoc and quarterly_rows:
        chatgpt_q = _quarter_floor(CHATGPT_LAUNCH)
        pre_total = sum(r[2] for r in quarterly_rows if r[0] < chatgpt_q)
        pre_flag = sum(r[3] for r in quarterly_rows if r[0] < chatgpt_q)
        post_total = sum(r[2] for r in quarterly_rows if r[0] >= chatgpt_q)
        post_flag = sum(r[3] for r in quarterly_rows if r[0] >= chatgpt_q)
        pre_pct = (pre_flag / pre_total * 100.0) if pre_total else 0.0
        post_pct = (post_flag / post_total * 100.0) if post_total else 0.0
        delta = post_pct - pre_pct
        lines.append(
            f"- Binoculars flagged **{post_pct:.2f}%** of post-Nov-2022 "
            f"speeches as AI-likely (paper's low-FPR threshold "
            f"<{BINOC_LOW_FPR_THRESHOLD}), vs **{pre_pct:.2f}%** in the "
            f"pre-Nov-2022 baseline (Δ = **{delta:+.2f} pts**). The "
            "baseline % is the detector's empirical false-positive rate "
            "on this corpus. Subtract it from the post-2022 number to get "
            "a rough adjusted estimate."
        )
        adjusted = max(0.0, delta)
        lines.append(
            f"- **Conservative adjusted estimate: ≈{adjusted:.2f}%** of "
            "post-Nov-2022 substantive federal speeches show AI-drafting "
            "signal beyond the detector's pre-2022 false-positive floor."
        )

    if have_stylo and per_mp_results:
        n_examined = len(per_mp_results)
        n_with_baseline = sum(
            1 for r in per_mp_results.values()
            if r.get("flag_reason") != "insufficient_baseline"
        )
        n_flagged = sum(1 for r in per_mp_results.values() if r.get("flagged"))
        lines.append(
            f"- {n_examined} MPs scanned for stylometric change-point; "
            f"{n_with_baseline} had a sufficient pre-Nov-2022 baseline "
            f"(≥{MIN_BASELINE_QUARTERS} usable quarters); "
            f"**{n_flagged} flagged** for sustained post-Nov-2022 shift."
        )
    lines.append("")

    # ---- Detector comparison -------------------------------------------
    if detector_compare:
        lines.append("## Detector comparison (pre vs post ChatGPT)")
        lines.append("")
        lines.append(
            "_Four detectors, four different answers. The disagreement IS "
            "the signal: each is sensitive to a different facet of "
            "AI-drafted text._"
        )
        lines.append("")
        lines.append("| Detector | Threshold | Pre-ChatGPT flagged | Post-ChatGPT flagged | Δ pts |")
        lines.append("| --- | --- | ---: | ---: | ---: |")

        if "binoculars" in detector_compare:
            pre_t, pre_f, post_t, post_f = detector_compare["binoculars"]
            pre_pct = (pre_f / pre_t * 100.0) if pre_t else 0.0
            post_pct = (post_f / post_t * 100.0) if post_t else 0.0
            lines.append(
                f"| Binoculars (Falcon-7B base/instruct) | "
                f"score < {BINOC_LOW_FPR_THRESHOLD} | "
                f"{pre_pct:.2f}% ({pre_f:,}/{pre_t:,}) | "
                f"{post_pct:.2f}% ({post_f:,}/{post_t:,}) | "
                f"{post_pct - pre_pct:+.2f} |"
            )

        if "fdgpt" in detector_compare and fdgpt_calibrated_threshold is not None:
            pre_t, pre_f, post_t, post_f, mean_pre, mean_post = (
                detector_compare["fdgpt"]
            )
            pre_pct = (pre_f / pre_t * 100.0) if pre_t else 0.0
            post_pct = (post_f / post_t * 100.0) if post_t else 0.0
            lines.append(
                f"| Fast-DetectGPT (gpt-neo-2.7B) | "
                f"score > {fdgpt_calibrated_threshold:.3f} (pre-2022 p95) | "
                f"{pre_pct:.2f}% ({pre_f:,}/{pre_t:,}) | "
                f"{post_pct:.2f}% ({post_f:,}/{post_t:,}) | "
                f"{post_pct - pre_pct:+.2f} |"
            )
            mean_pre_f = float(mean_pre) if mean_pre is not None else 0.0
            mean_post_f = float(mean_post) if mean_post is not None else 0.0
            lines.append(
                f"| _(FDGPT mean score)_ | _higher → more AI-like_ | "
                f"_{mean_pre_f:+.3f}_ | _{mean_post_f:+.3f}_ | "
                f"_{mean_post_f - mean_pre_f:+.3f}_ |"
            )

        if "judge" in detector_compare:
            pre_t, pre_6, pre_8, post_t, post_6, post_8 = detector_compare["judge"]
            pre_6_pct = (pre_6 / pre_t * 100.0) if pre_t else 0.0
            post_6_pct = (post_6 / post_t * 100.0) if post_t else 0.0
            pre_8_pct = (pre_8 / pre_t * 100.0) if pre_t else 0.0
            post_8_pct = (post_8 / post_t * 100.0) if post_t else 0.0
            lines.append(
                f"| Claude Haiku judge | score ≥ 6 (possibly AI-assisted) | "
                f"{pre_6_pct:.2f}% ({pre_6:,}/{pre_t:,}) | "
                f"{post_6_pct:.2f}% ({post_6:,}/{post_t:,}) | "
                f"{post_6_pct - pre_6_pct:+.2f} |"
            )
            lines.append(
                f"| Claude Haiku judge | score ≥ 8 (likely AI-drafted) | "
                f"{pre_8_pct:.2f}% ({pre_8:,}/{pre_t:,}) | "
                f"{post_8_pct:.2f}% ({post_8:,}/{post_t:,}) | "
                f"{post_8_pct - pre_8_pct:+.2f} |"
            )

        lines.append("")
        lines.append(
            "**Reading the table:** the *Δ pts* column is what to look at. "
            "Each detector has a baseline FPR on this corpus (the pre-ChatGPT "
            "rate); we'd expect the post-ChatGPT rate to be ≥ the baseline if "
            "AI use is real. A near-zero Δ for a detector is evidence that "
            "*that detector* doesn't see the kind of AI use that's happening."
        )
        lines.append("")

        # Cross-detector agreement
        if cross_agreement is not None:
            (
                n_speeches, n_binoc, n_fdgpt, n_judge6, n_judge8,
                all_three_strict, two_of_three_loose, two_of_three_strict
            ) = cross_agreement
            lines.append("### Cross-detector agreement (per-speech)")
            lines.append("")
            lines.append(
                f"_Restricted to the {n_speeches:,} speeches scored by all "
                "three per-speech detectors (Binoculars, Fast-DetectGPT, "
                "Claude Haiku judge). Stylometric is per-MP and not in this "
                "agreement set._"
            )
            lines.append("")
            lines.append("| Combination | Speeches |")
            lines.append("| --- | ---: |")
            lines.append(f"| Binoculars flagged | {n_binoc:,} |")
            lines.append(
                f"| Fast-DetectGPT flagged (pre-2022 p95 calibrated) | "
                f"{n_fdgpt:,} |"
            )
            lines.append(f"| Haiku judge ≥ 6 | {n_judge6:,} |")
            lines.append(f"| Haiku judge ≥ 8 | {n_judge8:,} |")
            lines.append(
                f"| **All three strict** (Binoculars AND FDGPT AND judge ≥ 8) | "
                f"**{all_three_strict:,}** |"
            )
            lines.append(
                f"| **2-of-3 strict** (any two of Binoculars / FDGPT / judge ≥ 8) | "
                f"**{two_of_three_strict:,}** |"
            )
            lines.append(
                f"| 2-of-3 loose (any two of Binoculars / FDGPT / judge ≥ 6) | "
                f"{two_of_three_loose:,} |"
            )
            lines.append("")
            if all_three_strict > 0:
                consensus_pct = all_three_strict / n_speeches * 100.0
                lines.append(
                    f"**Consensus AI-drafting rate (all three detectors "
                    f"agree at strict thresholds): {consensus_pct:.3f}% of "
                    f"the cross-scored set ({all_three_strict:,} speeches).** "
                    "These speeches are the most defensible candidates for "
                    "manual audit before naming any individual MP."
                )
            else:
                lines.append(
                    "**No speeches were flagged by all three detectors at "
                    "strict thresholds.** This is itself the headline: "
                    "current open-source detectors do not converge on a "
                    "consensus set of \"this is AI-drafted\" speeches in "
                    "the federal Hansard corpus."
                )
            lines.append("")

    # ---- Method ---------------------------------------------------------
    lines.append("## Method")
    lines.append("")
    lines.append(
        f"- **Scope**: `public.speeches WHERE jurisdiction='{JURISDICTION}' "
        f"AND word_count > {MIN_WORD_COUNT} AND date >= '{START_DATE}'`. "
        "Substantive speeches only (procedural one-liners excluded)."
    )
    lines.append(
        "- **Phase A — Binoculars** (Hans et al., ICML 2024): self-hosted "
        "open-source detector run on a rented A6000. Score is the ratio "
        "of perplexity under Falcon-7B-base to cross-perplexity under "
        "Falcon-7B-instruct. Lower score → more AI-like. Threshold "
        f"{BINOC_LOW_FPR_THRESHOLD} (paper's low-FPR setting)."
    )
    lines.append(
        "- **Phase B — stylometric change-point per MP**: ~135 features "
        "per speech (TTR, MTLD, Yule's K, sentence/word length, function-"
        "word vector, em-dash density, AI-tell phrase counts). Aggregate "
        "to (MP, quarter) means; require ≥5 speeches per bucket. For each "
        "MP with ≥4 baseline quarters, compute Mahalanobis distance of "
        "each post-Nov-2022 quarter from baseline; flag if max post-2022 "
        "distance > χ²₉₉(15) cutoff AND mean of last 3 quarters > "
        "baseline 95th percentile (sustained, not one-off). ruptures.Pelt "
        "(RBF kernel, pen=5) for breakpoint dating."
    )
    lines.append("")

    # ---- Corpus-wide trend ---------------------------------------------
    if have_binoc and "binoculars_quarterly" in chart_paths:
        lines.append("## Corpus-wide trend (Binoculars)")
        lines.append("")
        lines.append(
            f"![Binoculars-flagged % over time]"
            f"(ai_detection_{today}_figs/{chart_paths['binoculars_quarterly']})"
        )
        lines.append("")
        lines.append("See also: `ai_detection_{today}_quarterly.csv`.".replace(
            "{today}", today))
        lines.append("")

    # ---- By party -------------------------------------------------------
    if have_binoc and "binoculars_by_party" in chart_paths:
        lines.append("## By party")
        lines.append("")
        lines.append(
            f"![Binoculars-flagged % by party]"
            f"(ai_detection_{today}_figs/{chart_paths['binoculars_by_party']})"
        )
        lines.append("")
        lines.append(
            "Per-party series should be interpreted alongside the caveat that "
            "differences in portfolio mix and seniority across parties produce "
            "stylistic differences unrelated to AI use."
        )
        lines.append("")

    # ---- Per-MP heatmap -------------------------------------------------
    if have_stylo and "mp_heatmap" in chart_paths:
        lines.append("## Per-MP stylometric distance (top-30 most-prolific)")
        lines.append("")
        lines.append(
            f"![Per-MP Mahalanobis heatmap]"
            f"(ai_detection_{today}_figs/{chart_paths['mp_heatmap']})"
        )
        lines.append("")
        lines.append(
            "Each row is one MP; cells are Mahalanobis distance of that "
            "MP's per-quarter feature mean from their own pre-Nov-2022 "
            "baseline. The dashed vertical line is ChatGPT launch. A "
            "warm column post-launch with no warm cells pre-launch is the "
            "signature of a stylistic shift; warm columns _both_ pre- "
            "and post-launch indicate noise or staffer changes rather "
            "than AI use."
        )
        lines.append("")

    # ---- Flagged MPs ----------------------------------------------------
    if have_stylo and per_mp_results:
        lines.append("## Stylometrically-flagged MPs")
        lines.append("")
        flagged = sorted(
            (
                (nid, r) for nid, r in per_mp_results.items() if r.get("flagged")
            ),
            key=lambda x: -(x[1].get("max_distance") or 0),
        )
        if not flagged:
            lines.append(
                "_No MPs met the sustained-shift criteria. Either the corpus "
                "shows no detectable post-Nov-2022 stylistic shift at this "
                "threshold, or Phase B hasn't yet processed enough speeches._"
            )
        else:
            lines.append(
                "| MP | Party | Max dist | Last-3 mean | Baseline pct | "
                "Cross-check (% Binoc-flagged) | First breakpoint | "
                "Top shifted features |"
            )
            lines.append(
                "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"
            )
            for nid, r in flagged[:30]:
                disp = (members.get(nid) or (None, None))[0] or nid
                party = (members.get(nid) or (None, None))[1] or "—"
                cc = cross_check.get(nid, {})
                if cc.get("n_scored"):
                    cc_str = (
                        f"{cc['n_flagged']/cc['n_scored']*100:.1f}% "
                        f"({cc['n_flagged']}/{cc['n_scored']})"
                    )
                else:
                    cc_str = "—"
                lines.append(
                    f"| {disp} | {party} | {r['max_distance']:.2f} | "
                    f"{r['last3_mean_distance']:.2f} | "
                    f"{r['baseline_pct_distance']:.2f} | {cc_str} | "
                    f"{r['first_breakpoint_q'] or '—'} | "
                    f"{r['top_shifted_features']} |"
                )
        lines.append("")
        lines.append(
            f"Full per-MP CSV: `ai_detection_per_mp_{today}.csv` "
            f"(generated by `change-point` subcommand)."
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"_Plan: the methodology section in the README. "
        f"Phase A: `scripts/research_ai_binoculars.py` (GPU host). "
        f"Phase B + Phase C: `scripts/research_ai_detection.py`._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase A — Binoculars I/O bookends
# ---------------------------------------------------------------------------
#
# Phase A's heavy lifting (running Falcon-7B base + instruct) happens on a
# rented GPU host via scripts/research_ai_binoculars.py. This driver
# provides the bookends:
#
#   dump-speeches   Dump in-scope (id, text) JSONL for shipping to GPU host.
#                   Skips speeches that already have a Binoculars score.
#                   Speech text is truncated to MAX_INPUT_WORDS (kept in
#                   sync with the GPU wrapper).
#   load-binoculars Read JSONL of {id, score, is_ai} produced by the GPU
#                   wrapper, bulk-upsert into public.speech_ai_scores.

# Keep in sync with research_ai_binoculars.MAX_INPUT_WORDS. Mirroring the
# constant rather than importing avoids pulling the GPU-host script's
# imports into this file.
BINOC_MAX_INPUT_WORDS = 600
BINOC_LOW_FPR_THRESHOLD = 0.901


def cmd_dump_speeches(args) -> int:
    """Dump in-scope (id, text) pairs as JSONL for shipping to a GPU host."""
    import json as _json

    if os.environ.get("GR_DB_STATEMENT_TIMEOUT_MS", "30000") != "0":
        print(
            "WARNING: GR_DB_STATEMENT_TIMEOUT_MS is not 0. The streaming "
            "SELECT below scans the full corpus. Re-run with "
            "`GR_DB_STATEMENT_TIMEOUT_MS=0` if it times out.",
            file=sys.stderr,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(**DB_CONFIG)
    # Server-side cursor below requires a transaction; do not autocommit.
    conn.set_session(readonly=True, autocommit=False)
    n_written = 0
    n_skipped_empty = 0
    try:
        cur = conn.cursor(name="dump_speeches_fetch")
        cur.itersize = 2000
        # LEFT JOIN so we skip speeches that already have a Binoculars
        # score from a previous run (resumable).
        cur.execute(
            """
            SELECT s.id, s.speech_text
            FROM public.speeches s
            LEFT JOIN public.speech_ai_scores a ON a.speech_id = s.id
            WHERE s.jurisdiction = %s
              AND s.word_count > %s
              AND s.date >= %s
              AND a.binoculars_score IS NULL
            ORDER BY s.id
            """ + (f" LIMIT {int(args.limit)}" if args.limit else ""),
            (JURISDICTION, MIN_WORD_COUNT, START_DATE),
        )
        with output.open("w") as f:
            for sid, text in cur:
                if not text or not text.strip():
                    n_skipped_empty += 1
                    continue
                # Truncate to first BINOC_MAX_INPUT_WORDS words. Most
                # stylistic AI signal is in the introduction.
                parts = text.split()
                if len(parts) > BINOC_MAX_INPUT_WORDS:
                    text = " ".join(parts[:BINOC_MAX_INPUT_WORDS])
                f.write(_json.dumps({"id": sid, "text": text}) + "\n")
                n_written += 1
        cur.close()
    finally:
        conn.close()

    size_mb = output.stat().st_size / (1024 * 1024)
    print(
        f"[dump-speeches] wrote {n_written:,} speeches to {output} "
        f"({size_mb:.1f} MB); skipped {n_skipped_empty:,} empty.",
        file=sys.stderr,
    )
    return 0


def cmd_load_fdgpt(args) -> int:
    """Bulk-load JSONL of {id, score, is_ai} from the Fast-DetectGPT Modal
    run into public.speech_ai_scores. Mirrors cmd_load_binoculars."""
    import json as _json

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    model_version = args.model_version or "fdgpt-v1+gpt-neo-2.7b"

    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_session(autocommit=False)
    n_loaded = 0
    n_flagged = 0
    n_invalid = 0
    try:
        with conn.cursor() as cur:
            buf: list[tuple] = []

            def _flush():
                nonlocal buf
                if not buf:
                    return
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO public.speech_ai_scores
                        (speech_id, fdgpt_score, fdgpt_is_ai,
                         model_version, run_at)
                    VALUES %s
                    ON CONFLICT (speech_id) DO UPDATE SET
                        fdgpt_score = EXCLUDED.fdgpt_score,
                        fdgpt_is_ai = EXCLUDED.fdgpt_is_ai,
                        model_version = CASE
                            WHEN public.speech_ai_scores.model_version IS NULL THEN EXCLUDED.model_version
                            WHEN public.speech_ai_scores.model_version LIKE '%%' || EXCLUDED.model_version || '%%' THEN public.speech_ai_scores.model_version
                            ELSE public.speech_ai_scores.model_version || '+' || EXCLUDED.model_version
                        END,
                        run_at = EXCLUDED.run_at
                    """,
                    buf,
                    template="(%s, %s, %s, %s, now())",
                    page_size=500,
                )
                conn.commit()
                buf = []

            with input_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except _json.JSONDecodeError:
                        n_invalid += 1
                        continue
                    sid = rec.get("id")
                    score = rec.get("score")
                    if sid is None or score is None:
                        n_invalid += 1
                        continue
                    is_ai = bool(rec.get("is_ai", score > 0.0))
                    if is_ai:
                        n_flagged += 1
                    buf.append((int(sid), float(score), is_ai, model_version))
                    if len(buf) >= 500:
                        _flush()
                        n_loaded += 500
            if buf:
                n_loaded += len(buf)
                _flush()
    finally:
        conn.close()

    print(
        f"[load-fdgpt] loaded {n_loaded:,} scores; "
        f"{n_flagged:,} flagged at fdgpt_is_ai=true; "
        f"{n_invalid:,} invalid lines skipped.",
        file=sys.stderr,
    )
    return 0


def cmd_load_binoculars(args) -> int:
    """Bulk-load JSONL of {id, score, is_ai} from the GPU host into
    public.speech_ai_scores."""
    import json as _json

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    model_version = (
        args.model_version
        or "binoculars-v1+falcon-7b"  # caller can override with full pinned SHAs.
    )

    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_session(autocommit=False)
    n_loaded = 0
    n_flagged = 0
    n_invalid = 0
    try:
        with conn.cursor() as cur:
            buf: list[tuple] = []

            def _flush():
                nonlocal buf
                if not buf:
                    return
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO public.speech_ai_scores
                        (speech_id, binoculars_score, is_ai_at_low_fpr,
                         model_version, run_at)
                    VALUES %s
                    ON CONFLICT (speech_id) DO UPDATE SET
                        binoculars_score = EXCLUDED.binoculars_score,
                        is_ai_at_low_fpr = EXCLUDED.is_ai_at_low_fpr,
                        model_version = CASE
                            WHEN public.speech_ai_scores.model_version IS NULL THEN EXCLUDED.model_version
                            WHEN public.speech_ai_scores.model_version LIKE '%%' || EXCLUDED.model_version || '%%' THEN public.speech_ai_scores.model_version
                            ELSE public.speech_ai_scores.model_version || '+' || EXCLUDED.model_version
                        END,
                        run_at = EXCLUDED.run_at
                    """,
                    buf,
                    template="(%s, %s, %s, %s, now())",
                    page_size=500,
                )
                conn.commit()
                buf = []

            with input_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except _json.JSONDecodeError:
                        n_invalid += 1
                        continue
                    sid = rec.get("id")
                    score = rec.get("score")
                    if sid is None or score is None:
                        n_invalid += 1
                        continue
                    is_ai = bool(rec.get("is_ai", score < BINOC_LOW_FPR_THRESHOLD))
                    if is_ai:
                        n_flagged += 1
                    buf.append((int(sid), float(score), is_ai, model_version))
                    if len(buf) >= 500:
                        _flush()
                        n_loaded += 500
            if buf:
                n_loaded += len(buf)
                _flush()
    finally:
        conn.close()

    print(
        f"[load-binoculars] loaded {n_loaded:,} scores; "
        f"{n_flagged:,} flagged at score < {BINOC_LOW_FPR_THRESHOLD}; "
        f"{n_invalid:,} invalid lines skipped.",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# LLM-as-judge — Claude Haiku 4.5 via Anthropic Batch API
# ---------------------------------------------------------------------------
#
# Sends a curated subset of speeches to Claude for a structured 0-10 verdict
# on AI-drafting evidence. Uses the Batch API (50% discount) and prompt
# caching (cached system prompt + JSON-schema response). Total cost on the
# default 7k subset: ~$5 with Haiku 4.5.
#
# Subset composition:
#   * Stratified sample: --sample-size speeches per quarter, drawn at random.
#     Drives the corpus-wide pre/post comparison.
#   * Flagged-MP focus: all post-Nov-2022 speeches from the top --top-mps MPs
#     by stylometric max_distance. Drives per-MP claims.
#
# Flow:
#   1. cmd_judge selects speech IDs via SQL (read-only, idempotent).
#   2. Builds Anthropic batch requests with cached system prompt + structured
#      JSON output schema, submits via client.messages.batches.create().
#   3. Polls until processing_status == 'ended'.
#   4. Streams results back via client.messages.batches.results(), parses
#      the JSON verdicts, bulk-upserts into public.speech_ai_scores.
#
# Resumable in two ways: --batch-id <id> skips submission and just
# polls/loads an existing batch; speeches with non-NULL llm_judge_score are
# skipped on re-runs.

JUDGE_DEFAULT_MODEL = "claude-haiku-4-5"
# Claude's `reason` field on a 0-10 verdict can be 200-400 tokens of
# justification. 200 was too small — half the responses got truncated
# mid-string and broke JSON parsing. 600 is the comfortable ceiling.
JUDGE_MAX_OUTPUT_TOKENS = 600
JUDGE_BATCH_WRITE_BATCH = 200       # DB upsert page size.
JUDGE_TRUNCATE_WORDS = 600          # Match BINOC_MAX_INPUT_WORDS.
JUDGE_POLL_INTERVAL_S = 30          # How often to poll batch status.
# Anthropic batch body-size sweet spot. ~9k requests at ~800 bytes each
# would be a single ~7-8MB POST; we've seen mid-response chunked-read
# errors at that size. Keep each batch under ~3MB.
JUDGE_CHUNK_SIZE = 2500

JUDGE_SYSTEM_PROMPT = """You evaluate parliamentary speeches for evidence of AI drafting (e.g. ChatGPT, Claude, Gemini).

Context: Australian federal parliament, transcribed by Hansard editors. Speeches may be:
- Entirely human (most common; off-the-cuff or staffer-drafted)
- Edited from a human draft using AI for polishing
- Drafted by AI then human-edited

Score evidence of AI drafting on a 0-10 scale:
- 0-2: Clearly human (idiosyncratic phrasing, electorate-specific references, off-the-cuff feel)
- 3-5: Probably human (formal but no clear AI tells)
- 6-7: Possibly AI-assisted (some structural smoothness, ambiguous tells)
- 8-10: Likely AI-drafted (templatey structure, "delve/tapestry/realm/leverage", lacks human idiosyncrasy)

You will receive one speech and must respond with a JSON verdict only."""

JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        # Anthropic structured outputs don't accept `minimum`/`maximum`
        # on integer types — we clamp 0-10 client-side after parsing.
        "score": {"type": "integer"},
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "reason": {"type": "string"},
    },
    "required": ["score", "confidence", "reason"],
    "additionalProperties": False,
}


def _select_judge_speeches(
    conn,
    sample_size: int,
    top_mps: int,
    date_from: str | None = None,
    full_corpus: bool = False,
) -> list[tuple[int, str]]:
    """Return [(speech_id, text), ...] for the LLM-judge subset.

    Three modes:
      * full_corpus=True: select EVERY speech matching scope (commonwealth
        + word_count > MIN_WORD_COUNT + date >= date_from-or-START_DATE).
        For full-corpus runs (~60k+ post-ChatGPT speeches on Sonnet).
      * full_corpus=False (default): two pools, deduped:
        - Stratified random sample of sample_size speeches per year-quarter.
        - All post-Nov-2022 speeches from the top top_mps MPs by stylometric
          max_distance.
    """
    effective_date = date_from or START_DATE
    ids: set[int] = set()
    cur = conn.cursor()

    if full_corpus:
        # Exhaustive mode — bypass stratified sampling and flagged-MP focus.
        cur.execute(
            """
            SELECT id
            FROM public.speeches
            WHERE jurisdiction = %s
              AND word_count > %s
              AND date >= %s
            """,
            (JURISDICTION, MIN_WORD_COUNT, effective_date),
        )
        for (sid,) in cur.fetchall():
            ids.add(int(sid))
        print(
            f"[judge] full-corpus mode: {len(ids):,} speeches "
            f"(date >= {effective_date})",
            file=sys.stderr,
        )
    else:
        # ---- Stratified sample ----
        cur.execute(
            """
            SELECT id
            FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY date_trunc('quarter', date)
                           ORDER BY random()
                       ) AS rn
                FROM public.speeches
                WHERE jurisdiction = %s
                  AND word_count > %s
                  AND date >= %s
            ) t
            WHERE rn <= %s
            """,
            (JURISDICTION, MIN_WORD_COUNT, effective_date, sample_size),
        )
        for (sid,) in cur.fetchall():
            ids.add(int(sid))
        n_strat = len(ids)
        print(
            f"[judge] stratified sample: {n_strat:,} speeches "
            f"(date >= {effective_date})",
            file=sys.stderr,
        )

    # ---- Flagged-MP focus ---------------------------------------------
    if top_mps > 0 and not full_corpus:
        # Lazy-import numpy/scipy/ruptures only if we need the analyzer.
        try:
            import numpy as np
            from scipy.stats import chi2
            import ruptures as rpt
        except ImportError as exc:
            print(
                f"[judge] WARNING: missing research dep ({exc.name}); "
                "skipping flagged-MP pool, using stratified sample only.",
                file=sys.stderr,
            )
        else:
            fetch_cur = conn.cursor(name="judge_stylo_fetch")
            fetch_cur.itersize = 5000
            fetch_cur.execute(
                """
                SELECT s.name_id,
                       date_trunc('quarter', s.date)::date AS quarter,
                       a.stylo_features
                FROM public.speeches s
                JOIN public.speech_ai_scores a ON a.speech_id = s.id
                WHERE s.jurisdiction = %s
                  AND s.word_count > %s
                  AND s.date >= %s
                  AND s.name_id IS NOT NULL
                  AND a.stylo_features IS NOT NULL
                """,
                (JURISDICTION, MIN_WORD_COUNT, START_DATE),
            )
            per_mp: dict = {}
            for name_id, quarter, feats in fetch_cur:
                per_mp.setdefault(name_id, {}).setdefault(quarter, []).append(feats)
            fetch_cur.close()

            results = _analyze_all_mps(per_mp, np, chi2, rpt)
            ranked = sorted(
                (
                    (nid, r) for nid, r in results.items()
                    if r.get("max_distance") is not None
                ),
                key=lambda x: -(x[1]["max_distance"] or 0),
            )[:top_mps]
            top_ids = [nid for nid, _ in ranked]
            print(
                f"[judge] top {len(top_ids)} flagged MPs by stylometric distance: "
                f"{', '.join(top_ids[:5])}{'...' if len(top_ids) > 5 else ''}",
                file=sys.stderr,
            )

            if top_ids:
                cur.execute(
                    """
                    SELECT id
                    FROM public.speeches
                    WHERE jurisdiction = %s
                      AND word_count > %s
                      AND date >= %s
                      AND name_id = ANY(%s)
                    """,
                    (JURISDICTION, MIN_WORD_COUNT, CHATGPT_LAUNCH, top_ids),
                )
                added = 0
                for (sid,) in cur.fetchall():
                    if int(sid) not in ids:
                        ids.add(int(sid))
                        added += 1
                print(
                    f"[judge] flagged-MP focus added {added:,} new speeches "
                    f"(post-{CHATGPT_LAUNCH})",
                    file=sys.stderr,
                )

    # ---- Fetch text for selected ids -----------------------------------
    if not ids:
        return []
    cur.execute(
        """
        SELECT id, speech_text
        FROM public.speeches
        WHERE id = ANY(%s)
        """,
        (list(ids),),
    )
    rows = []
    for sid, text in cur:
        if not text or not text.strip():
            continue
        parts = text.split()
        if len(parts) > JUDGE_TRUNCATE_WORDS:
            text = " ".join(parts[:JUDGE_TRUNCATE_WORDS])
        rows.append((int(sid), text))
    cur.close()
    print(
        f"[judge] total subset: {len(rows):,} speeches "
        f"(after dedup, with non-empty text)",
        file=sys.stderr,
    )
    return rows


def _build_judge_request(speech_id: int, text: str, model: str) -> dict:
    """Build a single Batch API request for one speech."""
    return {
        "custom_id": f"speech-{speech_id}",
        "params": {
            "model": model,
            "max_tokens": JUDGE_MAX_OUTPUT_TOKENS,
            "system": [
                {
                    "type": "text",
                    "text": JUDGE_SYSTEM_PROMPT,
                    # Cache the (large, identical) system prompt across all
                    # batch requests — first request writes, the rest read.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": JUDGE_OUTPUT_SCHEMA,
                }
            },
            "messages": [
                {
                    "role": "user",
                    "content": f"Speech:\n\n{text}",
                }
            ],
        },
    }


def cmd_judge(args) -> int:
    """LLM-as-judge: send curated subset to Claude via Batch API, load verdicts."""
    try:
        import anthropic
    except ImportError:
        print(
            "ERROR: anthropic package not installed. Install with:\n"
            "  pip install -r requirements-research.txt",
            file=sys.stderr,
        )
        return 2

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        return 2

    # 5 retries with exponential backoff covers transient connection drops
    # (the request body for 9k+ batch requests is ~8MB; we've seen mid-
    # response chunked-read errors that leave the SDK uncertain whether
    # the batch was actually created server-side).
    # 10-minute per-request timeout — batch.create POSTing ~8MB can take
    # a while on slow uplinks before Anthropic ACKs.
    anthropic_client = anthropic.Anthropic(max_retries=5, timeout=600.0)

    # ---- Load existing batch results, or submit new batches -------------
    state_path = REPO_ROOT / "reports" / f"judge_batch_ids_{date.today().isoformat()}.txt"

    if args.batch_id:
        # Comma-separated list, OR a path to a file with one ID per line.
        raw = args.batch_id.strip()
        if Path(raw).exists():
            batch_ids = [
                ln.strip() for ln in Path(raw).read_text().splitlines() if ln.strip()
            ]
        else:
            batch_ids = [s.strip() for s in raw.split(",") if s.strip()]
        print(
            f"[judge] resuming with {len(batch_ids)} existing batch(es): "
            f"{', '.join(batch_ids)}",
            file=sys.stderr,
        )
    else:
        # Build the speech subset.
        if os.environ.get("GR_DB_STATEMENT_TIMEOUT_MS", "30000") != "0":
            print(
                "WARNING: GR_DB_STATEMENT_TIMEOUT_MS is not 0. Re-run with "
                "`GR_DB_STATEMENT_TIMEOUT_MS=0` if a query times out.",
                file=sys.stderr,
            )
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_session(readonly=True, autocommit=False)
        try:
            speeches = _select_judge_speeches(
                conn,
                args.sample_size,
                args.top_mps,
                date_from=args.date_from,
                full_corpus=args.full_corpus,
            )
        finally:
            conn.close()

        if args.dry_run:
            print(
                f"[judge] --dry-run: would submit {len(speeches):,} requests "
                f"to {args.model}.",
                file=sys.stderr,
            )
            return 0

        if not speeches:
            print("[judge] nothing to judge.", file=sys.stderr)
            return 0

        # Estimate cost (rough — actual Batch API discount is 50%).
        avg_words = (
            sum(len(t.split()) for _, t in speeches) / len(speeches) if speeches else 0
        )
        est_in_tokens = len(speeches) * (avg_words * 1.3 + 200)
        est_out_tokens = len(speeches) * 80
        # Per-model pricing ($/MTok). Batch API gives 50% off both.
        # Cache savings on the system prompt aren't included here —
        # actual cost will land 10-30% lower.
        model_prices = {
            "claude-haiku-4-5":  (1.0,  5.0),
            "claude-sonnet-4-6": (3.0, 15.0),
            "claude-opus-4-7":   (5.0, 25.0),
        }
        in_price, out_price = model_prices.get(args.model, (1.0, 5.0))
        est_cost = (
            (est_in_tokens / 1_000_000) * in_price * 0.5
            + (est_out_tokens / 1_000_000) * out_price * 0.5
        )
        n_chunks = (len(speeches) + JUDGE_CHUNK_SIZE - 1) // JUDGE_CHUNK_SIZE
        print(
            f"[judge] submitting {len(speeches):,} requests to {args.model} "
            f"in {n_chunks} chunk(s) of up to {JUDGE_CHUNK_SIZE}; "
            f"~{est_in_tokens/1e6:.1f}M input tokens, "
            f"~{est_out_tokens/1e6:.2f}M output tokens; "
            f"estimated cost ~${est_cost:.2f} (batch+cache discount applied).",
            file=sys.stderr,
        )

        if not args.yes:
            prompt = input("Proceed? [y/N] ").strip().lower()
            if prompt != "y":
                print("[judge] aborted.", file=sys.stderr)
                return 1

        requests = [
            _build_judge_request(sid, text, args.model) for sid, text in speeches
        ]
        # Chunked submission. Each chunk is a separate Anthropic batch.
        # Smaller bodies dodge the chunked-read drop we hit at ~8MB.
        state_path.parent.mkdir(parents=True, exist_ok=True)
        batch_ids: list[str] = []
        for i in range(0, len(requests), JUDGE_CHUNK_SIZE):
            chunk = requests[i : i + JUDGE_CHUNK_SIZE]
            chunk_idx = (i // JUDGE_CHUNK_SIZE) + 1
            print(
                f"[judge] submitting chunk {chunk_idx}/{n_chunks} "
                f"({len(chunk)} requests)…",
                file=sys.stderr,
            )
            batch = anthropic_client.messages.batches.create(requests=chunk)
            batch_ids.append(batch.id)
            # Append to state file immediately so a crash mid-loop doesn't
            # lose already-submitted batch IDs.
            with state_path.open("a") as f:
                f.write(batch.id + "\n")
            print(
                f"  → batch_id: {batch.id} "
                f"(status: {batch.processing_status}, "
                f"requests: {batch.request_counts.processing})",
                file=sys.stderr,
            )
        print(
            f"[judge] all {len(batch_ids)} batches submitted. "
            f"State file: {state_path}\n"
            f"[judge] resume any time with: --batch-id "
            f"{','.join(batch_ids)}",
            file=sys.stderr,
        )

    # ---- Poll all batches until each is 'ended' -------------------------
    print(
        f"[judge] polling {len(batch_ids)} batch(es) (every "
        f"{JUDGE_POLL_INTERVAL_S}s)…",
        file=sys.stderr,
    )
    t0 = time.time()
    pending = set(batch_ids)
    final_batches: dict[str, object] = {}
    while pending:
        for bid in list(pending):
            b = anthropic_client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                final_batches[bid] = b
                pending.remove(bid)
        if not pending:
            break
        # Print one-line aggregate status.
        agg_processing = agg_succeeded = agg_errored = 0
        for bid in pending:
            b = anthropic_client.messages.batches.retrieve(bid)
            agg_processing += b.request_counts.processing
            agg_succeeded += b.request_counts.succeeded
            agg_errored += b.request_counts.errored
        print(
            f"  [{(time.time() - t0)/60:.1f} min] {len(pending)} pending — "
            f"processing={agg_processing}, succeeded={agg_succeeded}, "
            f"errored={agg_errored}",
            file=sys.stderr,
        )
        time.sleep(JUDGE_POLL_INTERVAL_S)

    total_succ = sum(b.request_counts.succeeded for b in final_batches.values())
    total_err = sum(b.request_counts.errored for b in final_batches.values())
    total_exp = sum(b.request_counts.expired for b in final_batches.values())
    print(
        f"[judge] all batches ended in {(time.time() - t0)/60:.1f} min. "
        f"succeeded={total_succ}, errored={total_err}, expired={total_exp}.",
        file=sys.stderr,
    )

    # ---- Stream results into the DB -----------------------------------
    write_conn = psycopg2.connect(**DB_CONFIG)
    write_conn.set_session(autocommit=False)
    try:
        cur = write_conn.cursor()
        buf: list[tuple] = []
        n_loaded = 0
        n_errored = 0
        score_hist: dict[int, int] = {}

        def _flush():
            nonlocal buf
            if not buf:
                return
            # Column-prefix parameterisation: when the user passes
            # --save-as opus_judge, we write to opus_judge_* instead of
            # llm_judge_*. Keeps Haiku data intact for cross-model
            # comparison in the report.
            prefix = args.save_as
            score_col = f"{prefix}_score"
            conf_col = f"{prefix}_confidence"
            reason_col = f"{prefix}_reason"
            # llm_judge keeps its model column for backwards compat;
            # opus_judge stores model in `model_version` only.
            has_model_col = (prefix == "llm_judge")
            if has_model_col:
                model_col = f"{prefix}_model"
                psycopg2.extras.execute_values(
                    cur,
                    f"""
                    INSERT INTO public.speech_ai_scores
                        (speech_id, {score_col}, {conf_col},
                         {reason_col}, {model_col}, run_at)
                    VALUES %s
                    ON CONFLICT (speech_id) DO UPDATE SET
                        {score_col}  = EXCLUDED.{score_col},
                        {conf_col}   = EXCLUDED.{conf_col},
                        {reason_col} = EXCLUDED.{reason_col},
                        {model_col}  = EXCLUDED.{model_col},
                        model_version = CASE
                            WHEN public.speech_ai_scores.model_version IS NULL THEN EXCLUDED.{model_col}
                            WHEN public.speech_ai_scores.model_version LIKE '%%' || EXCLUDED.{model_col} || '%%' THEN public.speech_ai_scores.model_version
                            ELSE public.speech_ai_scores.model_version || '+' || EXCLUDED.{model_col}
                        END,
                        run_at = EXCLUDED.run_at
                    """,
                    buf,
                    template="(%s, %s, %s, %s, %s, now())",
                    page_size=JUDGE_BATCH_WRITE_BATCH,
                )
            else:
                # opus_judge — no separate _model column. Track which
                # model produced this score via model_version. We use
                # executemany rather than execute_values because the
                # ON CONFLICT clause needs the model_str passed multiple
                # times (which doesn't fit cleanly into the VALUES list).
                upsert_sql = f"""
                    INSERT INTO public.speech_ai_scores
                        (speech_id, {score_col}, {conf_col},
                         {reason_col}, run_at)
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT (speech_id) DO UPDATE SET
                        {score_col}  = EXCLUDED.{score_col},
                        {conf_col}   = EXCLUDED.{conf_col},
                        {reason_col} = EXCLUDED.{reason_col},
                        model_version = CASE
                            WHEN public.speech_ai_scores.model_version IS NULL THEN %s
                            WHEN public.speech_ai_scores.model_version LIKE '%%' || %s || '%%' THEN public.speech_ai_scores.model_version
                            ELSE public.speech_ai_scores.model_version || '+' || %s
                        END,
                        run_at = EXCLUDED.run_at
                """
                # buf rows: (sid, score, conf, reason, model_str)
                cur.executemany(upsert_sql, [
                    (sid, score, conf, reason,
                     model_str, model_str, model_str)
                    for sid, score, conf, reason, model_str in buf
                ])
            write_conn.commit()
            buf = []

        for bid in batch_ids:
            print(f"[judge] streaming results from {bid}…", file=sys.stderr)
            for result in anthropic_client.messages.batches.results(bid):
                custom_id = result.custom_id
                if not custom_id.startswith("speech-"):
                    continue
                try:
                    speech_id = int(custom_id.removeprefix("speech-"))
                except ValueError:
                    continue
                if result.result.type != "succeeded":
                    n_errored += 1
                    continue
                msg = result.result.message
                text_block = next(
                    (b for b in msg.content if getattr(b, "type", None) == "text"),
                    None,
                )
                if text_block is None:
                    n_errored += 1
                    continue
                # Lenient parse: try strict JSON first, then fall back to
                # regex extraction. Earlier runs lost ~half the verdicts
                # to JSON truncation when `max_tokens` was too small —
                # score and confidence land before `reason` in the
                # rendered JSON, so a truncated tail is still salvageable.
                raw = text_block.text
                score = confidence = None
                reason = ""
                try:
                    verdict = json.loads(raw)
                    score = max(0, min(10, int(verdict["score"])))
                    confidence = str(verdict.get("confidence", ""))[:16]
                    reason = str(verdict.get("reason", ""))[:512]
                except (json.JSONDecodeError, KeyError, ValueError):
                    m_score = re.search(r'"score"\s*:\s*(-?\d+)', raw)
                    m_conf = re.search(
                        r'"confidence"\s*:\s*"(low|medium|high)"', raw
                    )
                    m_reason = re.search(r'"reason"\s*:\s*"([^"]{0,500})', raw)
                    if m_score and m_conf:
                        score = max(0, min(10, int(m_score.group(1))))
                        confidence = m_conf.group(1)
                        reason = (m_reason.group(1) if m_reason else "")[:512]
                if score is None or confidence is None:
                    n_errored += 1
                    continue
                score_hist[score] = score_hist.get(score, 0) + 1
                buf.append(
                    (
                        speech_id,
                        score,
                        confidence,
                        reason,
                        msg.model,
                    )
                )
                if len(buf) >= JUDGE_BATCH_WRITE_BATCH:
                    _flush()
                    n_loaded += JUDGE_BATCH_WRITE_BATCH
        if buf:
            n_loaded += len(buf)
            _flush()

        cur.close()
    finally:
        write_conn.close()

    print(
        f"\n[judge] loaded {n_loaded:,} verdicts; {n_errored:,} errored/unparseable.",
        file=sys.stderr,
    )
    if score_hist:
        print(f"[judge] score distribution:", file=sys.stderr)
        for s in sorted(score_hist):
            n = score_hist[s]
            bar = "█" * int(n / max(score_hist.values()) * 30)
            print(f"  {s:>2}: {n:>5,} {bar}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------


def cmd_volume(args) -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"ai_detection_volume_{date.today().isoformat()}.md"
    conn = connect()
    try:
        with conn.cursor() as cur:
            report = generate_volume_report(cur)
    finally:
        conn.close()
    out_path.write_text(report)
    print(report)
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="phase")

    p_volume = sub.add_parser(
        "volume",
        help="Step 0: read-only volume sanity-check report.",
    )
    p_volume.set_defaults(func=cmd_volume)

    p_stylo = sub.add_parser(
        "stylo-features",
        help="Phase B step 1: extract stylometric features per speech.",
    )
    p_stylo.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N speeches this run (useful for sampling/testing).",
    )
    p_stylo.set_defaults(func=cmd_stylo_features)

    p_cp = sub.add_parser(
        "change-point",
        help="Phase B step 2: per-MP Mahalanobis + Pelt change-point flagging.",
    )
    p_cp.set_defaults(func=cmd_change_point)

    p_dump = sub.add_parser(
        "dump-speeches",
        help="Phase A bookend: dump JSONL of (id, text) for shipping to GPU host.",
    )
    p_dump.add_argument(
        "--output",
        required=True,
        help="Output JSONL path. Will be overwritten.",
    )
    p_dump.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap output to N speeches (for testing).",
    )
    p_dump.set_defaults(func=cmd_dump_speeches)

    p_load = sub.add_parser(
        "load-binoculars",
        help="Phase A bookend: load JSONL of {id, score} from GPU host into DB.",
    )
    p_load.add_argument(
        "--input",
        required=True,
        help="JSONL path produced by scripts/research_ai_binoculars.py.",
    )
    p_load.add_argument(
        "--model-version",
        default=None,
        help="Pinned model identifier to record in speech_ai_scores.model_version "
             "(e.g. 'binoculars@<sha>+falcon-7b@<rev>'). Defaults to "
             "'binoculars-v1+falcon-7b'.",
    )
    p_load.set_defaults(func=cmd_load_binoculars)

    p_load_fdgpt = sub.add_parser(
        "load-fdgpt",
        help="Load JSONL of {id, score, is_ai} from the Fast-DetectGPT Modal run.",
    )
    p_load_fdgpt.add_argument(
        "--input",
        required=True,
        help="JSONL path produced by scripts/research_ai_fast_detect_gpt_modal.py.",
    )
    p_load_fdgpt.add_argument(
        "--model-version",
        default=None,
        help="Pinned model identifier to record (default 'fdgpt-v1+gpt-neo-2.7b').",
    )
    p_load_fdgpt.set_defaults(func=cmd_load_fdgpt)

    p_report = sub.add_parser(
        "report",
        help="Phase C: cross-check + final markdown report with charts.",
    )
    p_report.set_defaults(func=cmd_report)

    p_judge = sub.add_parser(
        "judge",
        help="LLM-as-judge via Claude Batch API on a curated subset.",
    )
    p_judge.add_argument(
        "--sample-size", type=int, default=200,
        help="Stratified sample size per quarter (default 200 → ~5000 total).",
    )
    p_judge.add_argument(
        "--top-mps", type=int, default=10,
        help="Score all post-Nov-2022 speeches from top N stylometric-flagged MPs.",
    )
    p_judge.add_argument(
        "--model", default=JUDGE_DEFAULT_MODEL,
        help=f"Claude model ID (default {JUDGE_DEFAULT_MODEL}).",
    )
    p_judge.add_argument(
        "--batch-id", default=None,
        help="Skip submission; poll/load existing Anthropic batch ID(s). "
             "Accepts a single ID, comma-separated IDs, or a path to a "
             "newline-separated state file (one ID per line).",
    )
    p_judge.add_argument(
        "--dry-run", action="store_true",
        help="Print the subset size and estimated cost; don't submit.",
    )
    p_judge.add_argument(
        "--yes", action="store_true",
        help="Skip cost-confirmation prompt before submitting.",
    )
    p_judge.add_argument(
        "--save-as", default="llm_judge",
        choices=["llm_judge", "opus_judge", "sonnet_judge"],
        help="Which DB column prefix to write to. 'llm_judge' = Haiku, "
             "'opus_judge' = Opus 4.7 columns, 'sonnet_judge' = Sonnet 4.6 "
             "columns (each requires its respective migration).",
    )
    p_judge.add_argument(
        "--full-corpus", action="store_true",
        help="Bypass stratified sampling; select EVERY speech matching the "
             "scope filter. Use with --date-from to scope. Cost is much higher "
             "(score on tens of thousands of speeches rather than ~10k sample).",
    )
    p_judge.add_argument(
        "--date-from", default=None,
        help="Override the default 2018-01-01 scope start. e.g. '2022-11-30' "
             "to score only post-ChatGPT speeches.",
    )
    p_judge.set_defaults(func=cmd_judge)

    args = parser.parse_args()
    if not getattr(args, "phase", None):
        # Default subcommand: volume sanity-check. Backwards-compatible with
        # the original `--dry-run`-only entry point.
        return cmd_volume(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
