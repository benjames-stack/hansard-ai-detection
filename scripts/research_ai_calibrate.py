#!/usr/bin/env python3
"""Calibration test for the Hansard AI-detection pipeline.

Question: do our detectors catch known-AI speeches in parliamentary register?

Method:
1. Pick N pre-ChatGPT (pre-Nov 2022) Hansard speeches as guaranteed-human
   controls. Stratified by source_type to mirror the production set.
2. For each, generate an AI counterpart by asking Claude Sonnet 4.6 to
   write a speech on the same debate topic in the same parliamentary
   style. Sonnet is the realistic adversary (more polished than Haiku;
   less obvious than Opus).
3. Score all 2N speeches with the same Haiku 4.5 judge prompt as the
   production run, so the calibration uses the same yardstick.
4. Compare score distributions.

Interpretation:
- If known-AI is flagged at ≥8 ~70-80% and known-human at ~10%, the
  detector works in this register and the production "703 flagged" is
  a real signal.
- If both are flagged at ~10%, the detector is blind in this register
  and the true AI rate is much higher than 7%.
- Anything in between tells us the detector's sensitivity.

Cost: ~$0.55 for N=50.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_config import DB_CONFIG  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

GEN_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-haiku-4-5"
PRE_CHATGPT_CUTOFF = "2022-11-30"

GEN_USER_PROMPT = """Write a speech in the style of an Australian federal MP delivering it in the House of Representatives or Senate. The speech should be 500-700 words.

Debate topic: {topic}

Style guidance:
- Open with appropriate parliamentary address (e.g. "Madam Speaker," "Mr President," "Madam Deputy Speaker").
- Use Australian parliamentary rhetorical conventions: "the member opposite", "this side of the chamber", references to constituents.
- First person. Take a clear stance on the topic (advocate or oppose).
- Use the speech to make a substantive policy argument, not just procedural language.

Output the speech in plain text only — no stage directions, no annotations, no headings. Just the speech as it would appear in Hansard."""

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
        "score": {"type": "integer"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reason": {"type": "string"},
    },
    "required": ["score", "confidence", "reason"],
    "additionalProperties": False,
}


def _select_human_speeches(n: int) -> list[dict]:
    """Pull N random pre-ChatGPT Hansard speeches with their topic title."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_session(readonly=True, autocommit=False)
    try:
        with conn.cursor() as c:
            c.execute(
                """
                SELECT id, date, COALESCE(debate_title, ''), speech_text
                FROM public.speeches
                WHERE jurisdiction = 'commonwealth'
                  AND word_count > 200
                  AND date >= '2018-01-01'
                  AND date < %s
                  AND source_type = 'chamber'
                  AND debate_title IS NOT NULL
                  AND length(debate_title) > 10
                ORDER BY random()
                LIMIT %s
                """,
                (PRE_CHATGPT_CUTOFF, n),
            )
            rows = []
            for sid, dt, topic, text in c:
                # Truncate text to 600 words to match production scoring window.
                parts = text.split()
                if len(parts) > 600:
                    text = " ".join(parts[:600])
                rows.append({
                    "id": int(sid),
                    "date": str(dt),
                    "topic": topic,
                    "text": text,
                })
            return rows
    finally:
        conn.close()


def _generate_ai_speech(client, topic: str, retries: int = 3) -> str | None:
    """Call Claude Sonnet to write a speech on the given topic."""
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model=GEN_MODEL,
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": GEN_USER_PROMPT.format(topic=topic),
                }],
            )
            text_block = next(
                (b for b in r.content if getattr(b, "type", None) == "text"),
                None,
            )
            if text_block:
                return text_block.text.strip()
        except Exception as exc:
            print(f"  [gen] attempt {attempt+1} failed: {exc}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def _judge_speech(client, text: str, judge_model: str, retries: int = 3) -> dict | None:
    """Score one speech with the same Haiku judge prompt as production."""
    parts = text.split()
    if len(parts) > 600:
        text = " ".join(parts[:600])
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model=judge_model,
                max_tokens=600,
                system=[{"type": "text", "text": JUDGE_SYSTEM_PROMPT}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": JUDGE_OUTPUT_SCHEMA,
                    }
                },
                messages=[{"role": "user", "content": f"Speech:\n\n{text}"}],
            )
            text_block = next(
                (b for b in r.content if getattr(b, "type", None) == "text"),
                None,
            )
            if not text_block:
                continue
            raw = text_block.text
            try:
                v = json.loads(raw)
                return {
                    "score": max(0, min(10, int(v["score"]))),
                    "confidence": str(v.get("confidence", ""))[:16],
                    "reason": str(v.get("reason", ""))[:512],
                }
            except (json.JSONDecodeError, KeyError, ValueError):
                m_score = re.search(r'"score"\s*:\s*(-?\d+)', raw)
                m_conf = re.search(
                    r'"confidence"\s*:\s*"(low|medium|high)"', raw
                )
                if m_score and m_conf:
                    return {
                        "score": max(0, min(10, int(m_score.group(1)))),
                        "confidence": m_conf.group(1),
                        "reason": "(parse-fallback)",
                    }
        except Exception as exc:
            print(f"  [judge] attempt {attempt+1} failed: {exc}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=50,
                        help="Number of human/AI speech pairs (default 50).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--output", default=None,
                        help="JSONL output path (default reports/calibration_<date>.jsonl).")
    parser.add_argument("--judge-model", default="claude-haiku-4-5",
                        help="Model that scores both human and AI speeches "
                             "(default claude-haiku-4-5; pass "
                             "claude-opus-4-7 to validate the Opus production "
                             "run).")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Install with:\n"
              "  pip install -r requirements-research.txt", file=sys.stderr)
        return 2

    random.seed(args.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import date as _date
    out_path = Path(args.output) if args.output else (
        REPORTS_DIR / f"calibration_{_date.today().isoformat()}.jsonl"
    )

    client = anthropic.Anthropic(max_retries=5, timeout=300.0)

    # Stage 1: pull human speeches
    print(f"[calibrate] selecting {args.n} pre-ChatGPT human speeches…",
          file=sys.stderr)
    human = _select_human_speeches(args.n)
    if not human:
        print("[calibrate] ERROR: no speeches selected.", file=sys.stderr)
        return 1
    print(f"[calibrate] got {len(human)} human speeches.", file=sys.stderr)

    # Stage 2: generate AI counterparts
    print(f"[calibrate] generating {len(human)} AI counterparts via {GEN_MODEL}…",
          file=sys.stderr)
    pairs = []
    t0 = time.time()
    for i, h in enumerate(human, 1):
        ai_text = _generate_ai_speech(client, h["topic"])
        if not ai_text:
            print(f"  [{i}/{len(human)}] generation failed for id={h['id']}",
                  file=sys.stderr)
            continue
        pairs.append({"human": h, "ai_text": ai_text})
        if i % 10 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            print(f"  [{i}/{len(human)}] gen rate {rate:.1f}/s "
                  f"(eta {(len(human)-i)/rate:.0f}s)", file=sys.stderr)
    print(f"[calibrate] generated {len(pairs)} AI speeches.", file=sys.stderr)

    # Stage 3: judge both human and AI versions
    print(f"[calibrate] judging {len(pairs)*2} speeches via {args.judge_model}…",
          file=sys.stderr)
    results = []
    for i, p in enumerate(pairs, 1):
        h_verdict = _judge_speech(client, p["human"]["text"], args.judge_model)
        a_verdict = _judge_speech(client, p["ai_text"], args.judge_model)
        if h_verdict and a_verdict:
            results.append({
                "id": p["human"]["id"],
                "date": p["human"]["date"],
                "topic": p["human"]["topic"],
                "human_score": h_verdict["score"],
                "human_confidence": h_verdict["confidence"],
                "human_reason": h_verdict["reason"],
                "ai_score": a_verdict["score"],
                "ai_confidence": a_verdict["confidence"],
                "ai_reason": a_verdict["reason"],
            })
        if i % 10 == 0:
            print(f"  [{i}/{len(pairs)}] judging…", file=sys.stderr)

    # Persist
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[calibrate] wrote {len(results)} pairs to {out_path}.",
          file=sys.stderr)

    # Stage 4: report
    if not results:
        print("[calibrate] no usable results.", file=sys.stderr)
        return 1

    print("\n=== CALIBRATION RESULTS ===\n")
    h_scores = [r["human_score"] for r in results]
    a_scores = [r["ai_score"] for r in results]

    def _hist(label, scores):
        print(f"\n{label} (n={len(scores)}, mean={sum(scores)/len(scores):.2f}):")
        max_n = 0
        bins = [0] * 11
        for s in scores:
            bins[s] += 1
        max_n = max(bins) or 1
        for i, n in enumerate(bins):
            bar = "█" * int(n / max_n * 30)
            print(f"  {i:>2}: {n:>3} {bar}")

    _hist("Human (pre-ChatGPT real Hansard)", h_scores)
    _hist(f"AI ({GEN_MODEL}-generated)", a_scores)

    h_flagged_8 = sum(1 for s in h_scores if s >= 8)
    a_flagged_8 = sum(1 for s in a_scores if s >= 8)
    h_flagged_6 = sum(1 for s in h_scores if s >= 6)
    a_flagged_6 = sum(1 for s in a_scores if s >= 6)

    print(f"\nFlag rates at threshold ≥6 (possibly AI):")
    print(f"  human: {h_flagged_6}/{len(h_scores)} = {h_flagged_6/len(h_scores)*100:.0f}%")
    print(f"  ai:    {a_flagged_6}/{len(a_scores)} = {a_flagged_6/len(a_scores)*100:.0f}%")
    print(f"\nFlag rates at threshold ≥8 (likely AI):")
    print(f"  human: {h_flagged_8}/{len(h_scores)} = {h_flagged_8/len(h_scores)*100:.0f}%")
    print(f"  ai:    {a_flagged_8}/{len(a_scores)} = {a_flagged_8/len(a_scores)*100:.0f}%")

    # Verdict
    sensitivity = a_flagged_8 / max(1, len(a_scores))
    fpr = h_flagged_8 / max(1, len(h_scores))
    print(f"\n=== INTERPRETATION ===")
    print(f"  Sensitivity (true-AI flagged at ≥8): {sensitivity*100:.0f}%")
    print(f"  False-positive rate (human flagged at ≥8): {fpr*100:.0f}%")
    if sensitivity > 0.6:
        print(f"  ✓ Detector WORKS in this register: catches {sensitivity*100:.0f}% of "
              f"known-AI. Production count of 703 flagged is a real signal "
              f"(possibly an undercount by ~{(1/sensitivity):.1f}x if detector is "
              f"linear).")
    elif sensitivity > 0.3:
        print(f"  ~ Detector PARTIAL: catches {sensitivity*100:.0f}% of known-AI. True "
              f"AI rate is likely {(1/sensitivity):.1f}x the production count.")
    else:
        print(f"  ✗ Detector BLIND: catches only {sensitivity*100:.0f}% of known-AI in "
              f"this register. True AI rate could be much higher than the "
              f"production count of 703.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
