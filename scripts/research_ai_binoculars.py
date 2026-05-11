#!/usr/bin/env python3
"""Binoculars detector wrapper — runs on a rented GPU host, NOT on the VPS.

This is Phase A of the AI-drafted speech detection research analysis. It is a
standalone script with NO dependency on the checkhansard codebase or the GR
Postgres DB — it reads JSONL of `{"id": int, "text": str}` from a file (or
stdin) and writes JSONL of `{"id": int, "score": float, "is_ai": bool}` to a
file (or stdout). The driver script (research_ai_detection.py) handles the
DB-side dump and load steps.

Why a separate file: the upstream Binoculars repo
(https://github.com/ahans30/Binoculars) loads two ~7B parameter Falcon
models (~28GB VRAM in fp16) which must run on a GPU. The VPS does not have
a GPU, so this code never runs there. Keeping it separate also means
research_ai_detection.py imports cleanly without `transformers` or `torch`
installed locally.

Setup on a fresh A6000/A100 host (Runpod / Lambda Labs / Modal):

    git clone https://github.com/ahans30/Binoculars.git
    cd Binoculars && pip install -e .
    # (Binoculars pulls in transformers, torch, accelerate.)

    # Ship the speeches file from the VPS:
    scp speeches.jsonl gpu-host:/tmp/

    # On the GPU host:
    python research_ai_binoculars.py \\
        --input-file /tmp/speeches.jsonl \\
        --output-file /tmp/scores.jsonl

    # Ship results back:
    scp gpu-host:/tmp/scores.jsonl ./

Then locally:

    .venv-research/bin/python scripts/research_ai_detection.py load-binoculars \\
        --input /tmp/scores.jsonl

Privacy note: speech text is public Hansard, but the dump file ships only
(id, text) — no name_id — so the GPU host never sees speaker attribution.
The id-to-name_id join happens locally on load.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Binoculars threshold from the paper: scores below this are flagged as AI
# at a low (~0.01%) false-positive rate on the paper's eval set. Whether
# this transfers to parliamentary register is one of the questions the
# manual 100-speech audit will answer in Phase C.
LOW_FPR_THRESHOLD = 0.901

# Truncate inputs to this many words before scoring. Binoculars internally
# uses a max_observed_token window; passing huge texts wastes compute. Most
# stylistic AI signal is in the first ~600 words anyway. Keep this in sync
# with the dump-speeches step in research_ai_detection.py.
MAX_INPUT_WORDS = 600


def _truncate(text: str, max_words: int = MAX_INPUT_WORDS) -> str:
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="JSONL file with {id, text} per line. Use - for stdin.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="JSONL file to write {id, score, is_ai}. Use - for stdout.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=LOW_FPR_THRESHOLD,
        help=f"Score below this is flagged as AI. Default {LOW_FPR_THRESHOLD} "
        "(paper's low-FPR threshold).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="Print progress every N speeches.",
    )
    args = parser.parse_args()

    # Lazy import — only fail with a useful message if torch/transformers
    # aren't installed (i.e. this script was run on the VPS by mistake).
    try:
        from binoculars import Binoculars  # type: ignore[import-not-found]
    except ImportError as exc:
        print(
            f"ERROR: cannot import Binoculars ({exc}). This script is meant "
            "to run on a GPU host with the upstream Binoculars package "
            "installed:\n"
            "  git clone https://github.com/ahans30/Binoculars.git\n"
            "  cd Binoculars && pip install -e .",
            file=sys.stderr,
        )
        return 2

    print("[binoculars] loading Falcon-7B base + instruct (≈28GB VRAM)…",
          file=sys.stderr)
    t_load = time.time()
    bino = Binoculars()
    print(f"[binoculars] models loaded in {time.time() - t_load:.1f}s.",
          file=sys.stderr)

    # Streaming I/O so we never hold the full corpus in memory.
    if str(args.input_file) == "-":
        in_f = sys.stdin
        in_close = False
    else:
        in_f = args.input_file.open("r")
        in_close = True

    if str(args.output_file) == "-":
        out_f = sys.stdout
        out_close = False
    else:
        out_f = args.output_file.open("w")
        out_close = True

    n_done = 0
    n_flagged = 0
    t0 = time.time()
    try:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[binoculars] skipping unparseable line: {line[:80]!r}",
                      file=sys.stderr)
                continue
            text = _truncate(rec.get("text", "") or "")
            if not text:
                continue
            try:
                score = float(bino.compute_score(text))
            except Exception as exc:
                print(f"[binoculars] error scoring id={rec.get('id')}: {exc}",
                      file=sys.stderr)
                continue
            is_ai = score < args.threshold
            if is_ai:
                n_flagged += 1
            out_f.write(json.dumps({
                "id": rec["id"],
                "score": score,
                "is_ai": is_ai,
            }) + "\n")
            out_f.flush()
            n_done += 1
            if n_done % args.log_interval == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed else 0
                print(
                    f"[binoculars] {n_done:,} done "
                    f"({rate:.1f} sp/s, {n_flagged} flagged @ <{args.threshold})",
                    file=sys.stderr,
                )
    finally:
        if in_close:
            in_f.close()
        if out_close:
            out_f.close()

    elapsed = time.time() - t0
    print(
        f"[binoculars] done: {n_done:,} speeches scored in "
        f"{elapsed/60:.1f} min ({n_done/max(1,elapsed):.1f}/s); "
        f"{n_flagged} flagged at score < {args.threshold}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
