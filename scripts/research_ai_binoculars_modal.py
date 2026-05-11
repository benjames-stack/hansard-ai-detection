"""Modal version of the Binoculars wrapper — Phase A on serverless GPU.

This is a Modal app, not a script in the usual sense. You run it with:

    modal run scripts/research_ai_binoculars_modal.py \\
        --input-file /tmp/speeches.jsonl \\
        --output-file /tmp/scores.jsonl

Modal handles everything else: provisioning an A100, building the container
image (cached after first run), loading the Falcon-7B base + instruct
models (cached in a persistent Volume after first download), streaming
speeches up, streaming scores back. No SSH, no scp.

Requirements (one-time, on your local machine):

    pip install modal
    modal token new      # opens a browser, links to your Modal account

Pricing as of 2026: ~$4-5/hr for A100-40GB; full ~250k post-Nov-2022
substantive corpus is roughly 2-4 GPU-hrs (≈$10-20). The Falcon weights
download once on first run (~14GB; persists in the Volume) so subsequent
runs skip it.

Privacy: Modal is a US-based serverless GPU provider. Speech text is
public Hansard but no name_id is sent — the input file shipped here
contains only (id, text). Speaker attribution is rejoined locally.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

# Match the constants in scripts/research_ai_binoculars.py and the driver.
LOW_FPR_THRESHOLD = 0.901
MAX_INPUT_WORDS = 600

# Container image: torch + transformers + accelerate, plus the upstream
# Binoculars repo installed from git.
binoculars_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.40",
        "accelerate>=0.30",
        "sentencepiece",
        "protobuf",
    )
    .pip_install("git+https://github.com/ahans30/Binoculars.git")
)

# Persistent Volume so the ~14GB of Falcon-7B weights download only once
# across all runs of this Modal app.
hf_cache = modal.Volume.from_name(
    "binoculars-hf-cache", create_if_missing=True
)

app = modal.App("checkhansard-binoculars")


@app.cls(
    image=binoculars_image,
    gpu="A100-40GB",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=4 * 3600,        # up to 4 hours per container call
    scaledown_window=300,    # keep container warm 5 min between calls
)
class BinocularsScorer:
    """Wraps the Binoculars detector. Loaded once per container; reused
    across all speeches in a batch (or batches sent via .map())."""

    @modal.enter()
    def load(self):
        """Runs once when the container starts. Downloads weights to the
        Volume on first run; subsequent runs load from the Volume."""
        from binoculars import Binoculars  # type: ignore[import-not-found]
        print("[modal] loading Falcon-7B base + instruct…", flush=True)
        t0 = time.time()
        self.bino = Binoculars()
        print(f"[modal] models loaded in {time.time() - t0:.1f}s", flush=True)

    @modal.method()
    def score_batch(
        self,
        speeches: list[dict],
        threshold: float = LOW_FPR_THRESHOLD,
    ) -> list[dict]:
        """Score a list of {id, text} dicts; return list of
        {id, score, is_ai} (or {id, error} on per-speech failure)."""
        out: list[dict] = []
        for s in speeches:
            sid = s.get("id")
            text = s.get("text") or ""
            # Defensive truncation; the dump-speeches step already does this
            # but a stray full-text input shouldn't blow OOM.
            parts = text.split()
            if len(parts) > MAX_INPUT_WORDS:
                text = " ".join(parts[:MAX_INPUT_WORDS])
            if not text.strip():
                out.append({"id": sid, "error": "empty"})
                continue
            try:
                score = float(self.bino.compute_score(text))
                out.append({
                    "id": sid,
                    "score": score,
                    "is_ai": score < threshold,
                })
            except Exception as exc:
                out.append({"id": sid, "error": str(exc)})
        return out


@app.local_entrypoint()
def main(
    input_file: str,
    output_file: str,
    batch_size: int = 100,
    threshold: float = LOW_FPR_THRESHOLD,
    limit: int = 0,
):
    """Drive the Binoculars scoring run end-to-end.

    Reads JSONL of {id, text} from --input-file, sends in batches to a
    pool of GPU containers via .map(), streams JSONL of {id, score, is_ai}
    to --output-file.

    --batch-size  speeches per container call (default 100; increase to
                  amortise the per-call overhead, decrease to checkpoint
                  more often).
    --threshold   Binoculars score below this is flagged as AI. Default
                  is the paper's low-FPR threshold (0.901).
    --limit       Cap input to first N speeches (0 = no cap). Useful for
                  the smoke-test run before the full corpus.
    """
    in_path = Path(input_file)
    out_path = Path(output_file)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    speeches: list[dict] = []
    with in_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                speeches.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if limit and limit > 0:
        speeches = speeches[: int(limit)]
    print(f"[modal local] {len(speeches):,} speeches to score "
          f"(batch={batch_size}, threshold={threshold})", flush=True)

    if not speeches:
        print("[modal local] nothing to do", file=sys.stderr)
        return

    batches = [
        speeches[i : i + batch_size]
        for i in range(0, len(speeches), batch_size)
    ]
    print(f"[modal local] {len(batches):,} batches; dispatching…", flush=True)

    scorer = BinocularsScorer()
    n_done = 0
    n_flagged = 0
    n_err = 0
    t0 = time.time()
    with out_path.open("w") as out:
        for i, results in enumerate(scorer.score_batch.map(
            batches, kwargs={"threshold": threshold},
        )):
            for r in results:
                out.write(json.dumps(r) + "\n")
                if "error" in r:
                    n_err += 1
                elif r.get("is_ai"):
                    n_flagged += 1
            out.flush()
            n_done += len(results)
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed else 0
            eta = (len(speeches) - n_done) / rate if rate else 0
            print(
                f"[modal local] batch {i + 1}/{len(batches)}: "
                f"{n_done:,}/{len(speeches):,} ({rate:.1f}/s, "
                f"eta {eta / 60:.0f} min, {n_flagged} flagged, "
                f"{n_err} errors)",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"\n[modal local] done in {elapsed / 60:.1f} min: "
        f"{n_done:,} scored, {n_flagged:,} flagged at <{threshold}, "
        f"{n_err:,} errors. Output: {out_path}",
        flush=True,
    )
