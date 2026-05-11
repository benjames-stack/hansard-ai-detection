"""Fast-DetectGPT (Bao et al., 2024) on Modal — Phase A second-opinion detector.

Independent second open-source detector to triangulate against Binoculars.
Where Binoculars uses the ratio of perplexity under two models (base + instruct)
to score AI-likeness, Fast-DetectGPT uses **probability curvature under a single
model** — text where the model is highly confident in the actual tokens but
less confident in alternatives is flagged as AI-like.

Run:

    modal run scripts/research_ai_fast_detect_gpt_modal.py \\
        --input-file /tmp/speeches.jsonl \\
        --output-file /tmp/scores_fdgpt.jsonl

Same input format as the Binoculars wrapper ({id, text} JSONL); output is
{id, score, is_ai} JSONL. The two scores are NOT directly comparable —
Binoculars' threshold is 0.901 (lower → more AI-like); Fast-DetectGPT's
threshold is paper-default ~0.0 (higher → more AI-like). The cross-detector
agreement set is what we look for in the report.

Setup mirrors the Binoculars wrapper (no separate `pip install` needed by
the user — Modal builds the container image on first run, cached after).

Single-model approach: cheaper than Binoculars (one forward pass instead of
two), so cost is roughly half. Fits on smaller GPUs — `T4` (16GB) works
for the default `EleutherAI/gpt-neo-2.7B` scoring model. Default container
is `A10G` (24GB) for headroom.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

# Default threshold from Fast-DetectGPT paper §5.3 — speeches with score
# above this are flagged as AI-like. Calibrated on essays / news articles;
# real-world threshold on parliamentary register may need adjustment after
# manual spot-check (same audit step as for Binoculars).
DEFAULT_THRESHOLD = 0.0
MAX_INPUT_WORDS = 600   # Match the Binoculars wrapper for consistency.

fast_detect_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.40",
        "accelerate>=0.30",
        "sentencepiece",
    )
    # No git clone needed — we inline the scoring formula below to avoid
    # the upstream repo's transitive `matplotlib` / `datasets` deps in
    # its eval modules.
)

# Reuse the same HuggingFace cache Volume as the Binoculars wrapper —
# both jobs benefit from a shared model-weights cache.
hf_cache = modal.Volume.from_name(
    "binoculars-hf-cache", create_if_missing=True
)

app = modal.App("checkhansard-fast-detect-gpt")


@app.cls(
    image=fast_detect_image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=4 * 3600,
    scaledown_window=300,
)
class FastDetectGPTScorer:
    """Single-container Fast-DetectGPT scorer.

    Loads `EleutherAI/gpt-neo-2.7B` on enter (cached in the Volume after
    first run; ~5GB download). Scores each speech by computing the
    Fast-DetectGPT sampling-discrepancy criterion on the model's logits.
    """

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Inline the Fast-DetectGPT analytic sampling-discrepancy formula
        # (Bao et al., 2024 §3.3). This is pure tensor math — no need to
        # import the upstream repo, whose eval-side modules (metrics.py,
        # data_builder.py) drag in matplotlib/datasets/scikit-learn.
        def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
            """Analytic per-sequence Fast-DetectGPT criterion.

            logits_ref / logits_score: [B, T-1, V] — token-level logits
                from the reference and scoring models, aligned to labels.
            labels: [B, T-1] — the actual next-token IDs.
            Returns: a Python float (mean over batch).
            """
            lprobs_score = torch.log_softmax(logits_score, dim=-1)
            probs_ref = torch.softmax(logits_ref, dim=-1)
            log_likelihood = lprobs_score.gather(
                dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)              # [B, T-1]
            mean_ref = (probs_ref * lprobs_score).sum(dim=-1)         # [B, T-1]
            var_ref = (
                (probs_ref * lprobs_score.square()).sum(dim=-1)
                - mean_ref.square()
            )                                                          # [B, T-1]
            seq_var = var_ref.sum(dim=-1).clamp(min=1e-9).sqrt()       # [B]
            discrepancy = (
                (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / seq_var
            )                                                          # [B]
            return float(discrepancy.mean().item())

        self.discrepancy_fn = get_sampling_discrepancy_analytic

        # Default to gpt-neo-2.7B (paper's recommended scoring model).
        # ~5GB; cached in the Modal Volume after first download.
        model_name = "EleutherAI/gpt-neo-2.7B"
        print(f"[fdgpt] loading {model_name}…", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda"
        )
        self.model.eval()
        self.device = "cuda"
        print(f"[fdgpt] model loaded in {time.time() - t0:.1f}s", flush=True)

    @modal.method()
    def score_batch(
        self,
        speeches: list[dict],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> list[dict]:
        import torch

        out: list[dict] = []
        for s in speeches:
            sid = s.get("id")
            text = s.get("text") or ""
            parts = text.split()
            if len(parts) > MAX_INPUT_WORDS:
                text = " ".join(parts[:MAX_INPUT_WORDS])
            if not text.strip():
                out.append({"id": sid, "error": "empty"})
                continue
            try:
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                input_ids = tokenized.input_ids
                if input_ids.shape[1] < 5:
                    out.append({"id": sid, "error": "too_short"})
                    continue

                # Fast-DetectGPT criterion: compute logits, then the
                # analytic sampling discrepancy between the model's
                # distribution and the actual tokens. Self-discrepancy
                # variant: same model for reference and scoring (cheaper
                # than the dual-model setup; still effective per paper §5).
                with torch.no_grad():
                    logits = self.model(input_ids).logits[:, :-1]
                    labels = input_ids[:, 1:]
                    score = self.discrepancy_fn(logits, logits, labels)

                out.append(
                    {
                        "id": sid,
                        "score": score,
                        "is_ai": score > threshold,
                    }
                )
            except Exception as exc:
                out.append({"id": sid, "error": str(exc)})
        return out


@app.local_entrypoint()
def main(
    input_file: str,
    output_file: str,
    batch_size: int = 100,
    threshold: float = DEFAULT_THRESHOLD,
    limit: int = 0,
):
    """Drive the Fast-DetectGPT scoring run end-to-end.

    Same arguments as the Binoculars wrapper; pass `--limit 1000` for a
    smoke test before committing to the full corpus.
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
    print(
        f"[fdgpt local] {len(speeches):,} speeches to score "
        f"(batch={batch_size}, threshold={threshold})",
        flush=True,
    )
    if not speeches:
        return

    batches = [
        speeches[i : i + batch_size]
        for i in range(0, len(speeches), batch_size)
    ]
    scorer = FastDetectGPTScorer()
    n_done = 0
    n_flagged = 0
    n_err = 0
    t0 = time.time()
    with out_path.open("w") as out:
        for i, results in enumerate(
            scorer.score_batch.map(batches, kwargs={"threshold": threshold})
        ):
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
                f"[fdgpt local] batch {i + 1}/{len(batches)}: "
                f"{n_done:,}/{len(speeches):,} ({rate:.1f}/s, "
                f"eta {eta / 60:.0f} min, {n_flagged} flagged, "
                f"{n_err} errors)",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"\n[fdgpt local] done in {elapsed / 60:.1f} min: "
        f"{n_done:,} scored, {n_flagged:,} flagged at score > {threshold}, "
        f"{n_err:,} errors. Output: {out_path}",
        flush=True,
    )
