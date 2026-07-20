# Resume packaging

## Bullet A — use only after the causal and multimodal real runs complete

> Built a browser-based multimodal interpretability explorer for Gemma 4 E4B
> that visualizes Jacobian-lens predictions, sparse gradient-pursuit cones,
> cross-layer trajectories, and measured residual-stream interventions for
> text, image-conditioned, and audio-conditioned model states.

**Gate:** every claim in this sentence must be backed by a merged measured
bundle — the causal smoke run (`explorer_causal_bundle.json`) **and** the
multimodal capture (`multimodal_explorer_bundle.json`, image + audio both
captured) must have completed and be loaded by the explorer. Until then, use
Bullet B.

## Bullet B — truthful today (pre-completion)

> Built a browser-based interpretability explorer for Gemma 4 E4B that
> visualizes Jacobian-lens predictions, sparse gradient-pursuit cones, and
> cross-layer trajectories from completed text experiments, with a
> schema-validated pipeline and parity-gated intervention notebooks prepared
> for measured causal steering and image/audio-conditioned capture.

## Supporting facts (for interviews)

- Fitted Jacobian lens on 100 WikiText prompts at 7 layers of
  `google/gemma-4-E4B-it` (pinned revision, fingerprinted artifacts);
  evaluated against logit-lens and permuted/random/layer-mapped controls.
- 1,140 sparse nonnegative gradient-pursuit decompositions (k ∈ {10,16,25})
  over a 262k-atom dictionary (`W_U J_l`), with deterministic cone signatures
  and cross-layer trajectory records.
- Deterministic exporter → versioned JSON Schema (`jlens.explorer.bundle.v1`)
  → static Vite/React/TS app; byte-identical re-exports enforced by tests.
- Intervention backend with exact block-output hook editing, baseline-parity
  gates, deterministic condition IDs, exactly norm-matched random controls,
  and append-safe resumption; ~290 Python + frontend tests, CPU-only, no
  model download.
- Honest-by-construction UI: measured / synthetic-fixture / no-data states
  are first-class, and unpersisted fields render as "not recorded".

## Rules

- Never present fixture numbers or screenshots as results.
- Keep "measured" language tied to the record status fields; the explorer's
  badges are the source of truth for what may be claimed.
