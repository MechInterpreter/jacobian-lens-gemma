# Research log

Chronological engineering milestones for the Gemma 4 E4B Jacobian Lens
adaptation, on branch `gemma4-e4b` of
[MechInterpreter/jacobian-lens-gemma](https://github.com/MechInterpreter/jacobian-lens-gemma)
(fork of [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)).
Each entry cites the commit that landed it; dates are commit dates.

## 2026-07-01 — Upstream reference implementation

Commit [`581d398`](https://github.com/anthropics/jacobian-lens/commit/581d398613e5602a5af361e1c34d3a92ea82ba8e)
("Initial release") is Anthropic's official Jacobian Lens reference
implementation, cloned into this repository with full history preserved.
This commit is the pinned upstream starting point recorded in every
artifact's `environment.upstream_commit` field and is never modified in this
fork.

## 2026-07-14 — Gemma 4 E4B adapter, controls, configs, and tests

Commit [`b5953c7`](https://github.com/MechInterpreter/jacobian-lens-gemma/commit/b5953c7e3464d2d7aad2902e350589f54a15da9c)
added the narrow adaptation layer on top of the unmodified upstream `jlens`
package:

- `jlens/gemma4.py` — immutable Hugging Face revision resolution, gated model
  loading (`allow_model_load` flag), a `Gemma4LensModel` wrapper with
  explicit BOS handling and dual pre-softcap/softcapped readout, architecture
  verification, and a pre-fit memory/runtime probe.
- `jlens/controls.py` — negative controls: row-permuted fitted Jacobian
  (primary), scale-matched random matrix, wrong-layer application, plus
  overlap/rank metrics.
- `jlens/metadata.py` — config schema validation, config fingerprinting,
  environment manifest, and an atomic `write_metadata` helper used by every
  JSON artifact since.
- Three staged experiment configs (`configs/gemma_text_{microsmoke,smoke,pilot}.yaml`)
  and a plain-text fitting corpus separated from chat-templated evaluation
  prompts.
- `scripts/fit_gemma.py` / `scripts/apply_gemma.py` CLI entry points.
- `notebooks/gemma_4_e4b_text_jlens.ipynb`, the end-to-end pilot notebook.
- 47 new CPU-only tests (`tests/test_gemma4_adapter.py`,
  `test_controls.py`, `test_finite_difference.py`, `test_metadata.py`,
  `test_scripts.py`, `mock_gemma4.py`), bringing the suite to 79 tests total,
  all passing without network access or real model weights — including
  Jacobian orientation pinned both analytically and by finite-difference
  comparison against a perturbed activation.

## 2026-07-15 — Colab bootstrap for a fresh runtime

Commit [`348405b`](https://github.com/MechInterpreter/jacobian-lens-gemma/commit/348405bc6edac8b5ba48ef38a0247fee8c797f27)
made the notebook runnable from a clean Google Colab runtime: a bootstrap
cell clones (or fast-forward-only updates) this private repository at branch
`gemma4-e4b` into `/content/jacobian-lens-gemma`, authenticating via a
Colab-Secrets-supplied `GITHUB_TOKEN` that is never written to `.git/config`,
the remote URL, or notebook output.

## 2026-07-15 — Google Drive persistence

Commit [`ca33c54`](https://github.com/MechInterpreter/jacobian-lens-gemma/commit/ca33c545130ecccdd78dd881fcbb99a5441935cb)
added a Drive-persistence section so fitted lenses, checkpoints, and
experiment metadata survive a Colab runtime reset: it mounts
`/content/drive`, resolves a persistent project root at
`/content/drive/MyDrive/jacobian-lens-gemma`, and routes config-driven output
paths through it while leaving the git checkout and Hugging Face cache
ephemeral under `/content`. Also closed a gap where the notebook's J-lens vs
logit-lens comparison was print-only by capturing it into the saved
`eval_metadata.json`.

## 2026-07-15 — Run-scoped artifact and checkpoint paths

Commit [`d62814f`](https://github.com/MechInterpreter/jacobian-lens-gemma/commit/d62814fdf93b8eb36463c86a70ca711891d5c939)
fixed a collision risk in the Drive-persistence design: fitted lenses and
checkpoints had still been writing to a shared `artifacts/<mode>/` path,
so a later run of the same mode could silently overwrite an earlier run's
files. This commit scoped every execution's outputs beneath its own unique
`runs/<mode>_<timestamp>_<fingerprint>/` directory
(`artifacts/`, `checkpoints/`, `run_metadata.json`, `summary.md`), made
resuming a specific prior run's checkpoint an explicit, fingerprint-validated
opt-in (`JLENS_RESUME_RUN_DIR`) rather than automatic, and kept the existing
opt-in micro-smoke recovery cell unchanged. This is the exact code that
produced the smoke run below.

## 2026-07-15 — Smoke stage executed on real Gemma 4 E4B weights

Using the notebook and configuration from commit `d62814f`
(recorded as `environment.local_commit` in every artifact of this run),
`configs/gemma_text_smoke.yaml` was run to completion on a Colab GPU runtime
(NVIDIA L4): revision `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` of
`google/gemma-4-E4B-it` was resolved and loaded, the 79-test local suite
passed inside that session, architecture verification confirmed the expected
42-layer/2560-width/262144-vocab dense configuration, and the Jacobian lens
was fitted on 8 plain-text prompts at 5 source layers (7, 14, 21, 28, 35) in
361.5 seconds, producing finite `[2560, 2560]` Jacobians. Evaluation against
5 held-out prompt/position combinations and 3 negative controls plus the
logit-lens baseline followed. The complete run — fitted lens, checkpoint, and
all metadata — was copied into this repository at
[`runs/smoke_20260715T172315460316_fb2eefcd91cd/`](../runs/smoke_20260715T172315460316_fb2eefcd91cd/)
and is documented in full in [`smoke_report.md`](smoke_report.md).

## 2026-07-15 — Pilot stage executed on real Gemma 4 E4B weights

Using the notebook and configuration at commit `541b0b3`
(recorded as `environment.local_commit` in every artifact of this run),
`configs/gemma_text_pilot.yaml` was run to completion on a Colab GPU runtime
(NVIDIA L4): revision `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` of
`google/gemma-4-E4B-it` was loaded, and the Jacobian lens was fitted on
**100 WikiText-103 prompts** (128 tokens each, seed 42) at **seven source
layers (3, 7, 14, 21, 28, 35, 38)** in 9665.4 s, producing finite
`[2560, 2560]` Jacobians. Evaluation on the held-out prompt set with the
logit-lens baseline and three negative controls followed; layer 38 gave the
strongest next-token rank (12) and layers 28/35/38 strongly beat the
permuted and random controls. Run metadata is preserved under
[`runs/pilot_20260715T200437612150_311fd108c23a/`](../runs/pilot_20260715T200437612150_311fd108c23a/)
and documented in full — including an audit showing the single wrong-layer
control conflates adjacent and maximally-distant substitutions — in
[`pilot_report.md`](pilot_report.md). This 100-prompt pilot lens is the
frozen, authoritative lens artifact for all subsequent decomposition work.

## 2026-07-16 — J-space gradient-pursuit run completed and analyzed

The sparse J-space decomposition ran to completion on real Gemma 4 E4B
weights (run `jspace_20260716T170808536780_e4118850fb70`, Colab A100,
implementation commit `6442fff`): the frozen 100-prompt pilot lens
(fingerprint verified) was decomposed against at layers 14/21/28/35/38 for
k ∈ {10, 16, 25} over 76 held-out activations (38 prompts × 2 positions) —
1,140 decompositions, all 15 layer/k units complete, plus trajectories,
candidate-ignition diagnostics, recurring-signature tables, and the named
control-suite evaluation.

On branch `jspace-pursuit-analysis` this run was turned into a rigorous
offline analysis (no model download, no GPU, source artifacts byte-frozen):
deterministic analysis pipeline (`jlens/jspace_analysis.py`,
`jlens/similarity.py`, `scripts/analyze_jspace.py`; derived tables under
`reports/`), full integrity verification (zero issues), and
[`jspace_run_report.md`](jspace_run_report.md) with
[`jspace_similarity_analysis.md`](jspace_similarity_analysis.md) as the
recurrence methodology. Headlines: k=10 suffices (k25 adds 4–10 % relative
explained fraction as tail mass; supports are ~95–99 % nested); layer 21's
explained-fraction collapse (2.3e-5) traces to an outlier fitted Jacobian
(‖J₂₁‖_F = 188.7 vs 3.5–18.1 at all other layers) and stays unresolved
between fitting pathology and genuine mid-stack transition; the plain/chat
gap decomposes into a position-−1 tokenization artifact, mid-stack
template dominance, and a modest residual format shift; exact cone
signatures never repeat, but similarity-based recurrence finds
threshold-robust repeated structure (chat-template cones mid-stack;
antonym/copula completion-frame cones late); the 35→38 sparse-coordinate
stabilization survives every cut (k, format, position, category, frequency
adjustment, output-token removal) and is reported as late-layer
consolidation — explicitly NOT validated ignition. The L4 OOM in
whole-dictionary `isfinite` was fixed with chunked validation/norms and an
optional chunked dictionary build (bit-identical results, tested), and
`jlens.metadata.execution_record` now separates configured vs resolved
`allow_model_load` so the completed run's provenance ambiguity cannot
recur.

## Next planned milestone

Layer-21 refit diagnostic (per-prompt Jacobian variance at the outlier
layer, optional chat-formatted fitting subset), then a lower-memory
decomposition rerun on {21, 31, 35, 38} at k=10 — now feasible without an
A100 after the chunked-memory changes.


# 2026-07-17 — Gemma 4 Multimodal J-Lens Explorer (branch `multimodal-jlens-explorer`)

The completed text research was packaged into a resume-ready product: the
**Gemma 4 Multimodal J-Lens Explorer**, a static Vite/React/TypeScript
application (`explorer/`) that visualizes J-lens predictions, k=10
gradient-pursuit cones, step-replayable pursuit traces, and cross-layer
trajectories from the completed jspace run — no Python, no model, no backend
at browse time. Data flows through one versioned schema
(`schemas/explorer_bundle.schema.json`, `jlens.explorer.bundle.v1`) fed by a
deterministic exporter (`jlens/explorer_export.py`,
`scripts/export_explorer_bundle.py`; byte-identical re-exports, absolute
paths stripped, sources verified read-only). A committed 20-example demo
bundle covers all categories and both formats, deliberately including weak
examples; per-step pursuit coefficients and per-layer J-lens top-k lists
were not persisted by the completed run and are exported/rendered as
explicitly unavailable rather than reconstructed.

Two GPU notebooks were prepared (not executed — no model download, no GPU in
this pass), both consuming the frozen fingerprint-verified pilot lens:

- `notebooks/gemma_4_e4b_jspace_causal_smoke.ipynb` +
  `jlens/interventions.py`: measured residual-stream interventions
  (`h' = h + m·delta`) at the exact block_output sites, layers 35/38,
  multipliers −1/0/+1, three targeted families from the recorded k=10 cones
  plus exactly norm-matched deterministic random controls (~120 conditions,
  ≈30–45 min on an L4). Baseline-parity gate (unhooked vs multiplier-0 vs
  identical-copy writeback) aborts before any intervention on drift;
  deterministic condition IDs, per-condition JSONL checkpointing,
  append-safe resume, completed-run refusal. The 4-example manifest
  (`configs/causal_smoke_examples.json`) pins strong/semantic/chat/weak cases
  chosen from measured records, reasons recorded.
- `notebooks/gemma_4_e4b_multimodal_jlens_capture.ipynb`: first
  image-/audio-conditioned records (layer 38, k=10, position −1) with
  processor-interface inspection, clear failure on unsupported modalities,
  and explorer-ready bundles — recorded throughout as an exploratory
  application of the text-fitted lens, with no modality-invariance or
  pixel/audio-span claims.

Causal and multimodal UI states ship with deterministic, loudly-badged
synthetic fixtures (`scripts/make_ui_fixtures.py`); the frontend
auto-prefers measured bundles dropped under `explorer/public/data/measured/`.
Added ~70 Python tests (exporter determinism/immutability/merging,
intervention hook semantics and cleanup, config validation, notebook light
paths) and 24 frontend tests (Vitest + RTL); full CPU suite green with no
network access. Layer 21 remains documented history — visible in the
explorer as data, not a workstream.

## 2026-07-29 — Generative J-cone steering validation scaffold

Commits [`7c17547`](https://github.com/MechInterpreter/jacobian-lens-gemma/commit/7c17547),
[`b4e110e`](https://github.com/MechInterpreter/jacobian-lens-gemma/commit/b4e110e),
and [`347cd14`](https://github.com/MechInterpreter/jacobian-lens-gemma/commit/347cd14)
(branch `experiment/generative-jlens-validation`) built the go/no-go
experiment asking whether weighted reconstructions `q_C = Σ a_i v_i` of
active J-space generators, injected into neutral verbalization prompts at
the established `block_output` site, make Gemma's native decoder produce
multi-token concepts more specifically than matched controls:

- `jlens/generative.py` — steering schedules (prompt-only / constant /
  exponentially decaying reinjection), a schedule-weighted multi-position
  injection hook (float32 math, exact zero-delta parity, removal on every
  exit path), manual **uncached** greedy decoding, teacher-forced
  multi-token target scoring, the 13-condition vector battery
  (zero / full cone / coefficient-mass subcones at 60–80% / manual
  subcones / unrelated cone / random matched-norm / shuffled /
  sign-reversed / wrong-layer / wrong-position / raw activation /
  activation-diff), norm-relative strength scaling, and the aggregation +
  go/no-go verdict machinery. Unimplemented paths raise `GenerativeError`.
- `configs/generative_benchmark.json` — 16 multi-token concepts (split
  words, compounds, named entities, noun phrases) in dev/held-out splits
  with matched control prompts.
- `scripts/run_generative_validation.py` — gated runner (architecture +
  lens-fingerprint + zero-parity + manual-vs-`generate()` greedy
  equivalence gates), fresh per-example k=10 pursuit per steering layer
  (14/21/28), fsynced JSONL records, per-condition summaries, dev-split
  calibration, and the go/no-go report.
- `docs/generative_validation.md` — protocol, metrics, and the **verified**
  Google Colab CLI workflow (v0.6.0 syntax read from the installed package;
  native-Windows `termios` limitation documented with the notebook
  fallback).

## 2026-07-29 — Generative receiver prompting was confounded; fixed

The first generative smoke decodes ("Internal Concept") are **void**. Two
receiver-side defects compounded: the prompts themselves contained "internal
concept" / "internal representation" / "Label:", so a restatement of the prompt
was the model's likeliest continuation regardless of any injected vector; and
receiver prompts were tokenized raw through `model.encode()` even though the
pinned checkpoint (`google/gemma-4-E4B-it`) is instruction-tuned, so the model
never saw the chat turn structure and generation prefix it was tuned to answer
after.

Fixed on `experiment/generative-jlens-validation`: explicit receiver formats
(`chat` by default, `legacy_raw` for reproducing the confounded runs) behind one
centralized `encode_receiver_prompt` helper with structural verification of the
chat rendering; three priming-free default prompts (the old ones retained only
as `legacy-*` diagnostics); target ids derived contextually as the assistant
continuation of the exact formatted prompt, with chat-control tokens excluded and
the 2–6 token rule applied to those ids; the steering anchor revalidated against
the formatted prompt length and recorded with its token id and string; a
per-run `artifacts/prompt_debug.json`; and two added greedy gates (the reference
sequence must start with the exact prompt ids, and recorded text must decode the
generated ids alone). Source-prompt formatting is unchanged, so the pursuit/lens
side is unaffected. **No corrected GPU run has been performed** — no claim about
whether the steering works can be made from this branch yet.

## Next planned milestone

User-run L4 sessions: the causal smoke run, then the multimodal capture;
merge both bundles into the explorer, capture screenshots, and switch
resume packaging from the pre-completion bullet to the full bullet
(docs/resume_packaging.md gates this on the merged measured bundles).
After those, the generative validation GPU run — first
`scripts/validate_benchmark_targets.py` (contextual target counts under the real
tokenizer have not been observed yet), then the corrected two-concept smoke run,
then dev calibration, then the frozen held-out evaluation — per
docs/generative_validation.md.
