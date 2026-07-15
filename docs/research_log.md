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

## Next planned milestone

Sparse J-space decomposition of held-out activations by nonnegative
gradient pursuit against the frozen pilot lens (branch
`jspace-gradient-pursuit`): pilot analysis is documented in
[`pilot_report.md`](pilot_report.md), evaluation controls are being
disambiguated and extended, and the decomposition + cone-signature +
candidate-ignition tooling is described in
[`jspace_decomposition.md`](jspace_decomposition.md).
