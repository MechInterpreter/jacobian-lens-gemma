# Notebooks

Eight notebooks at this level are the **active research workflow**. Everything
else has been moved into `archive/`, where it is preserved rather than
preserved-and-confusing: an archived notebook produced a completed result, and
re-running it today would re-run an *older protocol*.

> **Read this before running anything in `archive/`.** Those notebooks are
> historical records. Several of them ask under the candidate-listed prompt
> protocol (`gemma-it-chat-balanced-options-v1`), which the current studies
> replaced with the open protocol `mmpilot.open_entity_identification.v1`.
> Numbers produced under the two are **different measurements** and must never
> be pooled, differenced or ranked against each other. Some also predate the
> image-unique selection profile, the capability-admissibility rule, and the
> scale-250 layer-32 confirmation. Run one only to reproduce the result it
> already reported, never to produce a current one.

---

## Active workflow

Run in this order. Each stage consumes the artifacts the one above it
published, and none of them fits a lens except the first two.

| # | notebook | what it is | produces |
|---|---|---|---|
| 1 | `research_grade_multilayer_jlens_calibration_colab.ipynb` | The research-grade multi-layer calibration. **Canonical** source of published text-only lenses at scale 100. | validated lenses at L35 / L38 / L40 |
| 2 | `research_grade_early_layer_jlens_extension_colab.ipynb` | The early-layer extension. Confirmed **physical layer 32 at scale 250** on its own untouched 256-prompt confirmation set (`EARLY_LAYER_CALIBRATION_GO`); L26 failed. | the published L32 artifact this fork's current work rests on |
| 3 | `multimodal_jspace_spokencoco_native_audio_colab.ipynb` | The completed three-modality transfer study at L35 (text / image / native spoken audio). **Canonical** for the L35 causal result. | the L35 transfer run |
| 4 | `multimodal_jspace_spokencoco_l32_followup_colab.ipynb` | The L32 open-prompt follow-up: integrity, representational transfer, causal transfer, native convergence, and a paired L35 reference under the same open prompt. **Canonical** for the open-prompt L32 result. | `mml32_*` run |
| 5 | `multimodal_jspace_l32_convergence_resolution_colab.ipynb` | **New.** The independent L32 convergence-resolution study: a fresh, verified-disjoint SpokenCOCO population scored against the *already frozen* criterion, with an optional conditional Stage-B causal replication. | `mml32res_*` run |
| 6 | `multimodal_jspace_coordinate_swap_mock_colab.ipynb` | MOCK-only. The Anthropic two-coordinate swap: its algebra, refusals, controls and involution on a synthetic world. Use notebook 8 for the real study. | nothing scientific; a protocol test |
| 7 | `research_grade_l27_l31_preconvergence_study_colab.ipynb` | **New.** The bounded L27–L31 transition study: the one interval left open by L26 (failed confirmation), L32 (confirmed but AMBIGUOUS twice) and L35 (CONVERGED). Fits all five candidates fresh at scale 250, confirms them on a genuinely untouched set, selects the *earliest* passer, and conditionally measures native convergence and cross-modal causal transfer on a fourth independent SpokenCOCO population. | `mmpre_*` run |
| 8 | `multimodal_jspace_anthropic_reasoning_swap_colab.ipynb` | **Primary paper-method test (v2).** Uses Anthropic's published two-coordinate exchange once at each independently confirmed physical layer and compares hidden animal-coordinate swaps directly with matched leg-answer swaps in text, image and spoken audio. It screens a fresh synchronized population first, uses alpha=1 as primary, reports alpha=2 separately, and keeps the final-token edit descriptive rather than blocking. | `mmpaper2_*` run and `paper_reasoning_swap_v2_report.json` |

Notebooks 1–5, 7 and 8 are all **switched off** in the committed file. Opening one
starts nothing, downloads nothing and spends nothing.

### Notebook 7 is a *transition* study, and it is bounded on purpose

It asks one question — is there a layer in 27–31 with a confirmed J-lens, a
`NOT_CONVERGED` native readout in all three modalities, and controlled
cross-modal causal transfer, all on one population — and the candidate interval
is **closed**. There is no clause that widens it to L33/L34 after a result, no
replacement layer, and no second scale: 250 is used because that is where L32
was confirmed.

Two things it deliberately does *not* reuse. The extension accumulator holds
source layers `[8, 14, 20, 26, 32, 35, 38, 40]`, so none of 27–31 is in it and
new Jacobian accumulation is required; `assert_new_source_layers` refuses a
seeded checkpoint rather than continuing one. And the extension's confirmation
set has been opened — that is *how* L26's failure and L32's pass are known — so
a new untouched set is drawn, or the study is **blocked**.

It runs in three sessions: a free-CPU Stage 0 (`PREPROCESSING_ONLY = True`), an
L4 session for Stages 1–3, and a conditional Stage 4. Every stage is
checkpointed; an interruption loses at most one bounded batch or unit.
`docs/l27_l31_preconvergence_study.md` has the full contract.

### Notebook 5 runs in two sessions, and the first one wants a CPU

The convergence-resolution study has to establish that its population is
disjoint from every photograph, recording and caption the completed runs spent.
That is Drive I/O over thousands of small files, and a GPU makes none of it
faster. Section 8 is therefore a checkpointed preparation that persists to Drive
and can be stopped at any moment:

* **session 1, free CPU** — `PREPROCESSING_ONLY = True`. Section 8 harvests and
  checkpoints; stopping repeats at most one bounded unit of ≤25 files and never
  restarts from zero. No run directory is created and no model is loaded.
* **session 2, L4** — `PREPROCESSING_ONLY = False` with the same scientific
  configuration. Section 8 loads and verifies the cache without reading a single
  source unit, and Stage A starts.

Section 11 refuses to load Gemma while the preparation is incomplete.
`docs/l32_resolution_preprocessing.md` has the full contract: the preparation
fingerprint, the minimal-source completeness proof, the shard/checkpoint
semantics and what a fresh process reconstructs versus re-proves.

### Which notebook is canonical for which claim

| claim | canonical notebook |
|---|---|
| published text-only lenses (L35/38/40, scale 100) | (1) calibration |
| published L32 lens (scale 250) | (2) early-layer extension |
| L35 three-modality causal transfer | (3) native audio transfer |
| L32 causal transfer under an **open** prompt, and the paired L35 reference | (4) L32 follow-up |
| L32 native direct-readout convergence on an **independent** population | (5) convergence resolution |
| anything about physical layers 27–31 | (7) the L27–L31 transition study |
| paper-style hidden-intermediate exchange versus the answer-swap confound | (8) Anthropic reasoning swap |

---

## Archive

### `archive/completed_studies/`

Studies that produced a scientific conclusion still cited in the docs. Their
run directories on Drive are the evidence; the notebook is how it was produced.

| notebook | what it concluded |
|---|---|
| `multimodal_jspace_spokencoco_pilot_colab.ipynb` | The four-concept SpokenCOCO pilot: the first cross-modal J-space transfer result, and the run whose image-level pseudoreplication motivated the image-unique selection profile. See `docs/multimodal_jspace_pilot.md`. |
| `multimodal_jspace_spokencoco_robustness_colab.ipynb` | The bounded six-concept robustness study: one synchronized group per photograph, eight distinct images per design cell. |
| `multimodal_jspace_layer_localization_colab.ipynb` | Layer localization with the tie-aware Stage-B gate and the frozen target policy. |
| `multimodal_jspace_output_convergence_audit_colab.ipynb` | The convergence-timing audit of the completed L35 transfer — the model's own output head applied to stored L35/38/40 residuals, read-only against the finished run. **This is where the frozen convergence criterion comes from.** See `docs/output_convergence_timing.md`. |
| `gemma4_native_spoken_audio_feasibility_colab.ipynb` | That Gemma 4 E4B accepts native spoken audio through the content-block path at all, and that component presence is not support. See `docs/native_spoken_audio.md`. |
| `gemma_4_e4b_layer32_confirmatory_validation_colab.ipynb` | The first layer-32 confirmatory validation. **Superseded** by (2), which confirmed L32 at scale 250; this notebook's scale-100 result is the failure that motivated the extension. |

### `archive/engineering_audits/`

Audits of the machinery rather than of the model.

| notebook | what it established |
|---|---|
| `mmpilot_image_independence_audit_colab.ipynb` | That repeated caption groups of one photograph had inflated the pilot's n, and the image-level correction for evidence already collected. |
| `gemma_4_e4b_layer21_diagnostic.ipynb` | The layer-21 refit diagnostic; a completed investigation kept as documented history. |

### `archive/legacy_prototypes/`

The notebooks the current pipeline grew out of. Useful for reading, not for
running: their protocols predate almost every rule the current studies enforce.

| notebook | what it was |
|---|---|
| `gemma_4_e4b_text_jlens.ipynb` | The original end-to-end fitting notebook; produced the smoke and pilot runs. |
| `gemma_4_e4b_jspace_pursuit.ipynb` | The decomposition notebook: verifies and consumes the frozen pilot lens, never refits. |
| `gemma_4_e4b_jspace_causal_smoke.ipynb` | The first parity-gated, checkpointed causal-steering smoke run. See `docs/causal_smoke_run.md`. |
| `gemma_4_e4b_multimodal_jlens_capture.ipynb` | The first image/audio-conditioned captures on the frozen lens. See `docs/multimodal_capture.md`. |
| `gemma_4_e4b_text_jlens_recalibration_colab.ipynb` | The bounded text-only recalibration runner named by `jlens.mmlocalize.lens_validity.RECALIBRATION_PLAN`. It was predeclared as a contingency and the contingency did not arise; it is kept because the plan still points at it. |

### `archive/protocol_mocks/`

MOCK-only notebooks whose subject is a *protocol*, not a result.

| notebook | what it demonstrates |
|---|---|
| `open_prompt_protocol_mock_colab.ipynb` | The legacy prompt rebuilt byte-for-byte, open prompts naming no candidate in any channel, the leakage refusals, and the external candidate-scoring boundary. See `docs/prompt_protocol.md`. |

---

## Why nothing was deleted

Three notebooks were considered for deletion —
`gemma_4_e4b_layer21_diagnostic.ipynb`,
`gemma_4_e4b_layer32_confirmatory_validation_colab.ipynb` and
`gemma_4_e4b_text_jlens_recalibration_colab.ipynb`. All three are archived
instead, because deleting them would have erased reproducibility context that
is still live in the repository:

* each has a dedicated test asserting its structure
  (`tests/test_layer21_diagnostic_setup.py`,
  `tests/test_layer32_confirmatory_notebook.py`,
  `tests/test_text_jlens_recalibration_notebook.py`);
* `jlens/mmlocalize/lens_validity.py` names the recalibration notebook as the
  `runner` of a predeclared plan, so deleting the file would leave a protocol
  pointing at nothing;
* the layer-32 confirmatory notebook is the **scale-100 failure** that the
  early-layer extension was built to answer, and reading the extension's
  `EARLY_LAYER_CALIBRATION_GO` without it loses why scale is part of
  "confirmed".

Git history alone was not judged sufficient: a path named in a live code
constant has to resolve.

## Regenerating a notebook

Every notebook here is generated from a builder in `scripts/`, so the committed
file stays output-free and byte-reproducible. Edit the builder, then re-run it:

```bash
python scripts/_build_l32_resolution_notebook.py
```

The builders write to the archive paths where their notebook was archived, so
re-running one never recreates a duplicate at the old top-level path. A test
enforces that (`tests/test_notebook_layout.py`).
