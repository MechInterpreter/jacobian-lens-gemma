# Gemma 4 Multimodal J-Lens Explorer — MVP specification

Status: implementation guide for the `multimodal-jlens-explorer` branch.
Product name (use consistently): **Gemma 4 Multimodal J-Lens Explorer**.

## Product scope

A browser-based, static-file interpretability explorer for
`google/gemma-4-E4B-it` (revision `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd`,
42 decoder layers, d_model 2560, vocab 262144) that visualizes:

1. J-lens vs. model predictions per layer/position (rank + top-k overlap);
2. sparse k=10 gradient-pursuit cones on the frozen pilot lens
   (`sha256:7229c756…c96f474`);
3. step-by-step gradient-pursuit playback from recorded residual histories;
4. cross-layer cone trajectories (layers 14, 21, 28, 35, 38 available;
   14/28/35/38 are the primary UI layers — 21 is retained as documented
   historical data, not a workstream);
5. measured causal steering/ablation records at layers 35/38 (once the user
   runs the causal smoke notebook);
6. image- and audio-conditioned records (once the user runs the multimodal
   capture notebook).

The explorer reads **static exported JSON bundles** only. Browsing completed
experiments never loads Gemma and never runs Python.

## User stories

- As a reviewer, I pick a saved example, click a token position, pick a
  layer, and see how well the J-lens approximates the model's next-token
  behaviour there (rank of model top-1, top-k overlap, explained fraction).
- As a researcher, I inspect the k=10 cone: which vocabulary atoms were
  selected, with what coefficients, whether the model's output token is one
  of them, and how much of the activation the cone actually explains.
- As a researcher, I replay the pursuit: which atom entered at each step and
  how the residual norm fell; I can see exactly which per-step quantities
  were recorded and which were not.
- As a researcher, I compare the same activation's cone across layers:
  retained/entered/exited atoms, Jaccard and weighted similarity.
- As a reviewer, I open the steering panel and compare a measured targeted
  intervention against its norm-matched random control at the multipliers
  that were actually run.
- As a portfolio reader, I switch modality to Image or Audio and inspect the
  first exploratory multimodal-conditioned records, clearly labelled as an
  application of the *text-fitted* lens to multimodal-conditioned decoder
  states.

## Supported record types

| Bundle section | Source of truth |
|---|---|
| `examples[]` | `run_metadata.json` `capture_meta` + `configs/prompts/eval_prompts_v2.json` |
| `layer_records[]` | `artifacts/eval_v2_results.json` (jlens rank/overlap) + `artifacts/cones/cones_layer{L}_k10.json` (reconstruction metrics) |
| `cones[]` | `artifacts/cones/cones_layer{L}_k10.json` (`jlens.cones.record.v1`) |
| `pursuit_traces[]` | derived from cone `stopping.residual_norm_history` + selection-ordered `selected_token_ids` |
| `trajectories[]` | `artifacts/trajectories_k10.json` (`jlens.cones.transition.v1`) |
| `causal_records[]` | causal smoke run `artifacts/explorer_causal_bundle.json` (merged post-run) |
| multimodal `examples[]`/`cones[]` | multimodal capture run `artifacts/multimodal_explorer_bundle.json` (merged post-run) |

## Supported modalities

`text` (measured now), `image_text` and `audio_text` (schema + UI ready;
records arrive after the user's capture run). Plain `image`/`audio` without a
text prompt are schema-valid but not produced by the planned notebooks.

## Measured data vs. synthetic fixtures — exact distinction

Every bundle carries `provenance.data_status` ∈
`{"measured", "imported", "synthetic_fixture"}` and every causal record
carries its own `status` field. The UI renders a persistent badge:

- **Measured intervention** — record produced by a real forward pass with a
  real hook edit, carried through `intervention_records.jsonl` provenance.
- **Synthetic UI fixture** — hand-written numbers used only to exercise UI
  states; stored only under `explorer/public/data/fixtures/`; never shown
  without the fixture badge; never used in screenshots presented as results.
- **No causal data available** — the truthful empty state.

The frontend auto-prefers measured bundles: if a measured causal/multimodal
bundle is present in `explorer/public/data/`, fixtures are not offered for
that section by default.

## Known data limitations (exposed, not papered over)

- Per-step pursuit **coefficients are not recorded** in the completed run;
  playback shows step-indexed atom entry and residual-norm decay (both
  recorded), and labels coefficient state "final only".
- Per-layer **J-lens top-k token lists were not persisted** by the completed
  text run; the prediction panel shows rank-of-model-top-1 and top-k overlap
  (recorded) and the model's measured top-1, and labels full lists
  unavailable for those records. The capture notebooks record full top-10
  lists for new runs.
- Explained fractions at k=10 are small (documented in the J-space run
  report); the cone panel shows the unexplained residual prominently.
- The lens was fitted on text; multimodal records are an exploratory
  application, and the UI says so.
- No pixel- or audio-span-level attribution is available or implied;
  input viewers show modality token ranges only when the processor exposes
  them.

## MVP acceptance criteria

1. `npm run build` produces a static site; opening it serves the committed
   text demo bundle with ≥12 real examples across categories.
2. Example → position → layer → predictions → cone → pursuit playback →
   trajectory → provenance flow works with keyboard and mouse.
3. Causal panel renders the measured-record schema and its empty/fixture
   states correctly; only measured multipliers are selectable.
4. Modality switcher renders image/audio input viewers from a bundle
   (fixture now, real after capture run) with correct badges.
5. Exporter is deterministic (byte-identical on re-run) and validates
   against `schemas/explorer_bundle.schema.json`.
6. Python and frontend test suites pass on CPU with no model download.

## Deliberately excluded from the first release

- Live model inference or Python backend of any kind;
- SAE training, lens refitting, pursuit reruns, layer-21 investigation;
- geometric/3D visualization of the 2560-dim space;
- interpolated causal effects between measured multipliers;
- pixel/audio-span attribution;
- k=16/k=25 cones in the UI (schema allows them; MVP ships k=10);
- cross-prompt and nuisance-direction causal controls (schema-ready only);
- deployment automation (site is static; hosting is a user step).
