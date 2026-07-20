# Paper outline (working title)

**Exploring a Text-Fitted Jacobian Lens Across Modalities: Sparse J-Space
Cones, Cross-Layer Trajectories, and Measured Steering in Gemma 4 E4B**

Status: outline only. Sections 5–7 depend on the causal smoke run and the
multimodal capture completing; nothing below may claim results that the
corresponding measured bundles do not contain.

## 1. Introduction

- The J-lens readout (`W_U norm(J_l h)`) as a verbalization probe; why
  Gemma 4 E4B (open weights, multimodal decoder, softcapped readout) is a
  useful adaptation target.
- Contribution list: (a) fitted lens + control-suite evaluation on Gemma 4
  E4B; (b) sparse nonnegative gradient-pursuit decomposition over the 262k
  J-lens dictionary with deterministic cone bookkeeping; (c) cross-layer
  trajectory analysis incl. the 35→38 consolidation and the layer-21 outlier;
  (d) parity-gated measured interventions on cone components vs norm-matched
  controls; (e) exploratory application of the text-fitted lens to image- and
  audio-conditioned decoder states; (f) an open static explorer that renders
  every record with measured/fixture provenance.

## 2. Methods

- Lens fitting on Gemma 4 E4B (sites, estimator, pinned revision, layer
  scalars, dual pre-softcap/softcapped readout).
- Gradient pursuit (nonnegative variant, selection/refinement, recorded
  residual histories); cone records and signatures as bookkeeping.
- Evaluation controls (permuted, scale-matched random, layer-mapped).
- Intervention semantics: `h' = h + m·delta` at block_output; delta families;
  exactly norm-matched isotropic controls; baseline-parity gate.
- Multimodal capture protocol: processor-interface inspection, modality token
  ranges, single-position capture at layer 38.

## 3. The fitted lens on text (completed)

- Rank/overlap vs logit lens and controls by layer/format/category.
- The plain/chat gap decomposition; interpretation limits.

## 4. Sparse J-space structure (completed)

- k sufficiency (10 vs 16 vs 25), nested supports, explained-fraction
  smallness as a headline honesty point.
- Recurring similarity-based structure; frequency/nuisance atoms.
- Trajectories: 35→38 consolidation robustness; layer-21 outlier Jacobian as
  a cautionary case for lens-fit pathology.

## 5. Measured steering (pending the causal smoke run)

- Targeted (output atom / top non-output atom / full cone) vs norm-matched
  random control at m ∈ {−1, 0, +1}, layers 35/38, four examples.
- Report exactly: target logit/rank/prob deltas, KL, top-10 overlap,
  completion changes; strong vs weak example contrast.
- Explicit scope: a smoke study; no generalization claims.

## 6. Exploratory multimodal application (pending the capture run)

- What the text-fitted lens reads out of image-/audio-conditioned states at
  layer 38; cone composition vs text cones for matched prompt frames.
- Framed strictly as a probe of the lens, not modality-invariant concepts.

## 7. The explorer as a research artifact

- Schema-first pipeline, determinism/provenance guarantees, honest-gap
  rendering; screenshots.

## 8. Limitations

- Linear approximation quality (small explained fractions); text-only fit;
  four-example causal scope; one-example-per-modality capture; no pixel or
  audio-span attribution; architectural deltas from the paper's models
  (sliding window, KV sharing, softcap); bf16 backward noise.

## 9. Reproducibility

- Pinned revisions, fingerprinted artifacts, deterministic exporters and
  condition IDs, committed configs/manifests, CPU test suite.
