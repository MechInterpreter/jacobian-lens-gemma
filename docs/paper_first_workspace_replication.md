# Paper-first workspace replication

This protocol separates three questions that the earlier experiments partially
mixed together.

1. Can Gemma reproduce Anthropic's text-only causal result with the same exact
   two-coordinate exchange?
2. Before intervening on multimodal examples, is the source concept visibly
   loaded in the J-lens coordinates at any layer and position?
3. After choosing the pair, band, and positions from clean loading only, does
   the result replicate on photographs and recordings that were not used for
   development?

The canonical runner is
[`notebooks/multimodal_jspace_workspace_replication_colab.ipynb`](../notebooks/multimodal_jspace_workspace_replication_colab.ipynb).

## Why the order matters

The text stage is an apples-to-apples check on the intervention before a
multimodal extension is interpreted. It uses the validated text-only L33–L40
lenses, the paper's all-prompt-position rule, unrestricted next-token output,
and `alpha=1`, which is the exact coordinate exchange.

The development stage performs no intervention. It captures the clean residual
stream and reports cosine loading and the two-coordinate read for the source and
target concepts. A pair is selected by its weakest modality, and a layer passes
only when source loading beats the target/unrelated control in every required
modality. The longest contiguous passing run is selected. Image/audio evidence
positions are used only when their loading beats non-evidence positions; text
always retains the all-position rule because it has no processor evidence span.

Only then is a confirmation design written. The primary alpha remains one.
The exploratory alpha `.75` from the completed dose-response run is carried as
a labelled interpolation sensitivity, not promoted into an exact exchange.

## Fresh population

Fresh means zero image or group overlap with both the development population
and every prior population whose identities are included in the pinned source
artifacts. The confirmation population is selected before its model answers are
opened, and the disjointness proof is part of the report.

## What a positive result would license

The strongest verdict requires all of the following:

- the text-only paper replication passes;
- clean loading licenses a contiguous multimodal band;
- the clean model can answer identity and leg-count questions in every channel;
- the alpha-one exact swap changes unrestricted identity and downstream
  property outputs more often than random and unrelated swaps on fresh media.

Development loading, target-logit movement, alpha `.75`, or a candidate-restricted
score cannot substitute for that result.

## Resume behavior

Each text task, clean loading sample, and causal condition is written as a
checksum-valid `UnitStore` JSON unit immediately after completion. A rerun with
the same fingerprint reuses it. Any change to the model, lens source, manifest,
population, task list, pair list, layers, thresholds, prompt protocol, or alpha
changes the fingerprint and refuses to mix results.
