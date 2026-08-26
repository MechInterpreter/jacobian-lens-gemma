# Country-workspace generalization benchmark

This study is a prospective attempt to broaden the confirmed bird-to-cat
leg-count result. It does not modify or reclassify that result.

## Question

If a pooled multimodal J-lens exchanges a country's identity coordinates, does
Gemma 4 recompute facts associated with the injected country? The benchmark
tests capital and continent for France and China and for Japan and Egypt, in
both directions, through text, flag images, and spoken country labels.

The claim ladder is explicit. A partial result requires one direction to pass
both facts in every modality. Bidirectional requires both directions of a pair.
Generalized requires at least one passing direction from each of the two pairs.
Full-grid requires all four directions. No finite grid is called universal.

## Why this dataset

SpokenCOCO cannot answer this question because its identities are COCO objects.
The pinned `tokeron/country-flags-variations` revision contains 20 independently
generated and image-verified flag variants for each evaluation country. The
design uses four for development and fourteen for confirmation, leaving two
outcome-blind reserves in case OCR detects a country or answer word. The eleven
countries used for lens fitting never appear in evaluation.

Audio is deterministic synthetic speech of the country label. It is saved as
16 kHz mono float32 WAV and passed through Gemma's native audio path. The
transcript is stored only as provenance and is never passed to the backend.
This supports a spoken-language claim, not a natural-environmental-audio claim.

## Frozen method

- Fit one pooled multimodal average-Jacobian lens on 99 examples: 33 text, 33
  image, and 33 spoken audio.
- Fit source layers L16 through L40 to target layer L41 in float32 on an 80 GB
  A100. Save the lens in float32 and checksum it.
- Use only exact alpha-one two-coordinate exchange at every original prompt
  position. There is no alpha sweep.
- Generate answers greedily from the unrestricted vocabulary. There is no
  teacher forcing and no candidate list.
- Select one path per property from six predeclared contiguous bands using only
  clean capability and a cumulative norm-matched direct-answer positive
  control. Exact identity-swap outputs are unavailable during selection.
- Compare exact exchange with zero, random-basis, and unrelated-country
  controls on development.
- Freeze passing directions before opening confirmation outputs.
- In confirmation, compare exact success with the strongest negative-control
  outcome for each photograph and Holm-correct across the predeclared cells.

## Resume and refusal behavior

CPU preparation writes one checksummed JSON per media unit and then seals the
complete population. Lens fitting checkpoints every five examples. Every
capability, localization, development, and confirmation condition is an atomic
`UnitStore` JSON. A disconnect loses only an incomplete condition or the
current fit checkpoint batch. A changed model, dataset, protocol, population,
layer grid, prompt rule, or intervention configuration opens a different run
or is refused rather than mixed.

The workflow stops without opening confirmation if clean capability, direct
answer leverage, or development fails. This is a scientific outcome, not a
notebook error.

## Run order

1. CPU: enable only Stage 0. This downloads the pinned dataset, audits images
   with OCR, renders audio, and seals the population.
2. 80 GB A100: enable only Stage 1 and the model/fp32/fit confirmations.
3. Same or later A100 session: enable Stage 2. Stop if either printed verdict is
   `NO_GO`.
4. Enable Stage 3. It freezes a confirmation design only for directions that
   pass both properties across all three modalities.
5. Enable Stage 4 only when the design file exists. Then enable Stage 5 to write
   the consolidated report.

The canonical notebook is
`notebooks/multimodal_country_workspace_generalization_colab.ipynb`.
