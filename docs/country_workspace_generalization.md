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

### No-refit causal-site diagnosis

The broad France-to-China identity exchange changed the generated country name
but did not reliably change its capital or continent. The exact intervention
often reduced the answer to punctuation, even though its numerical integrity
checks passed. This means the broad all-position L16-L40 exchange is not yet
evidence that the internal country state used for fact lookup was replaced.

The next diagnostic therefore performs no fitting and opens no fresh
confirmation data. It divides the four France development examples before any
new result is read:

1. One example screens the six frozen contiguous bands at two declared sites:
   the evidence endpoint and the final prompt token.
2. A path passes only if an actual clean China activation and a norm-matched
   direct-answer control both produce the correct capital and continent in
   text, image, and spoken audio. Replacing the state with the original France
   activation or an unrelated Italy activation must not do so.
3. Path selection never reads an exact coordinate-swap outcome.
4. If and only if the screen passes, the exact alpha-one France-to-China
   coordinate exchange is tested on the remaining three development examples,
   against zero, random-basis, and unrelated-country controls.

Every screen and follow-up condition is checksum-bound and atomically saved.
The CPU planning stage hashes the completed balanced-task lens but does not
load it. The GPU stages reuse that lens and perform zero backward passes. A
successful development diagnostic can motivate a separately frozen fresh
confirmation study; it does not itself reclassify the completed country result.

The completed screen produced a split diagnostic. Its direct-answer J-lens
control failed at every tested path, so the original conjunctive screen remains
`COUNTRY_CAUSAL_SITE_SCREEN_NO_GO`. In contrast, replacing the final prompt
state with a real clean China state produced both Beijing and Asia in text,
image, and spoken audio for L24-L31, L28-L35, L33-L40, and L16-L40. Self-state
and unrelated-state controls remained at zero, with every integrity check
passing. A versioned amendment therefore freezes the shortest state-valid path,
L24-L31 at the final prompt token, without reading any exact J-lens swap result.
On the three development examples not used by the screen, it reports two
separate arms: replication of the full-state transfer and the exact alpha-one
J-lens exchange at that same site. The original screen verdict is preserved.

### Direction-scoped capability amendment

The first capability run passed 34 of 36 cells. France, China, and Japan passed
all source-evidence cells; two Egyptian flag images were confused with nearby
flags and produced Riyadh instead of Cairo. The original implementation blocked
the entire study unless all four countries were usable as source evidence.
That was broader than the causal requirement: a failed Egyptian input can block
`Egypt->Japan`, but it cannot invalidate `Japan->Egypt`, where Japan is the
only supplied evidence and Egypt is the injected coordinate target.

The versioned `mmpilot.country_direction_capability.v2` amendment therefore
gates each direction by its source country. It admits `France->China`,
`China->France`, and `Japan->Egypt`, which still cover both predeclared pairs.
It opens confirmation only if development passes both properties for at least
one direction in each pair. This amendment was frozen before localization,
exact-swap development, or any confirmation output was opened. It creates a
new run fingerprint and checksum-reuses the completed fp32 lens; it does not
refit or reinterpret any causal outcome.

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

For the no-refit diagnosis, run Stage 6A alone on CPU to seal the plan. Then run
Stage 6B on an fp32 80 GB A100. Run Stage 6C only if Stage 6B prints
either a conjunctive passing path or a separately recorded state-valid path.
Stage 6C runs 54 full-state and 72 J-lens conditions on the three remaining
development examples. Stages 6A-6C must never be combined with a lens fit
toggle.

The canonical notebook is
`notebooks/multimodal_country_workspace_generalization_colab.ipynb`.
