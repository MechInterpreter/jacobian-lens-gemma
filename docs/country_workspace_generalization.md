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

That localized follow-up transferred the real China state in 16 of 18 cells:
6/6 text, 6/6 image, and 4/6 spoken audio. The two failures were the same audio
source/donor pair across both properties. Self-state and unrelated-state
controls remained at zero. The exact J-lens exchange remained at 0/18, with
all hook and numerical-integrity checks passing. This localizes a causal state
but also shows that the much smaller J-lens edit is overwritten or bypassed in
the ordinary prompt computation.

Stage 6D tests one specific explanation without refitting. The evidence is
still supplied normally and may influence the final prompt state below L24.
At L24 the final prompt token can no longer attend to the earlier prompt
prefix. Every generated token is also blocked from attending to that prefix at
every layer, which closes indirect bypasses through prompt tokens that had
already read the evidence. Generation can still attend to the encoded final
prompt token, making it the sole state bottleneck. The stage first
requires the sealed clean state to retain France facts and the sealed real
China state to produce China facts. It then tests the exact alpha-one J-lens
exchange against zero, random-basis, and unrelated-country controls at the
already selected L24-L31 final-token path. This is a controlled bottleneck
diagnostic, not a relabeling of the paper method. It uses only the already-open
development examples, writes one atomic unit per condition, performs zero
backward passes, and cannot open confirmation.

Stage 6D itself returned `COUNTRY_SEALED_EVIDENCE_NO_GO`, but its arms were not
comparable. The seal destroyed ordinary completion: the clean France baseline
was 0/18 and generated fragments such as `called`, `not`, and repeated `the`.
Meanwhile, the target-state arm repeatedly restored an unsealed China state at
every layer and reached 16/18. The result therefore diagnoses a destructive
bottleneck and an unmatched state scaffold; it is not evidence that the
J-lens failed under a capability-preserving bottleneck.

Stage 6E is the single frozen repair. Under the same seal and L24-L31 path, it
restores the same unsealed France final-token state in every coordinate arm.
Forward hooks are ordered and audited so that the source-state scaffold runs
first and the coordinate exchange runs second. The exact, zero, random, and
unrelated conditions therefore differ only in the declared two-coordinate
operation. Separate self-, target-, and unrelated-state arms require the
scaffold to preserve Paris/Europe and retain full-state causal leverage. There
is no layer search, alpha search, refit, backward pass, or fresh confirmation
access. This makes the outcome decisive for this specific matched-scaffold
bottleneck rather than guaranteeing a positive result.

Stage 6E preserved the full-state controls and produced the first coordinate
signal in this country study. For continent, the exact France-to-China
exchange produced Asia in 0/3 text, 1/3 image, and 2/3 spoken-audio examples;
zero, random-coordinate, and unrelated-country controls were 0/3 in every
cell. Every hook, numerical, scaffold-order, and exact-exchange check passed.
Capital remained 0/3 in every coordinate arm. The aggregate Stage 6E verdict
therefore remains `COUNTRY_MATCHED_SCAFFOLD_NO_GO`, but the pooled image plus
spoken-audio continent result is a prespecified candidate for one fresh test,
not another development search.

Stage 6F freezes that test on CPU before reading any output associated with
the 28 reserved France and China confirmation examples. It records their unit
IDs, audits every JSON artifact under the run root for prior generated output,
and pins France-to-China, continent, exact alpha one, L24-L31, the final prompt
token, and the matched France scaffold. Image and spoken audio form the pooled
primary endpoint; text is reported as secondary because development showed no
text success. The primary gate requires at least 50% pooled exact success, a
25-point margin over each of zero, random, and unrelated controls, evidence in
both primary modalities, familywise-corrected paired significance, intact
coordinate diagnostics, and successful real-state controls. Stage 6G runs
that frozen design once. It performs no fitting or search and saves every
condition atomically so a disconnected runtime can resume without repeating
completed work.

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
development examples. Stages 6A-6D must never be combined with a lens fit
toggle.

After Stage 6C, Stage 6D can be run by itself on the same fp32 A100. It adds
126 forward-only development conditions and reuses the checksum-pinned
balanced pooled J-lens. All fit toggles must remain false. A runtime restart
checksum-reuses every completed condition.

After the recorded Stage 6D failure, run Stage 6E alone. It adds 126
forward-only development conditions and atomically resumes each condition.
Every fitting toggle and every earlier scientific stage must remain false.

After Stage 6E, run Stage 6F alone on CPU. If its freshness audit is clean,
stop that runtime. Run Stage 6G alone on an fp32 80 GB A100 with the model and
fresh-confirmation budget confirmations enabled. Stage 6G runs 294
forward-only conditions, performs zero backward passes, and resumes at the
individual-condition level.

The canonical notebook is
`notebooks/multimodal_country_workspace_generalization_colab.ipynb`.
