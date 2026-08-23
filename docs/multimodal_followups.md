# Multimodal follow-ups: localization, a new property, and the asymmetry

Three studies that sit on top of the confirmed pooled multimodal result, plus
the corrections that constrain how any of them may be described.

## What the confirmed result is

* `google/gemma-4-E4B-it`, pooled multimodal J-lens fitted on interleaved text,
  image and spoken-audio examples.
* Exact coordinate exchange at alpha=1, contiguous layers 16-40, every original
  prompt position patched, direction `bird -> cat`.
* Free model output. No candidate list. No teacher forcing.
* Fresh confirmation: text 14/16, image 16/16, spoken audio 15/16; combined
  zero/random/unrelated controls 1/144. Verdict
  `FRESH_MULTIMODAL_CONFIRMATION_GO`, report checksum
  `sha256:2bb6dcc1346229573566125bc8d91c782247d55af5091f4215d98bb621472ff7`.

Reports: `broad_pooled_multimodal_j_workspace_report.json` (development) and
`fresh_multimodal_confirmation_report.json` (confirmation). Both are immutable
evidence; nothing in this document rewrites either.

## Four corrections baked into the code

1. **`cat -> bird` was tested.** It produced 0 successes in 24 trials at
   alpha=1, against 24/24 for `bird -> cat`. All six development directions are
   recorded in `development_direction_record()`, none as "untested", and the
   asymmetry is *not* asserted as a property of the representation: capability,
   prompt behaviour, coordinate quality and concept geometry remain live
   alternatives. Experiment C tests whether the difference replicates.
2. **Leg count cannot carry a property-generalization claim.** `bird=2`,
   `cat=4`, `zebra=4`, `giraffe=4`, so `bird->cat`, `bird->zebra` and
   `bird->giraffe` all test the same 2 -> 4 answer change and `cat->zebra`
   changes nothing observable. See `leg_count_property_limit()`.
3. **All 64 confirmation candidates are spent.** The confirmation opened 64
   photographs during capability screening (192 capability rows) and recruited
   16 of them. A photograph is spent once the model has been run on it at all,
   so all 64 are excluded from every later population — see
   `load_spent_confirmation_population()` and `exclusion_universe()`.
4. **Only the pooled lens spans L16-L40.** The broad study fitted the pooled
   early shard L16-L32 and combined it with the pooled L33-L40 shard. The
   text-only, image-only and spoken-audio-only lenses cover L33-L40 only. A
   four-arm L16-L40 comparison would require fitting L16-L32 for those three
   arms and is deferred.

## Experiment A — exploratory band localization

Same lens, direction, alpha, prompt, endpoint, controls and position rule as the
confirmed study. Only the band varies. The population is the already spent broad
development population, so **the whole analysis is exploratory and descriptive**
and nothing from it may be reported as confirmation.

The grid (`localization_grid()`, frozen with its analysis rule before any
sub-band outcome exists) has 15 bands in three families:

| family | bands | what it can show | what it cannot |
| --- | --- | --- | --- |
| suffix | 6, all ending at L40 | how late a band can start and still carry the effect | an onset — start layer and band length move together |
| prefix | 5, all starting at L16 | a sufficient early region | that a later region is not involved; same length confound |
| partition | 5 disjoint 5-layer windows covering L16-L40 | individual sufficiency of a region, without a nesting confound | necessity; a distributed or redundant code can fail every window while the full band passes |

`localization_claim_boundary()` refuses an onset claim in both cases — outright
for a nested passing chain, and on population/design grounds even for disjoint
passing windows. No band is claimed necessary: the complement of each band was
never ablated.

## Experiment B — a non-leg-count property

**B0 audit.** `PROPERTY_FAMILIES` declares candidates. `body_covering` is the
first choice; `animal_sound` is the fallback.

| family | admissible | refused, with reason |
| --- | --- | --- |
| body covering | bird=feathers, cat=fur, dog=fur, sheep=wool/fleece, bear=fur | horse, cow, zebra, giraffe, elephant — the correct surface answer is genuinely contested |
| animal sound | cat=meow, dog=bark/woof, cow=moo, sheep=baa/bleat | **bird** (gulls, ducks, pigeons and raptors do not share one conventional answer), horse, zebra, giraffe, elephant, bear |

`audit_property_family()` then filters by fresh-media availability and by clean
capability in all three modalities, and admits a direction only when both
endpoints survive and their answers differ (so `cat->dog` is rejected —
identical answer). The endpoint is unrestricted complete generation, scored
after the fact against the declared alias set; single-token answers are not
required, and no candidate list or teacher forcing exists anywhere.

**B1 development** runs the alpha=1 exchange over L16-L40 with zero, random and
unrelated controls on fresh media disjoint from every prior population. A NO_GO
closes confirmation. A control failure gets its own verdict
(`NEW_PROPERTY_DEVELOPMENT_CONTROL_FAILURE`) so it is never reported as a null.

**B2 freeze** writes `frozen_new_property_design.json` — property, prompt,
aliases, pair, lens checksum, layers, alpha, position rule, controls,
thresholds, tests, Holm correction, recruitment rule and exclusion digest —
before any confirmation photograph is opened. **B3** refuses to run without it,
recomputes the exclusion universe and refuses if it differs from the frozen one.

## Experiment C — prospective asymmetry replication

The identical leg-count protocol run backwards on fresh cat media, under a
design frozen before any photograph opens. A null replicates the observed
development difference and explains nothing about its cause; a clear effect
would show the development failure did not replicate and would retire the
asymmetry claim entirely. `asymmetry_replication_verdict()` reports it in those
terms and records `cause_of_asymmetry_identified: false`.

## Engineering contract

Every stage: one atomic checksum-verified JSON per unit on Drive, a
`RunFingerprint` that refuses a changed configuration instead of mixing runs,
at most one in-flight trial lost to a disconnect, zero lens fits and zero
backward passes, and a printed stage map and exact pass budget before the model
loads. Development and confirmation artifacts live in separate run
directories. See `jlens/mmpilot/multimodal_followup.py`, its MOCK world in
`multimodal_followup_mock.py`, and the notebook
`notebooks/multimodal_jspace_matched_jlens_colab.ipynb` (sections 12-18).
