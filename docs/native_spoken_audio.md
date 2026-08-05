# Native spoken audio on Gemma 4 E4B IT

This note records why `spoken_audio` was blocked, what the blocker actually was,
what the supported native path is, and what enabling it would and would not
license scientifically.

**Nothing here is evidence about cross-modal transfer.** It is about whether a
waveform can reach the model at all.

## The blocker, exactly

The pilot recorded:

> processor/model produced audio features but zero audio placeholder tokens;
> spoken_audio disabled without transcript substitution

That observation is correct and reproducible. It is **not** a statement that the
pinned checkpoint lacks audio. Against `google/gemma-4-E4B-it` at revision
`fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` with `transformers==5.13.1`:

```python
processor(text="Answer with exactly one word.", audio=waveform, return_tensors="pt")
# input_ids           (1, 11)
# input_features      (1, 99, 128)
# input_features_mask (1, 99)
# audio placeholders  0          <-- and no exception
```

### Why it happens

`Gemma4Processor` inherits `ProcessorMixin.__call__`. That method computes audio
features first, then calls `get_text_with_replacements`, which **expands
occurrences of `processor.audio_token` that are already present in the text**. A
plain prompt has none, so the features are computed and scattered into nothing.

### Why nothing raised

Two checks that look like they would catch it do not:

- `Gemma4Processor.validate_inputs` validates that the `<|image|>` count matches
  the number of images. It has **no audio equivalent**.
- `ProcessorMixin._check_special_mm_tokens` compares the audio-token count in
  the text against the count in `input_ids`. Both are zero, so it passes.

The mismatch surfaces only inside `Gemma4Model.forward`:

```
ValueError: Audio features and audio tokens do not match, tokens: 0, features: 25
```

So a bare processor call yields an input that is silently wrong at build time
and fatal at forward time. Blocking `spoken_audio` was the right call on the
evidence available; the diagnosis just stopped one layer short.

## The supported native path

Pass the waveform as a **chat-template audio content block**. The checkpoint's
`chat_template.jinja` renders `<|audio|>` for a block whose `type` is `audio`,
and the processor then expands that token against the features:

```python
processor.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "audio", "audio": samples},   # float32 mono ndarray @ 16 kHz
        {"type": "text", "text": prompt},
    ]}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
)
```

Measured against the pinned revision:

| input | `input_features` | audio placeholders |
| --- | --- | --- |
| bare call, 1.0 s | `(1, 99, 128)` | **0** |
| content block, 0.5 s | `(1, 50, 128)` | 13 |
| content block, 1.0 s | `(1, 99, 128)` | 25 |

The span is `<\|audio>` + N × `<\|audio\|>` + `<audio\|>`, contiguous, with N
derived from the feature mask through two stride-2 convolution reductions — the
same quantity `Gemma4Model.forward` compares against `audio_mask.sum()`.

### Resolved protocol

| property | value |
| --- | --- |
| model | `google/gemma-4-E4B-it` @ `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` |
| processor | `Gemma4Processor` |
| feature extractor | `Gemma4AudioFeatureExtractor`, 16 000 Hz |
| waveform | float32, mono, 1-D |
| processor argument | `audio` |
| call convention | `chat_template_audio_content_block` |
| placeholder token | `<\|audio\|>`, id `258881` |
| bracket tokens | `<\|audio>` id `256000`, `<audio\|>` id `258883` |
| feature keys | `input_features`, `input_features_mask` |
| audio tower | `audio_config` present, `model_type=gemma4_audio` |

## What the code does about it

`jlens/mmpilot/audio.py` implements the protocol centrally:

- `resolve_audio_interface` **probes** the path with a generated waveform at two
  durations and requires a verified placeholder span, or raises
  `SpokenAudioUnsupportedError` naming which link broke. Component presence is
  never taken as support.
- `prepare_waveform` coerces to float32 mono and **refuses a sample-rate
  mismatch rather than resampling** — `load_audio` passes an ndarray through
  untouched, so a 22 kHz array handed to a 16 kHz extractor is silently
  reinterpreted as a slower, lower-pitched recording.
- `verify_audio_encoding` turns the three fatal states into named refusals:
  features without placeholders, placeholders without features, and a count
  mismatch between them.
- `check_audio_tower_invoked` hooks the tower, because a placeholder span proves
  only that the *text* was built correctly.
- `assert_no_text_leakage` keeps captions, transcripts and file stems out of the
  prompt, enforced in `pipeline._build_inputs_for` where the group's own caption
  and paths are known.

`GemmaPilotBackend.supports("spoken_audio")` is True **only** when a probed
interface is attached. `build_real_backend(..., resolve_audio=True)` is opt-in
and defaults to False, so no existing run's behavior changes.

`jlens/mmpilot/audio_audit.py` runs the feasibility checks and computes the
verdict:

- **`AUDIO_READY`** — every required check passed; the channel is technically
  usable. Engineering evidence only.
- **`AUDIO_BLOCKED`** — the checkpoint, processor or library cannot support the
  path. Nothing was measured, so nothing may be called invalid.
- **`AUDIO_INVALID`** — an input *was* built and then failed a check that makes
  later numbers untrustworthy. This is the dangerous state: it looks like
  success.

`notebooks/gemma4_native_spoken_audio_feasibility_colab.ipynb` runs it, with
`RUN_REAL_AUDIO_AUDIT`, `RUN_MODEL_STAGE` and `CONFIRM_MODEL_LOAD` all False.

## Candidate scoring: validity is not capability

The first real audit passed every audio check and still returned
`AUDIO_INVALID`, on `candidate_sequence_scoring` — while scoring executed and
returned finite scores for both candidates:

```
cat   ids [5866]  n_tokens 1  sum -21.9234
dog   ids [4799]  n_tokens 1  sum -18.0171
```

**The scorer was not defective.** The audit was handed the pilot's behavioral
concepts, and Gemma encodes `" cat"` and `" dog"` as **single tokens**, so the
`any(len(ids) > 1)` term of the old rule was False. The rule reported a fixture
that could not exercise the multi-token path as though the scorer had failed.

`SCORING_VALIDITY_RULE = jlens.mmpilot.scoring_validity.v2` separates the two
questions that were run together:

- **Scoring validity** — does the mechanism score *complete* sequences
  correctly? This is what the audit measures.
- **Behavioral capability** — can Gemma recognize a concept from a recording?
  **Not measured, and not measurable here**: the audit's waveform is not
  selected to be about any candidate. That belongs to the SpokenCOCO experiment.

### The selection rule

`select_scoring_candidates` encodes a fixed, predeclared pool
(`SCORING_CANDIDATE_POOL`, COCO category names) through the backend's own
`encode_candidate` and takes the **first pair in pool order** satisfying, in
priority order: every candidate non-empty; token sequences distinct; neither a
prefix of the other; **both** multi-token if such a pair exists, otherwise at
least one. No phrase is assumed to be multi-token — assuming that is what caused
the original failure. If nothing qualifies it raises `ScoringCandidateError`
rather than degrading the check.

Under the pinned tokenizer this selects `traffic light` `[8827, 2214]` and
`fire hydrant` `[4304, 67175]`.

### The PASS criteria

`candidate_sequence_scoring` passes when, and only when:

- scoring executes and every candidate has a non-empty token sequence;
- `n_tokens` matches the ids supplied;
- at least one candidate is multi-token, and no candidate's tokens are a prefix
  of another's;
- every per-token term and every aggregate is finite;
- `sum_logprob` equals the sum of its own recorded `token_logprobs` within
  tolerance, and `mean_logprob × n_tokens` agrees;
- reversing the candidate order leaves each candidate's own score unchanged
  within tolerance.

It explicitly does **not** require the recording to contain a candidate concept,
a particular winner, semantic correctness, equal token lengths, or first-token
agreement. The highest-scoring candidate is recorded as
`reported_only_argmax` and is never a criterion.

`score_candidate_sequences(..., return_token_logprobs=True)` supplies the
per-token terms, so the aggregate is checked against the scorer's own terms
rather than against a second implementation. The flag is **off by default**, so
the units the scientific stages persist keep exactly the fields they had.

`AUDIT_VERSION` is now `…feasibility.v2`, and both `scoring_validity_rule` and
`scoring_candidate_token_ids` enter the audit's `report_checksum`: a verdict
reached under a different rule, or against candidates that tokenize
differently, is not the same verdict.

## Fingerprint consequences

Any change to **any** of the following must change the run fingerprint:

- model id
- model revision
- processor revision
- Transformers version
- audio protocol (`AUDIO_PROTOCOL_VERSION`)
- placeholder convention (token, id, bracketing, count rule)
- hook-position convention (which position is captured or edited)

`ResolvedAudioInterface.protocol_fingerprint` is a `sha256` over exactly those
configuration fields. Probe counts and notes are deliberately excluded, so two
machines resolving the same protocol agree. Bind it in:

```python
RunFingerprint(..., extra={"audio_protocol_fingerprint": resolved.protocol_fingerprint})
```

`extra` participates in the digest, so a run recorded under one protocol refuses
to resume under another, and a text-and-image run recorded without the key
refuses to resume as an audio run.

## J-lens compatibility

The validated lens
(`runs/text_jlens_early_layer_recalibration_v2/artifacts/lens.validated.pt`,
`sha256:4b17bf60…3641c2b`) was fitted and natively validated **against model
revision `fa62d88d…765f0cd`**, and `load_validated_lens` refuses any other.

Native spoken audio works on that same revision, so **no checkpoint migration is
required and the existing lens remains applicable.**

Had it required a different checkpoint or revision, the consequences would have
been: the current lens could not be reused, a new lens would have to be fitted
and natively validated against the migrated checkpoint first, and every existing
scientific fingerprint — pilot, robustness, localization — would cease to be
comparable. That is a separate decision, not a side effect of enabling a
modality, which is why `build_real_backend` never substitutes a checkpoint.

## What `AUDIO_READY` does not mean

`AUDIO_READY` says a waveform reaches the audio tower, changes the output, and
leaves the capture and intervention harness undisturbed.

It does **not** say:

- that Gemma recognizes concepts from spoken captions;
- that the frozen text-calibrated J-lens exposes anything useful on audio
  activations;
- that concept directions transfer to or from spoken audio;
- that the spoken-audio condition would pass the behavioral capability gate.

Those are the questions a spoken-audio J-space study would ask, and none of them
has been asked. The plumbing working is a precondition for that study, not a
preview of its result.

## SpokenCOCO caveat, unchanged

SpokenCOCO carries spoken *readings of written captions*. A `spoken_audio`
condition built on it tests **spoken-linguistic** evidence, not environmental
sound. No barking, no sirens, no instruments. That limitation is a property of
the dataset and is not affected by anything in this note.
