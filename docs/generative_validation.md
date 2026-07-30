# Generative J-cone steering validation — guide

Branch: `experiment/generative-jlens-validation`
· Backend: [`jlens/generative.py`](../jlens/generative.py)
· Runner: [`scripts/run_generative_validation.py`](../scripts/run_generative_validation.py)
· Config: [`configs/gemma_generative_validation.yaml`](../configs/gemma_generative_validation.yaml)
· Benchmark: [`configs/generative_benchmark.json`](../configs/generative_benchmark.json)
· Tests: `tests/test_generative.py`, `tests/test_generative_runner.py`

## The question

Can a **weighted reconstruction of the active J-space generators** at layer
`l` — `q_C = Σ_{i∈C} a_i v_i` over the raw dictionary atoms (rows of
`W_U J_l`), with the pursuit's own nonnegative coefficients — injected at
the same layer of a **neutral verbalization prompt**, make Gemma's native
autoregressive decoder produce the correct **multi-token** concept, more
specifically than matched controls, and at least as well as raw activation
transplantation?

Terminology guard: the *active cone* is the positive span of the selected
generators; the steering vector is the **weighted reconstruction** tied to
one activation (not an average); manually selected or thresholded subsets
are *candidate semantic groups*, not proven bindings. Nothing here claims to
solve the binding problem.

## Protocol

1. Capture `h` at the established `block_output` site (`layers[l]` output —
   post attention/MLP/PLE/`layer_scalar`), final token of the source prompt.
2. Fresh nonnegative gradient pursuit (k=10) against the layer's J-space
   dictionary gives active generators `v_i` and coefficients `a_i`.
3. Build the condition vector (19 conditions, below), rescale it — either to
   ‖δ‖ = ratio · ‖h_recv‖ (`h_recv` the unsteered residual at the injection
   site of the neutral prompt; requested **and** measured ratios are
   recorded), or, for `natural_scale` and its matched controls, to a fixed
   observed norm instead of a ratio (below).
4. Inject at the neutral prompt's final token position under a schedule:
   `prompt_only`, `constant` (every generated position), or `decaying`
   (weight `decay^offset`).
5. Score the target phrase teacher-forced (`Σ_t log P(y_t | y_<t, q)`), and
   (for the configured subset) decode greedily **uncached** — every step is
   a full forward pass, so injection semantics are exact.

Conditions: `none`, `zero` (parity), `full_cone`, `natural_scale` (the cone at
its **own** norm, unscaled), `mass_subcone` (smallest subset covering 60/70/80%
of positive coefficient mass), `manual_subcone` (manifest-provided indices),
`unrelated_cone` (another example's cone, **from the same split**),
`random_matched_norm`, `shuffled` (coordinate permutation), `sign_reversed`,
`wrong_layer`, `wrong_position`, `raw_activation`, `activation_diff` (source
minus matched control prompt); plus five **natural-scale-matched** controls —
`natural_unrelated_cone`, `natural_random_matched_norm`, `natural_shuffled`,
`natural_sign_reversed`, `natural_mass_subcone` — below.

### Natural-scale-matched controls

`natural_scale`'s low-strength gain (`+2.9` to `+3.1` mean target log-prob vs
zero at layer 14 on `dev-phrase-solar-eclipse`, varying by schedule) cannot by
itself show the gain is *specific to the correct direction*: `GONOGO_CONTROLS`
(random/shuffled/sign-reversed/unrelated) were only ever evaluated at a
calibrated *ratio*, i.e. at a different injected norm than `natural_scale`'s
own — and only under `prompt_only`. A control that "loses" at a different
magnitude and a narrower schedule set proves nothing about direction.

The five `natural_*` controls close that gap: each builds the exact same
*direction* as its ratio-scaled counterpart (same atoms, same donor, same
seed convention — see `jlens.generative.build_condition_vector`), then
rescales it via `jlens.generative.scale_to_norm` to `natural_scale`'s own
observed delta norm for that (example, layer) — not to any ratio of the
receiving activation — and sweeps every schedule `natural_scale` itself runs
under, not just `prompt_only`. `sign_reversed`/`shuffled` need no rescaling in
practice (negation and coordinate permutation both preserve norm exactly);
`unrelated_cone`/`mass_subcone`/`random_matched_norm` generally do.

Every condition's provenance is tagged with `jlens.generative
.condition_scaling_mode`: `"none"` (`none`/`zero`), `"natural_unscaled"`
(`natural_scale` — the reference), `"natural_matched"` (the five controls
above), or `"ratio_scaled"` (everything else). `summarize_by_condition` never
merges across conditions — `natural_unrelated_cone` and `unrelated_cone`
stay separate rows even though one is a rescaled copy of the other's
direction — so ratio-scaled and natural-scale-matched results cannot be
silently averaged together; `scaling_mode` just makes the grouping legible
without re-deriving it from the condition name.

`jlens.generative.natural_scale_verdicts` / `natural_scale_gonogo_report`
answer the motivating question directly, per (example, layer, schedule) since
there is no ratio to calibrate on: does `natural_scale` beat zero and every
`NATURAL_SCALE_GONOGO_CONTROLS` entry at the *same* injected norm? Written to
`artifacts/natural_scale_comparison.json` and mirrored into
`run_metadata.json` as `natural_scale_gonogo`, alongside (not merged with) the
ratio-scaled `gonogo.json`.

### Strength: the informative region is low

Weighted J-cone reconstructions are intrinsically small relative to the
receiving residual (layer-21 pilot cones explain ~1e-5 of it), so the sweep runs
`0.01, 0.03, 0.05, 0.1, 0.25` — bracketing where the cone naturally sits — and
retains `0.5, 1.0, 2.0` only as **stress tests**. At those magnitudes the
injection dominates the residual, so an effect there says little about whether
the cone means anything, and a failure there is not evidence against the
hypothesis.

`natural_scale` skips rescaling entirely and records `natural_delta_norm`,
`receiving_activation_norm`, and `natural_ratio` — the measurement that says
where the cone actually lives, rather than assuming a ratio. It carries
`requested_ratio: null` and always decodes.

### Benchmark requirements

Enforced, not merely documented:

- **Every target must tokenize to 2–6 tokens** under the pinned tokenizer,
  checked by `jlens.generative.validate_target_tokens` after the model loads
  (the count is a property of the checkpoint vocabulary). Violations abort the
  run, listing every offender's example id, phrase, token ids, and token
  strings. Check it in seconds beforehand with
  `scripts/validate_benchmark_targets.py` (tokenizer only, no weights, no GPU).

  A single-token target is not a small problem: it tests no multi-token scoring,
  and because every schedule injects identically at the prompt-final position
  and differs only at *generated* positions, it forces `prompt_only`,
  `constant`, and `decaying` to identical target log-probabilities — three
  duplicate rows and no schedule signal. `dev-split-photosynthesis` did exactly
  this (single id 93036), so single common words were replaced with rarer,
  morphologically complex forms.

- **Unrelated-cone donors come from the same split**, via
  `jlens.generative.select_split_examples`. `--limit-examples 1` used to leave
  one dev example and fill the donor slot from `heldout`, so the dev smoke
  borrowed `held-split-metamorphosis`. A development run must never read a
  held-out example or vector. Each split now leads with multi-word targets so
  `--smoke` (first example, second as donor) lands on genuinely multi-token
  targets, and a split with fewer than two examples fails loudly rather than
  reaching across.

- **Target token strings are recorded** next to the ids in every record
  (`target_token_strings`) and in `artifacts/targets.json`, so segmentation is
  auditable without the tokenizer.

Hard gates before any steering (any failure aborts): architecture
verification, lens SHA-256 fingerprint, zero-vector logit parity within
`parity.max_abs_logit_diff_tol`, and manual-uncached vs `generate()` greedy
equivalence. The reference `generate()` call explicitly sets
`do_sample=False`, `num_beams=1`, `use_cache=False` and matching EOS handling,
because the Gemma 4 E4B-it checkpoint's stored generation config defaults to
`do_sample=True, top_k=64, top_p=0.95`.

### Reading logits the way the model does

Every logit in these gates — and in every recorded measurement — is read
through `LensModel.logits_from_ids`, the model's own head. Two distinct
mistakes are ruled out there, both of which produced first-step mismatches on
the real checkpoint:

1. **Double final norm.** HuggingFace text models apply the final norm
   *before* returning `last_hidden_state`, so
   `unembed(forward(ids).last_hidden_state)` applies it twice. `unembed` is for
   residual-stream activations captured from block hooks, not for
   `last_hidden_state`. This changed the argmax outright.
2. **LM-head GEMM shape.** `GenerationMixin` sets `logits_to_keep=1` for any
   model whose forward accepts it, and Gemma 4 slices the hidden state *before*
   the LM head. So `generate()` runs a `[1, 1, d_model]` head while a
   full-sequence read runs `[1, seq, d_model]`. Same math, different reduction
   order — bit-identical results once matched (verified in float32), but a
   0.125 first-step log-probability gap in BF16 on an L4 when not matched.
   `logits_from_ids(ids, n_last=...)` forwards `logits_to_keep`, and decoding
   and scoring request exactly the positions they need.

### Why the greedy gate is not a max over the vocabulary

The gate asserts **token equality** unconditionally, plus first-step
distribution agreement: argmax, top-k set, top-k log-probabilities
(`parity.max_abs_logprob_diff_topk_tol`), and total variation
(`parity.max_total_variation_tol`). The max-over-vocabulary log-probability
difference is **recorded as a diagnostic but not gated**.

That is a deliberate, measured choice, not a loosened tolerance. `log_softmax`
turns a fixed absolute logit perturbation into a comparable log-probability
gap at *every* token, so on a 262k-token vocabulary the maximum is reported by
the deepest tail. Simulating a pure BF16 quantization floor on a realistic
softcapped Gemma 4 distribution: the worst-offending token had probability
8e-14 (rank 262136 of 262144), only 3 tokens exceeded a 0.05 gap and together
they held 1.95e-13 of the probability mass, while the argmax and the whole
top-10 were unchanged and total variation was 0.005. One BF16 ULP near
Gemma's `|logit| = 30` softcap bound is already 0.0625, and the observed 0.125
is exactly two such ULP — i.e. at the representation floor.

So a max-over-vocabulary threshold measures BF16 tail noise, not whether two
decoding paths agree. Token equality plus top-k plus total variation
constrains everything a decode or a target score can be sensitive to, and
cannot be dominated by a single tail outlier.

Two other candidate fixes were tested and **rejected**. Computing the final
hidden state and LM head in float32 does reduce quantization error, but
`generate()` still runs the head in BF16, so upcasting only one side makes the
two paths *less* equivalent — the goal is to match the model's own pathway, not
to compute a better one. Enabling `torch.use_deterministic_algorithms` does not
help either: both sides already call the same op, and determinism does not make
two *different* GEMM shapes accumulate alike. Matching the shape is the fix.

KV-cache optimization is deliberately **not** implemented yet;
it may be added only after these uncached results stand.

Go/no-go (held-out split): on a clear majority of examples the correct cone
or subcone must beat zero **and** the random/shuffled/sign-reversed/
unrelated controls on target log-probability, survive ≥ 2 neutral prompts,
recover a meaningful fraction of targets by decoding, and be competitive
with raw activation transplantation. One anecdotal exact match is not
success.

## Local (CPU, mock) checks

```bash
python -m pytest tests/test_generative.py tests/test_generative_runner.py -q
```

The runner test executes the full pipeline on the Gemma4-shaped mock —
gates, dictionaries, pursuits, all conditions, records, go/no-go artifacts.

## Remote GPU workflow (Colab CLI)

The official Google Colab CLI (`google-colab-cli`, released June 2026) is
the supported path. Command syntax below was verified against the installed
package (v0.6.0) — its `cli.py` and the `COLAB_SKILL.md` it ships — not
guessed.

> **Platform note (verified 2026-07-29):** the CLI supports **Linux and
> macOS only**. On native Windows it installs but fails at startup
> (`ModuleNotFoundError: No module named 'termios'`). Run it from WSL,
> Linux, or macOS. From a Windows-only machine, the repository's existing
> Colab notebook bootstrap (see `docs/causal_smoke_run.md`) remains the
> working fallback.

### One-time setup

```bash
pip install google-colab-cli    # or: uv tool install google-colab-cli
```

Authentication (ADC, the reliable headless path — the Colab backends need
all four scopes):

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Verify with `colab --auth=adc sessions` (read-only) or `colab whoami`.

### Provision a named GPU runtime

```bash
colab --auth=adc new -s jlens --gpu L4     # T4 | L4 | G4 | H100 | A100
colab --auth=adc status -s jlens
```

Always name the session (`-s jlens`); accelerator availability is
tier-gated, and an unrecognized `--gpu` value silently falls back to A100 —
spell it exactly.

### Install the repository and locked dependencies

```bash
echo "git clone --branch experiment/generative-jlens-validation https://github.com/MechInterpreter/jacobian-lens-gemma.git /content/jacobian-lens-gemma" | colab console -s jlens
colab install -s jlens pyyaml datasets accelerate pytest
echo "import subprocess; subprocess.run(['pip','install','-e','/content/jacobian-lens-gemma'],check=True)" | colab exec -s jlens
```

(For a private repo, mint a fine-grained read-only token and clone with
`https://x-access-token:<TOKEN>@github.com/...` — pass it via the exec
snippet, never commit it.)

### Supply model credentials without committing secrets

```bash
echo "import os; os.environ['HF_TOKEN'] = open('/content/hf_token').read().strip()" > /tmp/set_token.py
colab upload -s jlens ~/.cache/hf_token_file /content/hf_token
colab exec -s jlens -f /tmp/set_token.py
```

Kernel state persists across `colab exec` calls in the same session, so the
environment variable set above is visible to later executions. Nothing
token-shaped enters the repository or the run artifacts.

Persist results beyond the ephemeral VM (`colab drivemount` is interactive;
run it from a human terminal once per session if you want Drive):

```bash
colab drivemount -s jlens        # mounts /content/drive (interactive)
```

### Smoke test (Phase 2 gate)

```bash
colab exec -s jlens -f scripts/colab_smoke.py
```

Prints a JSON report (versions, GPU, config/benchmark validation, mock test
suite) and exits non-zero on failure.

### Validate target tokenization (cheap; do this before any GPU run)

```bash
echo "import subprocess,os; os.chdir('/content/jacobian-lens-gemma'); subprocess.run(['python','scripts/validate_benchmark_targets.py','--config','configs/gemma_generative_validation.yaml'],check=True)" | colab exec -s jlens
```

Downloads the tokenizer only (no weights, no GPU) and applies the same 2–6 token
requirement the run enforces, printing each target's count, ids, and per-token
strings. Exits non-zero if any target violates it, so a bad concept costs
seconds instead of a full run.

### Run the validation experiment

```bash
echo "import subprocess,os; os.chdir('/content/jacobian-lens-gemma'); subprocess.run(['python','scripts/run_generative_validation.py','--config','configs/gemma_generative_validation.yaml','--allow-model-load','--device-map','cuda','--runs-root','/content/drive/MyDrive/jacobian-lens-gemma/runs','--smoke'],check=True)" | colab exec -s jlens
```

Drop `--smoke` for the full dev-split run; add `--split heldout` only after
calibration is frozen. The `--runs-root` must contain the completed pilot
run (`pilot_20260715T200437612150_311fd108c23a/artifacts/lens.pt`) — on
Drive if mounted, otherwise `colab upload` the lens into a local
`runs/` mirror. Runs are timestamped directories; records are appended
(fsynced JSONL) after every condition.

### Broad development calibration

[`configs/gemma_generative_dev_calibration.yaml`](../configs/gemma_generative_dev_calibration.yaml)
is a **separate** config from the smoke/default one above, for a full dev-split
sweep once the smoke run has passed: every dev example, the informative-region
ratio sweep only (`0.01`–`0.25`, no `0.5`/`1.0`/`2.0` stress tail), `natural_scale`
plus all five natural-scale-matched controls, the four ratio-scaled specificity
controls (`GONOGO_CONTROLS`), and decoding disabled (scoring-only — see the
config file's header comment for the exact per-example record-count derivation,
4392 records total for the shipped 8-example dev benchmark).

```bash
echo "import subprocess,os; os.chdir('/content/jacobian-lens-gemma'); subprocess.run(['python','scripts/run_generative_validation.py','--config','configs/gemma_generative_dev_calibration.yaml','--allow-model-load','--device-map','cuda','--runs-root','/content/drive/MyDrive/jacobian-lens-gemma/runs'],check=True)" | colab exec -s jlens
```

Do not add `--smoke`, `--limit-examples`, or `--split heldout` — the whole
point is every dev example, and heldout stays untouched until calibration is
frozen.

### Download results and logs, then stop

```bash
colab download -s jlens /content/drive/MyDrive/jacobian-lens-gemma/runs/<run_id> ./runs/<run_id>
colab log -s jlens -o runs/<run_id>/colab_session.ipynb
colab stop -s jlens
```

`colab stop` releases the billable VM — never leave a session running.
Results live in three places by design (Drive, the local checkout, and the
exported session log); the ephemeral Colab filesystem is never the only
copy. For a fully ephemeral one-shot instead of a named session:

```bash
colab run --gpu L4 scripts/colab_smoke.py
```

(`colab run` = new + exec + stop, propagates the script's exit code.)

## Outputs

Each run directory contains `run_started.json`, `resolved_config.json`,
`artifacts/gates.json`, `artifacts/targets.json` (validated target token ids,
per-token strings, and counts per example), `artifacts/pursuits.json` (active
generator ids / labels / coefficients / explained fractions per example x
layer), `artifacts/records.jsonl` (schema `jlens.generative.record.v1`: run id,
commit, model revision, example, layers/positions, condition, schedule,
requested + measured ratios, vector + receiving norms, generator metadata
(including `scaling_mode`, and for natural-scale-matched conditions
`raw_delta_norm` / `scale_factor` / `reference_natural_cone_norm`), subset
indices and mass thresholds, target token ids **and token strings**, per-token
and total target log-probabilities, deltas vs zero and vs unrelated, KL from
baseline, generated tokens/text/stop reason, seeds), `artifacts/
summary_by_condition.json` (each row tagged with `scaling_mode`),
`artifacts/calibration.json` (dev), `artifacts/gonogo.json` (ratio-scaled
go/no-go), `artifacts/natural_scale_comparison.json` (natural-scale-matched
go/no-go, per (example, layer, schedule) — see above), `summary.md`,
`run_metadata.json` (environment, gates, wall time, `gonogo`,
`natural_scale_gonogo`). No raw activation tensors are stored.

## Interpretation limits

A fresh pursuit on one activation identifies a *local* cone; recurring
structure across prompts is a separate aggregation question. The layer-21
pilot records show J-space reconstructions capture a small fraction of the
activation norm, which is exactly why steering strength is normalized to
the receiving activation and why `raw_activation` is a first-class
comparison, not an afterthought. Automatic subcone clustering is out of
scope until manual subcones demonstrate an advantage (Phase 10).
