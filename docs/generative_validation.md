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
3. Build the condition vector (13 conditions, below), rescale it so
   ‖δ‖ = ratio · ‖h_recv‖ where `h_recv` is the unsteered residual at the
   injection site of the neutral prompt (requested **and** measured ratios
   are recorded).
4. Inject at the neutral prompt's final token position under a schedule:
   `prompt_only`, `constant` (every generated position), or `decaying`
   (weight `decay^offset`).
5. Score the target phrase teacher-forced (`Σ_t log P(y_t | y_<t, q)`), and
   (for the configured subset) decode greedily **uncached** — every step is
   a full forward pass, so injection semantics are exact.

Conditions: `none`, `zero` (parity), `full_cone`, `mass_subcone`
(smallest subset covering 60/70/80% of positive coefficient mass),
`manual_subcone` (manifest-provided indices), `unrelated_cone` (another
example's cone), `random_matched_norm`, `shuffled` (coordinate
permutation), `sign_reversed`, `wrong_layer`, `wrong_position`,
`raw_activation`, `activation_diff` (source minus matched control prompt).

Hard gates before any steering (any failure aborts): architecture
verification, lens SHA-256 fingerprint, zero-vector logit parity within
`parity.max_abs_logit_diff_tol`, and manual-uncached vs `generate()` greedy
equivalence — both the decoded tokens and the first-step log-probabilities
(within the same tolerance), so the gate cannot pass on a coincidental argmax
tie. The reference `generate()` call explicitly sets `do_sample=False`,
`num_beams=1`, `use_cache=False` and matching EOS handling, because the
Gemma 4 E4B-it checkpoint's stored generation config defaults to
`do_sample=True, top_k=64, top_p=0.95`.

All logits in these gates — and in every recorded measurement — are read
through the model's own head (`LensModel.logits_from_ids`). Note that
HuggingFace text models apply the final norm *before* returning
`last_hidden_state`, so `unembed(forward(ids).last_hidden_state)` would apply
the final norm twice; `unembed` is for residual-stream activations captured
from block hooks, not for `last_hidden_state`.

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
`artifacts/gates.json`, `artifacts/pursuits.json` (active generator ids /
labels / coefficients / explained fractions per example x layer),
`artifacts/records.jsonl` (schema `jlens.generative.record.v1`: run id,
commit, model revision, example, layers/positions, condition, schedule,
requested + measured ratios, vector + receiving norms, generator metadata,
subset indices and mass thresholds, per-token and total target
log-probabilities, deltas vs zero and vs unrelated, KL from baseline,
generated tokens/text/stop reason, seeds), `artifacts/
summary_by_condition.json`, `artifacts/calibration.json` (dev),
`artifacts/gonogo.json`, `summary.md`, `run_metadata.json` (environment,
gates, wall time). No raw activation tensors are stored.

## Interpretation limits

A fresh pursuit on one activation identifies a *local* cone; recurring
structure across prompts is a separate aggregation question. The layer-21
pilot records show J-space reconstructions capture a small fraction of the
activation norm, which is exactly why steering strength is normalized to
the receiving activation and why `raw_activation` is a first-class
comparison, not an afterthought. Automatic subcone clustering is out of
scope until manual subcones demonstrate an advantage (Phase 10).
