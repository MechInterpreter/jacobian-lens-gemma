# jlens-gemma — Jacobian lens on Gemma 4 E4B

Research fork adapting Anthropic's official **Jacobian Lens** reference
implementation to [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it).

- **Paper (authoritative method source):** [Verbalizable Representations Form a
  Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
  (Anthropic, 2026).
- **Upstream implementation:** [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens),
  cloned with full history; git remote `upstream`; starting commit
  **`581d398613e5602a5af361e1c34d3a92ea82ba8e`** ("Initial release"). Upstream is a
  reference implementation, is not maintained, and is treated here as research
  scaffolding that is audited and tested rather than assumed correct.
- **License:** Apache-2.0 throughout ([LICENSE](LICENSE)). Upstream code
  retains Anthropic PBC copyright notices; files new in this fork are marked in
  their headers.

## Current status

The **smoke** stage — the second of three planned stages (micro-smoke →
smoke → pilot) — has completed successfully on real `google/gemma-4-E4B-it`
weights (revision `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd`): the Jacobian
lens was fitted on 8 plain-text prompts at 5 source layers in 361.5 s on a
Colab GPU, producing finite Jacobians, and was evaluated against a logit-lens
baseline and three negative controls. Full results, methodology, and an
explicit separation of measured findings from speculative interpretation are
in **[docs/smoke_report.md](docs/smoke_report.md)**; the complete run
artifacts are preserved under
[`runs/smoke_20260715T172315460316_fb2eefcd91cd/`](runs/smoke_20260715T172315460316_fb2eefcd91cd/).
See [docs/research_log.md](docs/research_log.md) for the engineering
milestones that led to this run. The next planned stage is **pilot**
(`configs/gemma_text_pilot.yaml`).

## What is inherited vs new

**Inherited unchanged** (upstream `jlens` core — no modifications):
`jlens/protocol.py`, `jlens/hf.py`, `jlens/hooks.py`, `jlens/fitting.py`
(estimator, checkpointing, merging), `jlens/lens.py` (save/load/apply,
logit-lens baseline), `jlens/vis.py`, `jlens/examples.py`, all 32 upstream
tests, and the upstream walkthrough/README (below).

**New in this fork:**

| Path | Purpose |
|---|---|
| `jlens/gemma4.py` | Revision pinning, gated loading, explicit BOS control, pre-softcap + softcapped dual readout, architecture verification, fit-cost probe |
| `jlens/controls.py` | Negative controls: row-permuted fitted J (primary), scale-matched random, wrong-layer application; overlap/rank metrics |
| `jlens/metadata.py` | Config validation, config fingerprint, environment manifest, atomic metadata writes |
| `configs/gemma_text_{microsmoke,smoke,pilot}.yaml` | The three experiment stages (see below) |
| `configs/prompts/` | Plain-text fitting corpus (smoke) + separate plain/chat evaluation prompts |
| `scripts/fit_gemma.py`, `scripts/apply_gemma.py` | CLI: fit and evaluate with full metadata |
| `notebooks/gemma_4_e4b_text_jlens.ipynb` | End-to-end pilot notebook (sequential from a fresh runtime; model load behind `JLENS_ALLOW_GEMMA=1`) |
| `tests/test_gemma4_adapter.py`, `test_controls.py`, `test_finite_difference.py`, `test_metadata.py`, `test_scripts.py`, `mock_gemma4.py` | 47 new CPU-only tests (no network, no real model) |

**Validated so far:** the full local CPU suite (79 tests) passes: layout
auto-detection on a Gemma4-shaped mock, Jacobian orientation pinned
analytically and by finite differences, checkpoint/resume, controls, config
validation, and the complete fit→apply pipeline on the mock. **No real-model
results exist yet** — no scientific claims are made. Planned next: microsmoke
(1–2 prompts, 1 layer) → smoke (8 prompts, 5 layers) → pilot (100 WikiText
sequences, 7 layers) on a Colab A100. Anything beyond the text-only pilot
(k-cones, sparse decompositions, steering, multimodal) is future/speculative —
deliberately not implemented; `jlens/gemma4.py` and the configs are the
intended extension points.

## Gemma 4 E4B specifics

`Gemma4ForConditionalGeneration` (multimodal wrapper; the vision/audio towers
are loaded but never executed for text-only work): text decoder
`model.language_model` with **42 blocks, d_model 2560, vocab 262144**, dense
(`enable_moe_block: false`), tied unembedding, `final_logit_softcapping=30.0`,
per-layer embeddings (PLE) folded inside each block, sliding-window 512 with
every-6th-layer full attention, KV sharing across the last 18 layers.

- **Source site** `h_l`: output of `model.language_model.layers[l]` (after
  attention + MLP + PLE re-injection + `layer_scalar`) — the input to block `l+1`.
- **Target site** `h_final`: same site at block 41 (pre-final-norm residual).
- **Convention:** `J[i, j] = ∂h_final[i]/∂h_l[j]` (rows = target dims);
  transport is `J @ h`. Readout reports **pre-softcap** logits (paper
  convention) and **softcapped** logits (Gemma's real output pathway);
  the cap is monotonic so rankings are identical.
- The model repo revision is resolved to an immutable commit SHA at run time
  and recorded in every artifact; nothing is fitted against a mutable ref.

## Install & test

```bash
pip install -e .            # torch, transformers>=5.5, huggingface_hub, numpy
pip install -e .[gemma]     # + pyyaml, datasets, accelerate
pytest tests -q             # 79 CPU-only tests; no network, no model download
```

Model access: `google/gemma-4-E4B-it` (~16 GB bf16) is ungated on the HF Hub
but subject to the Gemma license; downloads happen only under the explicit
flag below.

## Running the experiment stages

Real-model work is intended for a **Colab A100 (40 GB)**; the model does not
fit a 12 GB local GPU, and CPU fitting is impractically slow. Start every new
environment with the memory probe at the configured small `dim_batch` (4–8)
and scale up only after reading its peak-memory figure.

```bash
# stage 1: micro-smoke — 1–2 prompts, one middle layer (21), seq ≤ 48, dim_batch 4
python scripts/fit_gemma.py --config configs/gemma_text_microsmoke.yaml \
    --allow-model-load --device-map cuda

# stage 2: smoke — 8 plain-text prompts, layers [7,14,21,28,35]
python scripts/fit_gemma.py --config configs/gemma_text_smoke.yaml \
    --allow-model-load --device-map cuda
python scripts/apply_gemma.py --config configs/gemma_text_smoke.yaml \
    --lens artifacts/smoke/lens.pt --allow-model-load --device-map cuda

# stage 3: pilot — 100 WikiText-103 sequences, layers [3,7,14,21,28,35,38]
python scripts/fit_gemma.py --config configs/gemma_text_pilot.yaml \
    --allow-model-load --device-map cuda
```

Or run `notebooks/gemma_4_e4b_text_jlens.ipynb` top-to-bottom
(`JLENS_MODE=microsmoke|smoke|pilot`, `JLENS_ALLOW_GEMMA=1`,
`JLENS_DEVICE_MAP=cuda`).

Artifacts land in `artifacts/<mode>/` (git-ignored): `lens.pt`, `ckpt.pt`,
`fit_metadata.json`, `apply_results.json`, `apply_summary.md`, optional slice
HTML. Every artifact embeds the config fingerprint, model revision, upstream
and local commits, prompt hashes, seeds, versions, runtime, and peak memory.
Smoke-stage corpora are far below the paper's ~1000 sequences: smoke-mode
readouts verify plumbing, not lens quality.

**Known limitations:** bf16 backward noise in the fp32-accumulated `J`;
sliding-window + KV-shared attention differ architecturally from the paper's
models; the `-it` chat distribution differs from the plain-text fitting corpus
(chat prompts are evaluated separately, never fitted on); no multimodal
support; the fork is text-only by design in this pass.

---

# Upstream README — jlens — Jacobian lens

> **Reference implementation.** Not maintained and not accepting contributions.

Companion code for [**Verbalizable Representations Form a Global Workspace in
Language Models**](https://transformer-circuits.pub/2026/workspace/index.html).

The Jacobian lens reads out what an internal activation is disposed to make the
model say. It linearly transports a residual-stream vector at any layer and
position into the final-layer basis, then decodes it with the model's own
unembedding into a ranked list of vocabulary tokens.

The transport is the average input–output Jacobian over a text corpus:

```
lens_l(h) = unembed( J_l @ h ), J_l = E[∂h_final / ∂h_l]
```

The expectation is over prompts, source positions, and all current-and-future
target positions in a generic web-text corpus; the precise estimator
(cotangents summed over target positions, then averaged over source positions)
is documented in the [`jlens.fitting`](jlens/fitting.py) module docstring.

This repo fits the lens on open-weights decoder transformers, applies it, and
renders the interactive layer × position view shown below. Examples use Qwen;
other HuggingFace decoders adapt cleanly.

![Slice visualisation: ASCII-face example](assets/slice_vis.png)

*The ASCII-face example: selecting the `^` (nose) position shows the lens
reading out "nose" at mid layers, although the word never appears in the
prompt.*

## Install

```bash
pip install -e .
```

## Usage

### Apply

To apply a pre-fitted lens:

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2])
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])
```

### Fit

To fit a lens on your own model:

```python
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like
corpus. Quality saturates quickly (§9.3); ~100 prompts is usable. This is a
reference implementation and is not optimized; fitting time is dominated by
the model's own backward pass. Parallelize by running `fit()` on disjoint
slices and combining with `JacobianLens.merge()`.

## Walkthrough

[`walkthrough.ipynb`](walkthrough.ipynb) is the end-to-end notebook: load a
model, load (or fit) a lens, apply it at a few layers, and render a slice page
like the one above.

Reading a slice page:

- Each cell shows the lens top-1 word at that (position, layer); the
  superscript is its rank over the full vocabulary.
- Click a cell to select a (position, layer) and pin its top-1 token; pinned
  tokens get rank-tracking charts and a rank heatmap.
- The bottom row (`L = n_layers − 1`) is the model's actual output.

## License and data

Code is released under the Apache License 2.0 — see [LICENSE](LICENSE).

The replication and lens-eval prompt sets in [`data/`](data/) are synthetic,
authored by Anthropic, and released under the same Apache License 2.0 as the
code. See the READMEs in [`data/experiments/`](data/experiments/) and
[`data/evaluations/`](data/evaluations/) for what each set contains.

The slice-vis pages use [d3](https://github.com/d3/d3) (ISC license), loaded
from the jsDelivr CDN with subresource integrity or inlined into
self-contained pages.

No model weights or text corpora are bundled; models and datasets downloaded
at run time are subject to their own licenses.
