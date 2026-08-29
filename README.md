# Jacobian Lens for Multimodal Working Memory

This repository adapts Anthropic's Jacobian Lens method to Gemma 4 E4B and tests whether a shared internal concept can be causally exchanged across text, image, and spoken-caption inputs.

## Main result

We fit one pooled multimodal J-lens on balanced text, image, and spoken-caption examples. We then exchanged the model's bird and cat coordinates with an exact alpha 1 intervention across the contiguous L16-L40 band. The model generated its answer freely. No answer token was appended, no candidate list selected the result, and no teacher-forced candidate token was scored.

On a fresh held-out population, the bird-to-cat exchange changed the model's downstream leg-count answer from two to four in:

- 14 of 16 text trials
- 16 of 16 image trials
- 15 of 16 spoken-caption audio trials

The zero, random-coordinate, and unrelated-concept controls produced one target answer across 144 trials.

A second fresh confirmation tested a different downstream property. The same bird-to-cat identity exchange changed the model's answer from bird to mammal for biological class in:

- 10 of 18 image trials
- 10 of 18 spoken-caption audio trials
- 3 of 18 text trials, reported as a prespecified secondary result

All five negative controls produced zero target answers in every modality. This supports a narrow causal claim: in this model and direction, the exchanged identity coordinate can drive more than one downstream fact, with the strongest generalization appearing in image and spoken-caption inputs.

## What the experiment does

1. Present exactly one modality per trial: text, an image, or native spoken-caption audio.
2. Read the model's hidden state at each layer in a contiguous band.
3. Use the fitted Jacobian to recover two concept coordinates.
4. Replace the bird coordinate with the cat coordinate while preserving the two-dimensional concept subspace.
5. Let the rest of the network run normally and score the unrestricted generated answer.

The implementation records checksums, configuration fingerprints, data exclusions, hook integrity, activation norms, and one atomic result unit per trial. Interrupted Colab runs resume from checksum-valid units instead of repeating completed work.

## Reproduce the active workflow

| Notebook | Purpose |
| --- | --- |
| `notebooks/multimodal_jspace_coordinate_swap_mock_colab.ipynb` | Small synthetic check of the coordinate-exchange implementation and controls. |
| `notebooks/multimodal_jspace_workspace_replication_colab.ipynb` | Text-only paper-style replication and unrestricted-generation diagnostics. |
| `notebooks/multimodal_jspace_matched_jlens_colab.ipynb` | Canonical multimodal lens fitting, development, and fresh confirmation workflow. |

The notebooks are intentionally output-free. Real model artifacts and population manifests live outside Git because they are large and may contain local paths. Reports are checksum-pinned by the notebooks that produced them.

## Repository layout

- `jlens/`: lens fitting, prompt construction, coordinate exchange, controls, resumable storage, and report logic
- `notebooks/`: the three supported Colab entry points
- `scripts/`: notebook builders and small analysis utilities
- `tests/`: unit, integration, notebook-layout, and real-path contract tests
- `docs/`: protocol and implementation notes
- `explorer/`: local visualization interface for J-space runs

## Setup

```bash
python -m pip install -e .
python -m pytest -q
```

The real Gemma 4 E4B experiments require access to the pinned model revision and enough accelerator memory for the selected dtype and fitting stage. The mock notebook and most tests run without model weights.

## Scope and limitations

- The confirmed results use one model, Gemma 4 E4B.
- The strongest causal results use bird-to-cat, not the reverse direction.
- Audio means spoken captions describing an image, not recordings of animal sounds.
- The second property generalized strongly in image and audio but weakly in text.
- Exact coordinate exchange did not generalize to every concept pair or property tested.
- The evidence supports specific causal transfers, not a universal multimodal workspace claim.

## Attribution

This project builds on Anthropic's open-source [Jacobian Lens](https://github.com/anthropics/jacobian-lens) work. Gemma is provided by Google. The repository is licensed under Apache-2.0; see `LICENSE` and `NOTICE`.
