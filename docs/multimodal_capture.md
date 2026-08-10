# Multimodal capture — guide

Notebook: [`notebooks/archive/legacy_prototypes/gemma_4_e4b_multimodal_jlens_capture.ipynb`](../notebooks/archive/legacy_prototypes/gemma_4_e4b_multimodal_jlens_capture.ipynb)
· Config: [`configs/gemma_multimodal_jlens_capture.yaml`](../configs/gemma_multimodal_jlens_capture.yaml)

## What it captures

The explorer's first **real** image-conditioned and audio-conditioned records:
for one user-supplied image and one audio clip (with text prompts), at layer
**38**, position **−1** (the final text-generation position):

- processor/model input metadata and the resolved processor interface;
- sequence structure and the modality token range when the processor exposes
  a contiguous one;
- the model's top-10 next-token prediction (softcapped pathway, with probs);
- the J-lens top-10 (pre-softcap `W_U norm(J h)`, paper convention) plus rank
  of the model top-1 and top-10 overlap;
- the k=10 gradient-pursuit cone and residual history — same
  `PursuitSettings` as the completed text run, so cones are comparable;
- an explorer bundle that loads directly in the frontend.

**Estimated L4 runtime:** ≈15–25 min (model load dominates; each capture is a
single forward pass plus one dictionary build).

## Interpretation boundary (recorded in every artifact)

The lens is **frozen and text-fitted** (fingerprint-verified; never refitted).
Applying it to multimodal-conditioned decoder states is an exploratory probe
of those states. It does not establish that text and multimodal inputs share
concepts, and no pixel- or audio-span-level attribution is claimed — only
processor-exposed token ranges are recorded.

## Robust interface handling

The notebook loads `AutoProcessor` at the pinned revision and **inspects** it:
component classes, `__call__` parameters, image/audio support, special token
ids, chat-template availability — printed and saved. If a requested modality
is unsupported by the resolved processor, the notebook fails with a clear
error; it never substitutes fake data. Input building tries the direct
processor call first and falls back to the chat template, recording which
route was used.

## Colab steps

1. Upload one image (e.g. JPEG/PNG) and one short audio clip (16-bit PCM WAV
   recommended) to Drive or the VM.
2. Open the notebook, GPU runtime = L4; set `IMAGE_PATH`, `AUDIO_PATH` (and
   optionally the prompts) in the **INPUTS** cell.
3. Run all cells. Bootstrap clones branch `multimodal-jlens-explorer`; Drive
   must contain the completed pilot run (the frozen lens).
4. After layer 38 succeeds, optionally set `capture.layers: [35, 38]` in the
   config and rerun for the layer-35 records.

## Artifacts

`run_started.json`, `resolved_config.json`, `artifacts/image_record.json`,
`artifacts/audio_record.json`, `artifacts/multimodal_explorer_bundle.json`,
`artifacts/assets/<your files>`, `summary.md`, `run_metadata.json`.

## Feeding the explorer

```bash
cp <run>/artifacts/multimodal_explorer_bundle.json explorer/public/data/measured/multimodal.json
mkdir -p explorer/public/data/measured/assets
cp <run>/artifacts/assets/* explorer/public/data/measured/assets/
```

The Image and Audio tabs then show the measured examples (badge: **Measured**)
and the multimodal fixture stops loading. Keep assets you are comfortable
publishing — they ship with any deployed copy of the site.
