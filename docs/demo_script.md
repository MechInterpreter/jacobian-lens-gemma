# Demo script (60–90 seconds)

Setup: `cd explorer && npm run dev` (or serve `dist/`). Land on the Text tab.

1. **Select a prompt (0:00–0:10).** In the example browser, click
   *"The capital city of Australia is"* (factual, tagged **strong**). Point at
   the header: model revision, source run ID, and the **Measured** badge —
   everything on screen comes from a completed, fingerprinted run.

2. **Inspect token and layer (0:10–0:20).** The input viewer shows the
   recorded positions; keep **−1** (` is`). On the layer rail, note the
   explained-fraction bars and the output-alignment dots; layer 38's dot is
   lit — the model's answer token is inside the k=10 cone.

3. **J-lens predictions (0:20–0:30).** The prediction panel: the model's
   measured top-1 is ` Canberra`, and the J-lens rank of that token at layer
   38 is **0** — the linear readout already agrees three layers before the
   output. (The panel says exactly which fields the run persisted.)

4. **Replay the pursuit (0:30–0:45).** In the playback panel press **play**:
   atoms enter one by one — ` Canberra` first — while the residual-norm curve
   drops. Point at the honesty label: per-step coefficients weren't recorded,
   so the player shows exactly what was.

5. **Compare layers (0:45–0:55).** The trajectory panel: from L35 to L38 most
   atoms persist (highlighted), Jaccard ≈ 0.5+, and the output token stays in
   the cone; from L14 to L21 nothing survives — the documented layer-21
   anomaly, visible as history.

6. **Steering result (0:55–1:10).** Open the causal panel *(after the causal
   smoke run has been merged)*: select the output atom at layer 38, multiplier
   **−1** — target logit drop, rank shift, and the completion change, side by
   side with the norm-matched random control that does nothing. Only measured
   multipliers are clickable.

7. **Switch to Image (1:10–1:20).** Click **Image**: the captured image, its
   token range in the sequence, the model's answer, and the cone the
   text-fitted lens reads out of an image-conditioned state — badged
   exploratory/measured accordingly.

8. **Switch to Audio (1:20–1:30).** Click **Audio**: play the clip inline,
   same panels for the audio-conditioned state. Close on the provenance panel:
   lens fingerprint, schema version, and the limitations list.

Fallback when causal/multimodal runs haven't been merged yet: steps 6–8 show
the **Synthetic UI fixture** badge or the honest *No causal data available*
state — say so and move on; the text panels (steps 1–5) are fully measured.
