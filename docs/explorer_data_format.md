# Explorer bundle format (`jlens.explorer.bundle.v1`)

Authoritative schema: [`schemas/explorer_bundle.schema.json`](../schemas/explorer_bundle.schema.json)
(JSON Schema 2020-12). Semantic version in `provenance.schema_version`
(currently `1.0.0`); the `v1` line only adds optional fields.

A bundle is one self-contained JSON object with seven sections:

| Section | One record per | Produced by |
|---|---|---|
| `provenance` | bundle | every producer |
| `examples[]` | saved example (prompt or prompt+asset) | exporter / capture notebook |
| `layer_records[]` | (example, layer, position) | exporter / capture notebook |
| `cones[]` | (example, layer, position) k=10 pursuit result | exporter / capture notebook |
| `pursuit_traces[]` | same key — step-indexed playback data | exporter / capture notebook |
| `trajectories[]` | (example, position, layer_from→layer_to) | exporter |
| `causal_records[]` | intervention condition | causal notebook |

## Identity and status conventions

- `example_id = "<modality>:<slug>:<prompt_hash>"` — stable across exports;
  causal and multimodal bundles attach to text examples via the same IDs.
- `prompt_hash` — `sha256(prompt_text)[:16]`, the repo-wide
  `jlens.metadata.prompt_hashes` convention. For chat examples the recorded
  hash covers the *rendered chat template*; the bundle stores the pre-template
  user message with `prompt_text_is_pre_template: true`.
- `condition_id` — deterministic `cond_<sha256[:16]>` over the condition's
  defining coordinates (`jlens.interventions.condition_id`).
- `data_status` ∈ `measured | imported | synthetic_fixture` at bundle level
  AND on every example / layer record / cone / trace / trajectory /
  causal record. Merging keeps per-record statuses and degrades the bundle
  level to the weakest input, so a fixture can never masquerade as measured.
- Positions use Python negative indexing into the tokenized sequence
  (`-1` = last token). `causal_records[].provenance.resolved_position`
  records the absolute index actually edited.

## Honest-gap fields

The completed text run did not persist everything the schema can carry.
Nullable fields mean "not recorded", never zero:

- `layer_records[].jlens_topk` / `model_topk` — `null` for the completed text
  run (rank + overlap are recorded); populated by the capture notebooks.
- `pursuit_traces[].per_step_coefficients_available: false` with
  `steps[].coefficients_after: null` — only selection order, residual-norm
  history, and final coefficients were recorded.
- `steps[].final_coefficient_zero` refers to the FINAL solution (per-step
  zeroing was not recorded).
- `input.text.tokenization_available: false` — only per-position token
  metadata exists; the UI shows position chips instead of a full token strip.
- `modality_token_range` — present only when the processor exposes a
  contiguous run of modality tokens; `null` otherwise. No pixel or audio-span
  attribution is ever implied.

## Merging

`jlens.explorer_export.merge_bundles(base, extra)` (mirrored client-side in
`explorer/src/lib/loadBundle.ts`): sections are deduplicated by their stable
identity with *extra* winning — that is the documented replacement workflow
(measured records replace fixtures under the same IDs). Provenance appends the
merged bundle's provenance to `merged_bundles`, unions
`modalities_present`/`source_run_ids`, and degrades `data_status`.

## Determinism contract

`canonical_json` renders with sorted keys, compact separators,
`ensure_ascii=False`, and a trailing newline; `created_utc` comes from source
metadata. Re-running the exporter (or `scripts/make_ui_fixtures.py`) on the
same inputs is byte-identical — enforced by
`tests/test_explorer_export.py::test_real_run_export_matches_committed_demo_bundle`.

## Validation

```bash
python - <<'PY'
import json, jsonschema
schema = json.load(open("schemas/explorer_bundle.schema.json", encoding="utf-8"))
bundle = json.load(open("explorer/public/data/text_demo.json", encoding="utf-8"))
jsonschema.validate(instance=bundle, schema=schema)
print("valid")
PY
```

The exporter validates on every write (`--no-validate` to skip); notebooks
validate when `jsonschema` is installed; the frontend re-checks the
UI-critical structure on load and shows a labelled error state for malformed
files.
