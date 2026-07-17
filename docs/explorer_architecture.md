# Explorer architecture

The Gemma 4 Multimodal J-Lens Explorer is a static single-page application
plus a deterministic export pipeline. There is deliberately no backend:
everything the browser shows comes from versioned JSON bundles committed or
dropped into `explorer/public/data/`.

## Pipeline

```
completed runs (immutable)            exporters (CPU, deterministic)          browser
─────────────────────────            ────────────────────────────────        ───────
runs/pilot_*/lens.pt        ──┐
runs/jspace_*/cones,          ├──►  scripts/export_explorer_bundle.py  ──►  data/text_demo.json
  trajectories, eval          │       (jlens/explorer_export.py)
reports/<run>/atom_freqs    ──┘

causal notebook (L4)        ──────►  artifacts/explorer_causal_bundle.json ──► data/measured/causal.json
multimodal notebook (L4)    ──────►  artifacts/multimodal_explorer_bundle.json ──► data/measured/multimodal.json
scripts/make_ui_fixtures.py ──────►  data/fixtures/{causal,multimodal}_fixture.json
```

All producers validate against `schemas/explorer_bundle.schema.json`
(`jlens.explorer.bundle.v1`) and share the bundle constructors in
`jlens/explorer_export.py` (`assemble_bundle`, `make_provenance`,
`merge_bundles`), so the browser only ever parses one shape.

## Determinism and safety invariants

- **Byte-identical re-export:** `created_utc` derives from the source run's
  recorded `written_utc` (never the wall clock), every array is explicitly
  sorted, JSON is rendered with sorted keys and fixed separators.
- **No absolute paths:** `assert_no_absolute_paths` runs on every bundle
  write and merge; Colab `/content/...` provenance is reduced to run-relative
  identifiers.
- **Sources are read-only:** the exporter never writes into `runs/` or
  `reports/`; tests hash the source tree before/after export.
- **Nothing fabricated:** fields the source runs did not persist (per-step
  pursuit coefficients, per-layer J-lens top-k lists for the completed text
  run) are exported as explicitly unavailable, and the UI labels them.

## Frontend

`explorer/` — Vite + React + TypeScript, no runtime dependencies beyond
react/react-dom. One-way data flow: `App.tsx` owns four pieces of selection
state (modality tab, example, position, layer) and derives everything else
from the loaded bundle.

```
main.tsx
└─ App.tsx                 selection state + record lookups
   ├─ lib/loadBundle.ts    fetch, structural validation, client-side merge,
   │                       measured-over-fixture preference
   ├─ Header               product name, run/revision, status badge, modality tabs
   ├─ ExampleBrowser       search + category/format filters, strength chips
   ├─ InputViewer          prompt, token/position chips, image/audio assets
   ├─ LayerRail            explained fraction, output alignment, Jaccard stability
   ├─ PredictionPanel      model vs J-lens (rank, overlap; honest gaps)
   ├─ ConePanel            coefficient bars, output/nuisance markers, residual
   ├─ PursuitPlayer        play/step/slider over recorded residual history
   ├─ TrajectoryView       per-layer columns, retained/entered/exited atoms
   ├─ CausalPanel          measured multipliers only, targeted vs matched control
   └─ ProvenancePanel      run ids, fingerprints, schema version, limitations
```

### Bundle loading policy

`loadDefaultBundles()` fetches `data/text_demo.json` (required), then for each
of causal and multimodal data tries `data/measured/<name>.json` first and only
falls back to `data/fixtures/<name>_fixture.json` when no measured bundle
exists. A fixture that does not declare `data_status: synthetic_fixture` is
rejected. Bundle-level status degrades to the weakest merged input, and every
example and causal record also carries its own status; the header badge
follows the *currently viewed example*, panels badge per record.

### Accessibility and quality bars

- All interactive elements are native buttons/inputs/selects (keyboard
  reachable); selected state is exposed via `aria-pressed`.
- Panels are labelled `role`d regions so tests and screen readers can target
  them.
- Light/dark via `prefers-color-scheme`; no color is the sole carrier of
  meaning (markers carry text).
- Missing optional fields render as explicit "not recorded/not persisted"
  text, never as fabricated values or blank crashes; a malformed bundle
  produces a labelled error state.

## Testing

- Python: `tests/test_explorer_export.py` (schema validity, determinism,
  stable IDs, path stripping, malformed/missing artifacts, immutability,
  merge semantics — plus a byte-identity check against the committed demo
  bundle when the real run directory is present), `tests/test_interventions.py`,
  `tests/test_causal_multimodal_configs.py`, `tests/test_new_notebooks_light.py`.
- Frontend: Vitest + React Testing Library in `explorer/src/test/` (bundle
  validation/merge, example/position/layer selection, cone/pursuit/trajectory
  rendering, causal multiplier selection, badges, empty states, image/audio
  rendering, malformed-bundle error state).
- A Playwright browser test was considered and deliberately skipped for the
  MVP (heavy browser download, low marginal coverage over the RTL suite); the
  demo script in [demo_script.md](demo_script.md) doubles as the manual smoke
  checklist.
