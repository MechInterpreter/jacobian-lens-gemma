# Causal steering smoke run — guide

Notebook: [`notebooks/gemma_4_e4b_jspace_causal_smoke.ipynb`](../notebooks/gemma_4_e4b_jspace_causal_smoke.ipynb)
· Config: [`configs/gemma_jspace_causal_smoke.yaml`](../configs/gemma_jspace_causal_smoke.yaml)
· Manifest: [`configs/causal_smoke_examples.json`](../configs/causal_smoke_examples.json)
· Backend: [`jlens/interventions.py`](../jlens/interventions.py)

## What it measures

For four manifest-pinned examples (strong factual, non-output semantic, chat,
weak/incoherent control — each chosen from measured jspace records, reasons in
the manifest), at layers **35 and 38**, position **−1**:

```
h_intervened[row, pos] = h_original[row, pos] + multiplier · delta
```

applied at the exact `block_output` site the lens reads, with float32
arithmetic cast back to the block dtype. Targeted delta families, all derived
from the completed run's k=10 cone records under the audited pursuit
convention (coefficients over **raw** atoms `W_U J_l`):

1. `output_atom_contribution` — `c_v · d_v` for the model's top-1 token's atom;
2. `top_non_output_atom_contribution` — highest-coefficient non-output atom;
3. `full_cone_reconstruction` — the recorded reconstruction `Σ c_v d_v`;
4. `isotropic_random_direction` — control: deterministic CPU-seeded Gaussian,
   rescaled to **exactly** the matched targeted delta's norm, one per targeted
   condition at each nonzero multiplier.

Multipliers: **−1, 0, +1** (±0.5 are schema-supported and off for the first
run; the norm-preserving variant likewise). Recorded per condition: target
logit/rank/prob before+after, top-1/top-10 before+after, KL(after‖before),
‖Δ‖/‖h‖, and greedy 8-token completions before/after (cache-free decoding so
the intervention applies identically at every step).

**Condition count:** 4 examples × 2 layers × 3 families × 3 multipliers = 72
targeted, plus 48 matched controls (nonzero multipliers) ≈ **120 conditions**.
**Estimated L4 runtime:** ≈30–45 min including the ~10 min model load
(completions dominate; disable `intervention.generate_completions` to halve it).

## Safety gates (all enforced in code)

- Frozen-lens fingerprint and jspace-run config fingerprint verified before
  anything runs; mismatch aborts.
- **Baseline-parity gate** before any intervention: unhooked forward vs
  multiplier-0 hook vs identical-copy writeback must agree within
  `parity.max_abs_logit_diff_tol` (0.05 pre-softmax on the softcapped readout)
  with identical top-1 — recorded in `artifacts/baseline_parity.json`; any
  violation aborts the run.
- Deterministic condition IDs; checkpoint after **every** condition
  (JSONL append + fsync); resume skips completed IDs; a run directory with a
  final `run_metadata.json` refuses resumption.
- Model parameters frozen; positions validated at forward time; nonfinite
  deltas/outputs raise; hooks are removed on every exit path.

## Colab steps

1. Open the notebook, GPU runtime = L4.
2. Add the `GITHUB_TOKEN` secret (read access to this repo) if not already set.
3. Run all cells. Bootstrap clones branch `multimodal-jlens-explorer`; Drive
   must contain the completed pilot and jspace runs under
   `MyDrive/jacobian-lens-gemma/runs/`.
4. On interruption: rerun with
   `os.environ["CAUSAL_RESUME_RUN_DIR"] = "<the run dir>"` set in the gates
   cell — completed conditions are skipped.

## Artifacts (in the timestamped Drive run directory)

`run_started.json`, `resolved_config.json`, `selected_examples.json`,
`condition_plan.json`, `artifacts/baseline_parity.json`,
`artifacts/intervention_records.jsonl` (+ `.csv`),
`artifacts/control_matching.json`, `artifacts/explorer_causal_bundle.json`,
`artifacts/analysis_summary.json`, `summary.md`, `run_metadata.json`.

## Feeding the explorer

```bash
cp <run>/artifacts/explorer_causal_bundle.json explorer/public/data/measured/causal.json
```

`data/measured/` is git-ignored; the frontend automatically prefers it and
stops offering the synthetic causal fixture. The causal panel then shows
**Measured intervention** badges, measured multipliers only, and targeted vs
matched-control comparisons.

## Interpretation limits

Four examples at two layers is a smoke run: it establishes that the
intervention machinery works end-to-end and what these specific conditions
measured — not that cone atoms are concepts, not effect sizes in general, and
nothing between the measured multipliers. Cross-prompt and nuisance-direction
controls are schema-ready but deferred to a later, larger study.
