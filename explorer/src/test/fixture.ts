import type { ExplorerBundle } from "../types";
import type { LoadedData } from "../lib/loadBundle";

/** Small but complete in-memory bundle for component tests: one text example
 * (two positions, two layers), one image example, one audio example, and
 * causal records with a matched control. */
export function testBundle(): ExplorerBundle {
  const textId = "text:factual-canberra:26682621dffc6240";
  const imageId = "image_text:fixture-card:aaaabbbbccccdddd";
  const audioId = "audio_text:fixture-sweep:eeeeffff00001111";

  const coneAtoms = (outputId: number) => [
    {
      token_id: outputId,
      label: " Canberra",
      coefficient: 20,
      is_output_token: true,
      is_effective: true,
      coefficient_share: 0.8,
      nuisance: null,
    },
    {
      token_id: 7,
      label: " capital",
      coefficient: 5,
      is_output_token: false,
      is_effective: true,
      coefficient_share: 0.2,
      nuisance: { total_selections: 40, n_distinct_prompts: 12, high_frequency: true },
    },
  ];

  const cone = (exampleId: string, layer: number, position: number) => ({
    example_id: exampleId,
    layer,
    position,
    requested_k: 10,
    n_selected: 2,
    data_status: (exampleId === textId ? "measured" : "synthetic_fixture") as
      | "measured"
      | "synthetic_fixture",
    selected_atoms: coneAtoms(42),
    coefficient_sum: 25,
    top_coefficient: 20,
    concentration: { herfindahl: 0.68, top1_share: 0.8, n_nonzero: 2 },
    reconstruction: {
      target_norm: 100,
      residual_norm: 90,
      relative_residual: 0.9,
      explained_fraction: 0.19,
    },
    cone_signature_digest: "sha256:0011",
    source_provenance: { run_id: "run_x", artifact: "artifacts/cones/x.json" },
  });

  const trace = (exampleId: string, layer: number, position: number) => ({
    example_id: exampleId,
    layer,
    position,
    requested_k: 10,
    n_iterations: 2,
    stop_reason: "max_atoms",
    data_status: "measured" as const,
    per_step_coefficients_available: false,
    initial_residual_norm: 100,
    target_norm: 100,
    steps: [
      {
        step: 1,
        added_token_id: 42,
        added_label: " Canberra",
        support_after: [42],
        residual_norm: 95,
        relative_residual: 0.95,
        explained_fraction: 0.0975,
        coefficients_after: null,
        final_coefficient_zero: false,
      },
      {
        step: 2,
        added_token_id: 7,
        added_label: " capital",
        support_after: [42, 7],
        residual_norm: 90,
        relative_residual: 0.9,
        explained_fraction: 0.19,
        coefficients_after: null,
        final_coefficient_zero: false,
      },
    ],
  });

  const layerRecord = (
    exampleId: string,
    layer: number,
    position: number,
    rank: number,
  ) => ({
    example_id: exampleId,
    layer,
    position,
    source_site: "block_output" as const,
    data_status: "measured" as const,
    input_token_id: 563,
    input_token: " is",
    model_topk: null,
    jlens_topk: null,
    rank_of_model_top1: rank,
    topk_overlap_with_model: 0.2,
    eval_metadata: { top_k: 10 },
    target_activation_norm: 100,
    residual_norm: 90,
    relative_residual: 0.9,
    explained_fraction: 0.19,
  });

  const causal = (
    conditionId: string,
    multiplier: number,
    controlFamily: string | null,
    matched: string | null,
  ) => ({
    condition_id: conditionId,
    example_id: textId,
    layer: 38,
    position: -1,
    target_kind: (controlFamily
      ? "isotropic_random_direction"
      : "output_atom_contribution") as "output_atom_contribution",
    atom_token_id: controlFamily ? null : 42,
    atom_label: controlFamily ? null : " Canberra",
    multiplier,
    status: "measured" as const,
    norm_preserving: false,
    delta_norm: 20,
    activation_norm: 100,
    delta_to_activation_ratio: 0.2,
    target_token_id: 42,
    target_token: " Canberra",
    target_logit_before: 21,
    target_logit_after: 21 + 2 * multiplier,
    target_logit_delta: 2 * multiplier,
    target_rank_before: 0,
    target_rank_after: multiplier < 0 ? 3 : 0,
    target_prob_before: 0.6,
    target_prob_after: multiplier < 0 ? 0.1 : 0.8,
    top1_before: { token_id: 42, token: " Canberra", logit: 21, prob: 0.6 },
    top1_after:
      multiplier < 0
        ? { token_id: 506, token: " the", logit: 19, prob: 0.3 }
        : { token_id: 42, token: " Canberra", logit: 23, prob: 0.8 },
    top10_before: [{ token_id: 42, token: " Canberra", logit: 21, prob: 0.6 }],
    top10_after: [{ token_id: 42, token: " Canberra", logit: 23, prob: 0.8 }],
    top10_overlap: 0.9,
    kl_divergence_after_vs_before: 0.05,
    completion_before: " Canberra.",
    completion_after: multiplier < 0 ? " the capital." : " Canberra.",
    control_family: controlFamily,
    matched_target_condition_id: matched,
    provenance: {},
  });

  return {
    schema: "jlens.explorer.bundle.v1",
    provenance: {
      schema_version: "1.0.0",
      exporter_version: "1.0.0",
      source_run_ids: ["jspace_test"],
      source_artifact_fingerprints: {},
      lens_fingerprint: "sha256:7229c7562d1d5542",
      model_repo_id: "google/gemma-4-E4B-it",
      model_revision: "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
      implementation_commit: null,
      created_utc: "2026-07-16T17:09:59+00:00",
      data_status: "measured",
      modalities_present: ["text", "image_text", "audio_text"],
      merged_bundles: [],
      notes: "",
    },
    examples: [
      {
        example_id: textId,
        prompt_slug: "factual-canberra",
        prompt_hash: "26682621dffc6240",
        category: "factual",
        format: "plain",
        modality: "text",
        display_title: "The capital city of Australia is",
        prompt_text: "The capital city of Australia is",
        data_status: "measured",
        seq_len: 7,
        selected_positions: [-2, -1],
        model_output: {
          "-2": {
            input_token_id: 8187,
            input_token: " Australia",
            model_top1_id: 563,
            model_top1_token: " is",
            model_topk: null,
          },
          "-1": {
            input_token_id: 563,
            input_token: " is",
            model_top1_id: 42,
            model_top1_token: " Canberra",
            model_topk: null,
          },
        },
        strength: { tag: "strong", basis: "layer-38 rank 0" },
        selection_reason: null,
        input: {
          text: {
            token_ids: null,
            token_labels: null,
            positions_available: [-2, -1],
            special_token_flags: null,
            prompt_text_is_pre_template: false,
            tokenization_available: false,
          },
          image: null,
          audio: null,
        },
      },
      {
        example_id: imageId,
        prompt_slug: "fixture-card",
        prompt_hash: "aaaabbbbccccdddd",
        category: "multimodal-fixture",
        format: "plain",
        modality: "image_text",
        display_title: "The dominant color is",
        prompt_text: "The dominant color is",
        data_status: "synthetic_fixture",
        seq_len: 295,
        selected_positions: [-1],
        model_output: {
          "-1": {
            input_token_id: 563,
            input_token: " is",
            model_top1_id: 3730,
            model_top1_token: " blue",
            model_topk: null,
          },
        },
        strength: null,
        selection_reason: null,
        input: {
          text: {
            positions_available: [-1],
            tokenization_available: false,
          },
          image: {
            asset_url: "data/fixtures/assets/fixture_image.png",
            width: 96,
            height: 64,
            prompt_text: "The dominant color is",
            modality_token_range: [1, 257],
            processor_metadata: null,
          },
          audio: null,
        },
      },
      {
        example_id: audioId,
        prompt_slug: "fixture-sweep",
        prompt_hash: "eeeeffff00001111",
        category: "multimodal-fixture",
        format: "plain",
        modality: "audio_text",
        display_title: "The sound is a",
        prompt_text: "The sound is a",
        data_status: "synthetic_fixture",
        seq_len: 140,
        selected_positions: [-1],
        model_output: {
          "-1": {
            input_token_id: 496,
            input_token: " a",
            model_top1_id: 10238,
            model_top1_token: " tone",
            model_topk: null,
          },
        },
        strength: null,
        selection_reason: null,
        input: {
          text: {
            positions_available: [-1],
            tokenization_available: false,
          },
          image: null,
          audio: {
            asset_url: "data/fixtures/assets/fixture_audio.wav",
            duration_seconds: 0.4,
            sample_rate: 16000,
            prompt_text: "The sound is a",
            modality_token_range: [1, 121],
            processor_metadata: null,
          },
        },
      },
    ],
    layer_records: [
      layerRecord(textId, 35, -1, 2),
      layerRecord(textId, 38, -1, 0),
      layerRecord(textId, 35, -2, 9),
      layerRecord(textId, 38, -2, 4),
      layerRecord(imageId, 38, -1, 2),
      layerRecord(audioId, 38, -1, 3),
    ],
    cones: [
      cone(textId, 35, -1),
      cone(textId, 38, -1),
      cone(textId, 35, -2),
      cone(textId, 38, -2),
      cone(imageId, 38, -1),
      cone(audioId, 38, -1),
    ],
    pursuit_traces: [
      trace(textId, 35, -1),
      trace(textId, 38, -1),
      trace(textId, 35, -2),
      trace(textId, 38, -2),
      trace(imageId, 38, -1),
      trace(audioId, 38, -1),
    ],
    trajectories: [
      {
        example_id: textId,
        position: -1,
        layer_from: 35,
        layer_to: 38,
        retained_atoms: [{ token_id: 42, label: " Canberra" }],
        entered_atoms: [{ token_id: 7, label: " capital" }],
        exited_atoms: [{ token_id: 9, label: " city" }],
        jaccard: 0.33,
        weighted_similarity: 0.7,
        explained_fraction_from: 0.15,
        explained_fraction_to: 0.19,
        delta_explained_fraction: 0.04,
        output_token_persistence: { in_from: true, in_to: true },
        data_status: "measured",
      },
    ],
    causal_records: [
      causal("cond_target_plus", 1, null, null),
      causal("cond_target_zero", 0, null, null),
      causal("cond_target_minus", -1, null, null),
      causal("cond_control_minus", -1, "isotropic_random_direction", "cond_target_minus"),
    ],
    causal_baseline_parity: { worst_max_abs_logit_diff: 0.001 },
  };
}

export function loadedData(bundle = testBundle()): LoadedData {
  return {
    bundle,
    sources: [{ path: "data/text_demo.json", status: "measured" }],
    warnings: [],
  };
}
