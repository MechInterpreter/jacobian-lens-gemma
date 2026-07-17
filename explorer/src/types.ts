// TypeScript mirror of schemas/explorer_bundle.schema.json (v1.x).

export type DataStatus = "measured" | "imported" | "synthetic_fixture";
export type Modality = "text" | "image" | "audio" | "image_text" | "audio_text";

export interface TokenScore {
  token_id: number;
  token: string | null;
  logit?: number | null;
  prob?: number | null;
}

export interface PositionOutput {
  input_token_id?: number;
  input_token?: string;
  model_top1_id?: number;
  model_top1_token?: string;
  model_topk?: TokenScore[] | null;
}

export interface TextInput {
  token_ids?: number[] | null;
  token_labels?: string[] | null;
  positions_available?: number[];
  special_token_flags?: boolean[] | null;
  prompt_text_is_pre_template?: boolean | null;
  tokenization_available: boolean;
}

export interface ImageInput {
  asset_url?: string | null;
  width?: number | null;
  height?: number | null;
  prompt_text?: string | null;
  modality_token_range?: [number, number] | null;
  processor_metadata?: Record<string, unknown> | null;
}

export interface AudioInput {
  asset_url?: string | null;
  duration_seconds?: number | null;
  sample_rate?: number | null;
  prompt_text?: string | null;
  modality_token_range?: [number, number] | null;
  processor_metadata?: Record<string, unknown> | null;
}

export interface ExampleInput {
  text?: TextInput | null;
  image?: ImageInput | null;
  audio?: AudioInput | null;
}

export interface Strength {
  tag: "strong" | "weak" | "middling";
  basis: string;
}

export interface Example {
  example_id: string;
  prompt_slug?: string | null;
  prompt_hash: string;
  category: string;
  format: "plain" | "chat";
  modality: Modality;
  display_title: string;
  prompt_text?: string | null;
  data_status: DataStatus;
  seq_len?: number | null;
  selected_positions?: number[];
  model_output?: Record<string, PositionOutput>;
  strength?: Strength | null;
  selection_reason?: string | null;
  input: ExampleInput;
}

export interface LayerRecord {
  example_id: string;
  layer: number;
  position: number;
  source_site: "block_output";
  data_status: DataStatus;
  input_token_id?: number | null;
  input_token?: string | null;
  model_topk?: TokenScore[] | null;
  jlens_topk?: TokenScore[] | null;
  rank_of_model_top1?: number | null;
  topk_overlap_with_model?: number | null;
  eval_metadata?: Record<string, unknown> | null;
  target_activation_norm?: number | null;
  residual_norm?: number | null;
  relative_residual?: number | null;
  explained_fraction?: number | null;
}

export interface ConeAtom {
  token_id: number;
  label: string;
  coefficient: number;
  is_output_token: boolean;
  is_effective: boolean;
  coefficient_share?: number | null;
  nuisance?: {
    total_selections?: number;
    n_distinct_prompts?: number;
    high_frequency?: boolean;
  } | null;
}

export interface ConeRecord {
  example_id: string;
  layer: number;
  position: number;
  requested_k: number;
  n_selected?: number;
  data_status: DataStatus;
  selected_atoms: ConeAtom[];
  coefficient_sum?: number | null;
  top_coefficient?: number | null;
  concentration?: {
    herfindahl: number;
    top1_share: number;
    n_nonzero: number;
  } | null;
  reconstruction: {
    target_norm: number;
    residual_norm: number;
    relative_residual: number;
    explained_fraction: number;
  };
  cone_signature_digest?: string | null;
  source_provenance: Record<string, unknown>;
}

export interface PursuitStep {
  step: number;
  added_token_id: number;
  added_label: string;
  support_after?: number[];
  residual_norm: number;
  relative_residual: number;
  explained_fraction?: number | null;
  coefficients_after?: number[] | null;
  final_coefficient_zero?: boolean | null;
}

export interface PursuitTrace {
  example_id: string;
  layer: number;
  position: number;
  requested_k?: number;
  n_iterations: number;
  stop_reason: string;
  data_status: DataStatus;
  per_step_coefficients_available: boolean;
  initial_residual_norm: number;
  target_norm: number;
  steps: PursuitStep[];
}

export interface LabelledToken {
  token_id: number;
  label: string;
}

export interface TrajectoryTransition {
  example_id: string;
  position: number;
  layer_from: number;
  layer_to: number;
  retained_atoms?: LabelledToken[];
  entered_atoms?: LabelledToken[];
  exited_atoms?: LabelledToken[];
  jaccard: number;
  weighted_similarity: number;
  explained_fraction_from?: number | null;
  explained_fraction_to?: number | null;
  delta_explained_fraction?: number | null;
  output_token_persistence?: { in_from: boolean; in_to: boolean } | null;
  data_status: DataStatus;
}

export type TargetKind =
  | "output_atom_contribution"
  | "top_non_output_atom_contribution"
  | "full_cone_reconstruction"
  | "isotropic_random_direction"
  | "nuisance_direction"
  | "cross_prompt_cone";

export interface CausalRecord {
  condition_id: string;
  example_id: string;
  layer: number;
  position: number;
  target_kind: TargetKind;
  atom_token_id?: number | null;
  atom_label?: string | null;
  multiplier: number;
  status: DataStatus;
  norm_preserving?: boolean | null;
  delta_norm?: number | null;
  activation_norm?: number | null;
  delta_to_activation_ratio?: number | null;
  target_token_id?: number | null;
  target_token?: string | null;
  target_logit_before?: number | null;
  target_logit_after?: number | null;
  target_logit_delta?: number | null;
  target_rank_before?: number | null;
  target_rank_after?: number | null;
  target_prob_before?: number | null;
  target_prob_after?: number | null;
  top1_before?: TokenScore | null;
  top1_after?: TokenScore | null;
  top10_before?: TokenScore[] | null;
  top10_after?: TokenScore[] | null;
  top10_overlap?: number | null;
  kl_divergence_after_vs_before?: number | null;
  completion_before?: string | null;
  completion_after?: string | null;
  control_family?: string | null;
  matched_target_condition_id?: string | null;
  provenance?: Record<string, unknown> | null;
}

export interface BundleProvenance {
  schema_version: string;
  exporter_version: string;
  source_run_ids: string[];
  source_artifact_fingerprints?: Record<string, string>;
  lens_fingerprint?: string;
  model_repo_id: string;
  model_revision: string;
  implementation_commit?: string | null;
  created_utc: string;
  data_status: DataStatus;
  modalities_present: Modality[];
  merged_bundles?: Record<string, unknown>[];
  notes?: string;
}

export interface ExplorerBundle {
  schema: "jlens.explorer.bundle.v1";
  provenance: BundleProvenance;
  examples: Example[];
  layer_records: LayerRecord[];
  cones: ConeRecord[];
  pursuit_traces: PursuitTrace[];
  trajectories: TrajectoryTransition[];
  causal_records: CausalRecord[];
  causal_baseline_parity?: Record<string, unknown>;
  notes?: string;
}
