import { useMemo, useState } from "react";
import type { CausalRecord } from "../types";
import { formatNumber, formatPercent, tokenLabel } from "../lib/format";
import { StatusBadge } from "./Badge";

const KIND_LABELS: Record<string, string> = {
  output_atom_contribution: "output atom",
  top_non_output_atom_contribution: "top non-output atom",
  full_cone_reconstruction: "full cone",
  isotropic_random_direction: "random control",
  nuisance_direction: "nuisance control",
  cross_prompt_cone: "cross-prompt control",
};

function RecordReadout({ record, title }: { record: CausalRecord | null; title: string }) {
  if (!record) {
    return (
      <div className="causal-readout">
        <h4>{title}</h4>
        <p className="muted small">no measured condition here</p>
      </div>
    );
  }
  return (
    <div className="causal-readout">
      <h4>
        {title} <StatusBadge status={record.status} />
      </h4>
      <dl className="stat-list">
        <dt>target token</dt>
        <dd>
          <code>{tokenLabel(record.target_token)}</code>
          {record.target_token_id != null && (
            <span className="muted small"> #{record.target_token_id}</span>
          )}
        </dd>
        <dt>top-1 before → after</dt>
        <dd>
          <code>{tokenLabel(record.top1_before?.token)}</code> →{" "}
          <code>{tokenLabel(record.top1_after?.token)}</code>
        </dd>
        <dt>target logit Δ</dt>
        <dd>{formatNumber(record.target_logit_delta, 3)}</dd>
        <dt>target prob before → after</dt>
        <dd>
          {formatPercent(record.target_prob_before, 2)} →{" "}
          {formatPercent(record.target_prob_after, 2)}
        </dd>
        <dt>target rank before → after</dt>
        <dd>
          {record.target_rank_before ?? "—"} → {record.target_rank_after ?? "—"}
        </dd>
        <dt>KL(after ‖ before)</dt>
        <dd>{formatNumber(record.kl_divergence_after_vs_before, 4)}</dd>
        <dt>top-10 overlap</dt>
        <dd>{formatPercent(record.top10_overlap)}</dd>
        <dt>‖Δ‖ / ‖h‖</dt>
        <dd>{formatNumber(record.delta_to_activation_ratio, 3)}</dd>
      </dl>
      {(record.completion_before || record.completion_after) && (
        <div className="completion-pair small">
          <p>
            <strong>before:</strong> {record.completion_before ?? "—"}
          </p>
          <p>
            <strong>after:</strong> {record.completion_after ?? "—"}
          </p>
        </div>
      )}
    </div>
  );
}

/** Measured causal steering records. Only multipliers that were actually
 * run are selectable — nothing is interpolated. Synthetic fixtures are
 * loudly badged and never presented as experimental evidence. */
export function CausalPanel({
  records,
  exampleId,
}: {
  records: CausalRecord[];
  exampleId: string | null;
}) {
  const forExample = useMemo(
    () => records.filter((r) => r.example_id === exampleId),
    [records, exampleId],
  );

  const targeted = forExample.filter((r) => !r.control_family);
  const kinds = [...new Set(targeted.map((r) => r.target_kind))].sort();
  const [kind, setKind] = useState<string>("");
  const activeKind = kinds.includes(kind as never) ? kind : (kinds[0] ?? "");

  const kindRecords = targeted.filter((r) => r.target_kind === activeKind);
  const layers = [...new Set(kindRecords.map((r) => r.layer))].sort((a, b) => a - b);
  const [layerChoice, setLayerChoice] = useState<number | null>(null);
  const layer = layerChoice !== null && layers.includes(layerChoice) ? layerChoice : layers[0];

  const layerRecords = kindRecords.filter((r) => r.layer === layer);
  const positions = [...new Set(layerRecords.map((r) => r.position))].sort((a, b) => a - b);
  const [positionChoice, setPositionChoice] = useState<number | null>(null);
  const position =
    positionChoice !== null && positions.includes(positionChoice)
      ? positionChoice
      : positions[0];

  const positionRecords = layerRecords.filter((r) => r.position === position);
  const normVariants = [...new Set(positionRecords.map((r) => Boolean(r.norm_preserving)))];
  const [normPreserving, setNormPreserving] = useState(false);
  const activeNorm = normVariants.includes(normPreserving) ? normPreserving : normVariants[0];

  const variantRecords = positionRecords.filter(
    (r) => Boolean(r.norm_preserving) === activeNorm,
  );
  const multipliers = [...new Set(variantRecords.map((r) => r.multiplier))].sort(
    (a, b) => a - b,
  );
  const [multiplierChoice, setMultiplierChoice] = useState<number | null>(null);
  const multiplier =
    multiplierChoice !== null && multipliers.includes(multiplierChoice)
      ? multiplierChoice
      : multipliers[0];

  const record =
    variantRecords.find((r) => r.multiplier === multiplier) ?? null;
  const control =
    record == null
      ? null
      : (forExample.find(
          (r) =>
            r.control_family &&
            r.matched_target_condition_id === record.condition_id,
        ) ??
        forExample.find(
          (r) =>
            r.control_family &&
            r.layer === record.layer &&
            r.position === record.position &&
            r.multiplier === record.multiplier &&
            Boolean(r.norm_preserving) === Boolean(record.norm_preserving),
        ) ??
        null);

  if (forExample.length === 0) {
    return (
      <section className="panel causal-panel" aria-label="Causal steering">
        <h2>Causal steering</h2>
        <StatusBadge status={null} />
        <p className="muted">
          No intervention records exist for this example yet. Run
          <code> notebooks/gemma_4_e4b_jspace_causal_smoke.ipynb</code> on an L4 and
          merge its <code>explorer_causal_bundle.json</code> to populate this panel
          with measured results.
        </p>
      </section>
    );
  }

  return (
    <section className="panel causal-panel" aria-label="Causal steering">
      <h2>Causal steering</h2>
      <div className="causal-controls">
        <label>
          target
          <select value={activeKind} onChange={(e) => setKind(e.target.value)}>
            {kinds.map((value) => (
              <option key={value} value={value}>
                {KIND_LABELS[value] ?? value}
              </option>
            ))}
          </select>
        </label>
        <label>
          layer
          <select
            value={String(layer)}
            onChange={(e) => setLayerChoice(Number(e.target.value))}
          >
            {layers.map((value) => (
              <option key={value} value={value}>
                L{value}
              </option>
            ))}
          </select>
        </label>
        <label>
          position
          <select
            value={String(position)}
            onChange={(e) => setPositionChoice(Number(e.target.value))}
          >
            {positions.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        {normVariants.length > 1 && (
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={activeNorm}
              onChange={(e) => setNormPreserving(e.target.checked)}
            />
            norm-preserving variant
          </label>
        )}
        <div className="multiplier-group" role="group" aria-label="Measured multipliers">
          {multipliers.map((value) => (
            <button
              key={value}
              className={value === multiplier ? "mult-btn mult-btn-active" : "mult-btn"}
              aria-pressed={value === multiplier}
              onClick={() => setMultiplierChoice(value)}
            >
              {value > 0 ? `+${value}` : String(value)}
            </button>
          ))}
          <span className="muted small">only measured multipliers are shown</span>
        </div>
      </div>

      {record?.atom_label != null && (
        <p>
          intervening on atom <code>{tokenLabel(record.atom_label)}</code>
          <span className="muted small"> #{record.atom_token_id}</span> ×{" "}
          {record.multiplier}
        </p>
      )}

      <div className="causal-compare">
        <RecordReadout record={record} title="Targeted intervention" />
        <RecordReadout
          record={control}
          title={`Matched control${control?.control_family ? ` (${KIND_LABELS[control.target_kind] ?? control.control_family})` : ""}`}
        />
      </div>
    </section>
  );
}
