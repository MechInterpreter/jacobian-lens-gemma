import type { BundleProvenance } from "../types";
import { shortSha } from "../lib/format";
import { StatusBadge } from "./Badge";

export function ProvenancePanel({
  provenance,
  layer,
  position,
  warnings,
}: {
  provenance: BundleProvenance;
  layer: number | null;
  position: number | null;
  warnings: string[];
}) {
  return (
    <section className="panel provenance-panel" aria-label="Provenance and interpretation">
      <h2>Provenance &amp; interpretation</h2>
      <dl className="stat-list">
        <dt>source runs</dt>
        <dd>{provenance.source_run_ids.join(", ")}</dd>
        <dt>model</dt>
        <dd>
          {provenance.model_repo_id} @ <code>{shortSha(provenance.model_revision)}</code>
        </dd>
        <dt>lens fingerprint</dt>
        <dd>
          <code>{shortSha(provenance.lens_fingerprint, 16)}</code>
        </dd>
        <dt>viewing</dt>
        <dd>
          {layer !== null ? `layer ${layer} (block_output)` : "no layer selected"}
          {position !== null ? `, position ${position}` : ""}
        </dd>
        <dt>bundle schema</dt>
        <dd>
          jlens.explorer.bundle.v1 (v{provenance.schema_version}, exporter{" "}
          {provenance.exporter_version})
        </dd>
        <dt>data status</dt>
        <dd>
          <StatusBadge status={provenance.data_status} />
        </dd>
      </dl>
      {warnings.length > 0 && (
        <ul className="provenance-warnings small">
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      <details>
        <summary>limitations</summary>
        <ul className="small">
          <li>
            The J-lens is a linear approximation fitted on 100 text prompts; it
            does not replace the model's forward computation.
          </li>
          <li>
            k=10 cones explain a small fraction of activation norm at most
            layers; the residual bar shows exactly how much is unexplained.
          </li>
          <li>
            Cone signatures are deterministic bookkeeping, not concept claims;
            recurrence does not establish shared semantics.
          </li>
          <li>
            Image/audio records apply the <em>text-fitted</em> lens to
            multimodal-conditioned decoder states — an exploratory probe, not
            evidence of modality-invariant concepts.
          </li>
          <li>
            No pixel- or audio-span attribution is available; only recorded
            modality token ranges are shown.
          </li>
          <li>
            Causal effects are shown only at multipliers that were actually
            measured; nothing is interpolated.
          </li>
          <li>
            Layer 21's anomalous lens fit is documented in the research log;
            its records are historical data, not an active workstream.
          </li>
        </ul>
      </details>
    </section>
  );
}
