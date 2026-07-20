import type { ConeRecord } from "../types";
import { formatNumber, formatPercent, tokenLabel } from "../lib/format";

/** k=10 sparse cone: coefficient bars per atom, output-token and
 * high-frequency markers, reconstruction quality. The bar chart shows
 * coefficient magnitudes — it is NOT a geometric picture of the
 * 2560-dimensional residual space. */
export function ConePanel({ cone }: { cone: ConeRecord | null }) {
  if (!cone) {
    return (
      <section className="panel cone-panel" aria-label="Sparse cone">
        <h2>Sparse cone (k=10)</h2>
        <p className="muted">No cone record for this selection.</p>
      </section>
    );
  }
  const maxCoefficient = Math.max(
    ...cone.selected_atoms.map((atom) => Math.abs(atom.coefficient)),
    1e-9,
  );
  const explained = cone.reconstruction.explained_fraction;
  return (
    <section className="panel cone-panel" aria-label="Sparse cone">
      <h2>
        Sparse cone (k={cone.requested_k}, layer {cone.layer}, position {cone.position})
      </h2>
      <ul className="atom-list">
        {cone.selected_atoms.map((atom) => (
          <li key={atom.token_id} className={atom.is_effective ? "" : "atom-zero"}>
            <span className="atom-label">
              <code>{tokenLabel(atom.label)}</code>
              {atom.is_output_token && (
                <span className="marker marker-output" title="model's measured top-1 output token">
                  ★ output
                </span>
              )}
              {atom.nuisance?.high_frequency && (
                <span
                  className="marker marker-nuisance"
                  title={`selected in ${atom.nuisance.n_distinct_prompts} distinct prompts across the run — candidate frequency/nuisance atom`}
                >
                  ◆ frequent
                </span>
              )}
              {!atom.is_effective && (
                <span className="marker" title="coefficient refined to zero">
                  zeroed
                </span>
              )}
            </span>
            <span className="atom-bar" aria-hidden="true">
              <span
                className={atom.is_output_token ? "atom-fill atom-fill-output" : "atom-fill"}
                style={{ width: `${(100 * Math.abs(atom.coefficient)) / maxCoefficient}%` }}
              />
            </span>
            <span className="atom-value">
              {formatNumber(atom.coefficient, 2)}
              {atom.coefficient_share != null && (
                <span className="muted small"> ({formatPercent(atom.coefficient_share)})</span>
              )}
            </span>
            <span className="muted small atom-id">#{atom.token_id}</span>
          </li>
        ))}
      </ul>
      <div className="recon-summary">
        <div className="recon-bar" role="img" aria-label={`explained fraction ${formatPercent(explained)}`}>
          <span className="recon-fill" style={{ width: `${Math.min(100, 100 * explained)}%` }} />
        </div>
        <dl className="stat-list stat-inline">
          <dt>explained fraction</dt>
          <dd>{formatPercent(explained, 2)}</dd>
          <dt>unexplained residual (relative)</dt>
          <dd>{formatPercent(cone.reconstruction.relative_residual, 2)}</dd>
          <dt>target ‖h‖</dt>
          <dd>{formatNumber(cone.reconstruction.target_norm, 1)}</dd>
          {cone.concentration && (
            <>
              <dt>top-1 coefficient share</dt>
              <dd>{formatPercent(cone.concentration.top1_share)}</dd>
            </>
          )}
        </dl>
      </div>
      <p className="muted small">
        Bars show nonnegative pursuit coefficients over raw J-lens atoms; this is a
        coefficient readout, not a geometric rendering of the 2560-dimensional space.
      </p>
      <details>
        <summary>exact record metadata</summary>
        <pre className="metadata-dump">{JSON.stringify(
          {
            cone_signature_digest: cone.cone_signature_digest,
            n_selected: cone.n_selected,
            concentration: cone.concentration,
            reconstruction: cone.reconstruction,
            source_provenance: cone.source_provenance,
          },
          null,
          2,
        )}</pre>
      </details>
    </section>
  );
}
