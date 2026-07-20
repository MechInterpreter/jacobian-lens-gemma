import type { ConeRecord, TrajectoryTransition } from "../types";
import { formatNumber, formatPercent, tokenLabel } from "../lib/format";

/** The same activation's cone across layers: per-layer atom columns joined
 * by transition stats (Jaccard, weighted similarity, Δ explained fraction)
 * with retained / entered / exited atoms. */
export function TrajectoryView({
  cones,
  transitions,
}: {
  cones: ConeRecord[];
  transitions: TrajectoryTransition[];
}) {
  const ordered = [...cones].sort((a, b) => a.layer - b.layer);
  if (ordered.length === 0) {
    return (
      <section className="panel trajectory-panel" aria-label="Cross-layer trajectory">
        <h2>Cross-layer trajectory</h2>
        <p className="muted">No cones for this selection.</p>
      </section>
    );
  }
  const byPair = new Map(
    transitions.map((t) => [`${t.layer_from}->${t.layer_to}`, t]),
  );
  return (
    <section className="panel trajectory-panel" aria-label="Cross-layer trajectory">
      <h2>Cross-layer trajectory</h2>
      <div className="trajectory-flow">
        {ordered.map((cone, index) => {
          const next = ordered[index + 1];
          const transition = next
            ? byPair.get(`${cone.layer}->${next.layer}`)
            : undefined;
          const retained = new Set(
            (transition?.retained_atoms ?? []).map((a) => a.token_id),
          );
          return (
            <div className="trajectory-cell" key={cone.layer}>
              <div className="trajectory-layer">
                <h3>L{cone.layer}</h3>
                <p className="muted small">
                  explained {formatPercent(cone.reconstruction.explained_fraction, 2)}
                </p>
                <ul className="trajectory-atoms">
                  {cone.selected_atoms
                    .filter((atom) => atom.is_effective)
                    .map((atom) => (
                      <li
                        key={atom.token_id}
                        className={
                          retained.has(atom.token_id)
                            ? "trajectory-atom trajectory-atom-retained"
                            : "trajectory-atom"
                        }
                      >
                        <code>{tokenLabel(atom.label)}</code>
                        {atom.is_output_token && <span className="marker marker-output">★</span>}
                      </li>
                    ))}
                </ul>
              </div>
              {transition && (
                <div className="trajectory-transition" role="group"
                  aria-label={`transition L${transition.layer_from} to L${transition.layer_to}`}>
                  <span className="trajectory-arrow" aria-hidden="true">→</span>
                  <dl className="stat-list small">
                    <dt>Jaccard</dt>
                    <dd>{formatNumber(transition.jaccard, 2)}</dd>
                    <dt>weighted sim</dt>
                    <dd>{formatNumber(transition.weighted_similarity, 2)}</dd>
                    <dt>Δ explained</dt>
                    <dd>
                      {transition.delta_explained_fraction == null
                        ? "—"
                        : formatPercent(transition.delta_explained_fraction, 2)}
                    </dd>
                  </dl>
                  <p className="small trajectory-inout">
                    <span className="entered">
                      +{transition.entered_atoms?.length ?? 0} entered
                    </span>{" "}
                    <span className="exited">
                      −{transition.exited_atoms?.length ?? 0} exited
                    </span>{" "}
                    <span className="retained">
                      {transition.retained_atoms?.length ?? 0} retained
                    </span>
                  </p>
                  {transition.output_token_persistence && (
                    <p className="small muted">
                      output token: {transition.output_token_persistence.in_from ? "in" : "out"} →{" "}
                      {transition.output_token_persistence.in_to ? "in" : "out"}
                    </p>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <p className="muted small">
        Highlighted atoms persist into the next layer's cone. Layer 21's anomalous
        lens fit is documented in the run report; its records are shown as
        historical data.
      </p>
    </section>
  );
}
