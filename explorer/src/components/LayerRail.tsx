import type { ConeRecord, TrajectoryTransition } from "../types";
import { formatPercent } from "../lib/format";

export interface LayerSummary {
  layer: number;
  explainedFraction: number | null;
  outputAligned: boolean | null;
  /** Jaccard similarity with the previous available layer's cone. */
  stability: number | null;
}

export function summarizeLayers(
  layers: number[],
  cones: ConeRecord[],
  trajectories: TrajectoryTransition[],
): LayerSummary[] {
  return layers.map((layer) => {
    const cone = cones.find((c) => c.layer === layer);
    const inbound = trajectories.find((t) => t.layer_to === layer);
    return {
      layer,
      explainedFraction: cone?.reconstruction.explained_fraction ?? null,
      outputAligned: cone
        ? cone.selected_atoms.some((a) => a.is_output_token && a.is_effective)
        : null,
      stability: inbound ? inbound.jaccard : null,
    };
  });
}

export function LayerRail({
  summaries,
  selectedLayer,
  onSelect,
}: {
  summaries: LayerSummary[];
  selectedLayer: number | null;
  onSelect: (layer: number) => void;
}) {
  return (
    <nav className="panel layer-rail" aria-label="Layer selection">
      <h2>Layers</h2>
      <div className="layer-rail-buttons" role="group">
        {summaries.map((summary) => (
          <button
            key={summary.layer}
            className={
              summary.layer === selectedLayer ? "layer-btn layer-btn-active" : "layer-btn"
            }
            aria-pressed={summary.layer === selectedLayer}
            onClick={() => onSelect(summary.layer)}
          >
            <span className="layer-number">L{summary.layer}</span>
            <span
              className="layer-ef-bar"
              title={`explained fraction ${formatPercent(summary.explainedFraction)}`}
              aria-hidden="true"
            >
              <span
                className="layer-ef-fill"
                style={{
                  width: `${Math.min(100, 100 * (summary.explainedFraction ?? 0))}%`,
                }}
              />
            </span>
            <span className="layer-indicators">
              <span
                className={
                  summary.outputAligned
                    ? "indicator indicator-on"
                    : "indicator indicator-off"
                }
                title={
                  summary.outputAligned === null
                    ? "no cone at this layer"
                    : summary.outputAligned
                      ? "model output token is in the k=10 cone"
                      : "model output token is NOT in the k=10 cone"
                }
              >
                ●
              </span>
              <span className="layer-stability" title="Jaccard vs previous layer">
                {summary.stability === null ? "·" : summary.stability.toFixed(2)}
              </span>
            </span>
          </button>
        ))}
      </div>
      <p className="muted small">
        bar: explained fraction · dot: output-token alignment · number:
        cross-layer Jaccard stability
      </p>
    </nav>
  );
}
