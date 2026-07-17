import type { Example, LayerRecord } from "../types";
import { formatPercent, formatRank, tokenLabel } from "../lib/format";

function TokenList({
  tokens,
  emptyNote,
}: {
  tokens: { token_id: number; token: string | null; prob?: number | null }[] | null | undefined;
  emptyNote: string;
}) {
  if (!tokens || tokens.length === 0) {
    return <p className="muted small">{emptyNote}</p>;
  }
  return (
    <ol className="token-rank-list">
      {tokens.map((token) => (
        <li key={token.token_id}>
          <code>{tokenLabel(token.token)}</code>
          {token.prob != null && (
            <span className="muted small"> {formatPercent(token.prob)}</span>
          )}
        </li>
      ))}
    </ol>
  );
}

/** Model vs J-lens readout at the selected layer/position. The J-lens is an
 * approximation of the model's computation, not the model itself — the two
 * columns are labelled accordingly. */
export function PredictionPanel({
  example,
  record,
  position,
}: {
  example: Example;
  record: LayerRecord | null;
  position: number;
}) {
  const output = example.model_output?.[String(position)];
  return (
    <section className="panel prediction-panel" aria-label="Predictions">
      <h2>Predictions</h2>
      {!record ? (
        <p className="muted">No layer record for this selection.</p>
      ) : (
        <div className="prediction-grid">
          <div>
            <h3>Model (measured output)</h3>
            <p>
              final top-1:{" "}
              <code className="token-big">{tokenLabel(output?.model_top1_token)}</code>
            </p>
            <TokenList
              tokens={record.model_topk}
              emptyNote="full model top-k list not persisted by the source run"
            />
          </div>
          <div>
            <h3>J-lens (layer {record.layer} approximation)</h3>
            <TokenList
              tokens={record.jlens_topk}
              emptyNote="J-lens top-k token list not persisted by the source run; agreement metrics below are measured"
            />
            <dl className="stat-list">
              <dt>rank of model top-1 in J-lens readout</dt>
              <dd>{formatRank(record.rank_of_model_top1)}</dd>
              <dt>top-k overlap with model</dt>
              <dd>
                {record.topk_overlap_with_model == null
                  ? "not recorded"
                  : formatPercent(record.topk_overlap_with_model)}
                {record.eval_metadata != null && (
                  <span className="muted small"> (aggregated over the example's positions)</span>
                )}
              </dd>
            </dl>
          </div>
        </div>
      )}
      <p className="muted small">
        The J-lens transports the layer-{record?.layer ?? "l"} residual through a
        fitted linear map; it approximates, and does not replace, the model's own
        forward computation.
      </p>
    </section>
  );
}
