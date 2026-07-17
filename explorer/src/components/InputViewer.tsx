import type { Example } from "../types";
import { formatNumber, tokenLabel } from "../lib/format";

/** Input viewer: prompt, token/position metadata, and (for image/audio)
 * the raw asset. Localization beyond recorded token ranges is never
 * implied — the lens reads decoder states, not pixels or audio spans. */
export function InputViewer({
  example,
  selectedPosition,
  onSelectPosition,
}: {
  example: Example;
  selectedPosition: number | null;
  onSelectPosition: (position: number) => void;
}) {
  const positions = example.selected_positions ?? [];
  const image = example.input.image;
  const audio = example.input.audio;
  const text = example.input.text;

  return (
    <section className="panel input-viewer" aria-label="Input viewer">
      <h2>Input</h2>

      {image && (
        <figure className="input-asset">
          {image.asset_url ? (
            <img
              src={image.asset_url}
              alt={image.prompt_text ?? "input image"}
              className="input-image"
            />
          ) : (
            <p className="muted">image asset not bundled</p>
          )}
          <figcaption className="muted">
            {image.width && image.height ? `${image.width}×${image.height}px` : "size unknown"}
            {image.modality_token_range
              ? ` · image tokens at sequence positions [${image.modality_token_range[0]}, ${image.modality_token_range[1]})`
              : " · image token range not recorded"}
          </figcaption>
        </figure>
      )}

      {audio && (
        <figure className="input-asset">
          {audio.asset_url ? (
            <audio controls src={audio.asset_url} aria-label="input audio clip" />
          ) : (
            <p className="muted">audio asset not bundled</p>
          )}
          <figcaption className="muted">
            {audio.duration_seconds != null
              ? `${formatNumber(audio.duration_seconds, 1)} s`
              : "duration unknown"}
            {audio.sample_rate ? ` · ${audio.sample_rate} Hz` : ""}
            {audio.modality_token_range
              ? ` · audio tokens at sequence positions [${audio.modality_token_range[0]}, ${audio.modality_token_range[1]})`
              : " · audio token range not recorded"}
          </figcaption>
        </figure>
      )}

      <p className="prompt-text">
        {example.prompt_text ?? image?.prompt_text ?? audio?.prompt_text ?? (
          <span className="muted">prompt text not recorded (hash {example.prompt_hash})</span>
        )}
        {text?.prompt_text_is_pre_template && (
          <span className="muted">
            {" "}
            (user message; the run tokenized the rendered chat template)
          </span>
        )}
      </p>

      <div className="token-strip" role="group" aria-label="Token positions">
        {text?.tokenization_available && text.token_labels ? (
          text.token_labels.map((label, index) => {
            const position = index - text.token_labels!.length;
            const selectable = positions.includes(position);
            return (
              <button
                key={index}
                className={
                  position === selectedPosition
                    ? "token-chip token-chip-active"
                    : "token-chip"
                }
                disabled={!selectable}
                aria-pressed={position === selectedPosition}
                onClick={() => onSelectPosition(position)}
              >
                {tokenLabel(label)}
              </button>
            );
          })
        ) : (
          <>
            <span className="muted">
              full tokenization not persisted by the source run · sequence length{" "}
              {example.seq_len ?? "?"} · recorded positions:
            </span>
            {positions.map((position) => {
              const output = example.model_output?.[String(position)];
              return (
                <button
                  key={position}
                  className={
                    position === selectedPosition
                      ? "token-chip token-chip-active"
                      : "token-chip"
                  }
                  aria-pressed={position === selectedPosition}
                  onClick={() => onSelectPosition(position)}
                >
                  {position}: {tokenLabel(output?.input_token)}
                </button>
              );
            })}
          </>
        )}
      </div>
    </section>
  );
}
