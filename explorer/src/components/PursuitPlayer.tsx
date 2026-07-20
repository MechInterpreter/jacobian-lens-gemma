import { useEffect, useRef, useState } from "react";
import type { PursuitTrace } from "../types";
import { formatNumber, formatPercent, tokenLabel } from "../lib/format";

/** Step-by-step playback of the recorded gradient pursuit. Every number
 * shown comes from the run's recorded residual-norm history and selection
 * order; per-step coefficients were not recorded and are labelled as such,
 * never invented. */
export function PursuitPlayer({ trace }: { trace: PursuitTrace | null }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const traceKey = trace ? `${trace.example_id}|${trace.layer}|${trace.position}` : "";

  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [traceKey]);

  useEffect(() => {
    if (!playing || !trace) return;
    timer.current = setInterval(() => {
      setStep((current) => {
        if (current >= trace.steps.length) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 700);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [playing, trace]);

  if (!trace) {
    return (
      <section className="panel pursuit-panel" aria-label="Gradient pursuit playback">
        <h2>Gradient-pursuit playback</h2>
        <p className="muted">No pursuit trace for this selection.</p>
      </section>
    );
  }

  const norms = [trace.initial_residual_norm, ...trace.steps.map((s) => s.residual_norm)];
  const maxNorm = Math.max(...norms, 1e-9);
  const minNorm = Math.min(...norms);
  const width = 420;
  const height = 120;
  const pad = 8;
  const x = (index: number) =>
    pad + (index * (width - 2 * pad)) / Math.max(1, norms.length - 1);
  const y = (value: number) => {
    const span = Math.max(maxNorm - minNorm, 1e-9);
    return pad + ((maxNorm - value) * (height - 2 * pad)) / span;
  };
  const path = norms.map((v, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(v)}`).join(" ");
  const current = step > 0 ? trace.steps[step - 1] : null;

  return (
    <section className="panel pursuit-panel" aria-label="Gradient pursuit playback">
      <h2>Gradient-pursuit playback</h2>
      <div className="pursuit-controls">
        <button
          onClick={() => setPlaying((value) => !value)}
          aria-label={playing ? "Pause playback" : "Play playback"}
        >
          {playing ? "⏸ pause" : "▶ play"}
        </button>
        <button
          onClick={() => setStep((value) => Math.max(0, value - 1))}
          disabled={step === 0}
          aria-label="Previous step"
        >
          ◀ prev
        </button>
        <button
          onClick={() => setStep((value) => Math.min(trace.steps.length, value + 1))}
          disabled={step >= trace.steps.length}
          aria-label="Next step"
        >
          next ▶
        </button>
        <input
          type="range"
          min={0}
          max={trace.steps.length}
          value={step}
          aria-label="Pursuit step"
          onChange={(event) => setStep(Number(event.target.value))}
        />
        <span className="pursuit-step-label">
          step {step} / {trace.steps.length}
        </span>
      </div>

      <svg
        className="residual-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Residual norm per pursuit step"
      >
        <path d={path} className="residual-line" fill="none" />
        {norms.map((value, index) => (
          <circle
            key={index}
            cx={x(index)}
            cy={y(value)}
            r={index === step ? 5 : 2.5}
            className={index === step ? "residual-dot residual-dot-active" : "residual-dot"}
          />
        ))}
      </svg>

      <div className="pursuit-state">
        {step === 0 ? (
          <p>
            start: residual = target, ‖r‖ = {formatNumber(trace.initial_residual_norm, 1)}
          </p>
        ) : (
          current && (
            <p>
              step {current.step}: added{" "}
              <code className="pursuit-added">{tokenLabel(current.added_label)}</code>{" "}
              (#{current.added_token_id}) · ‖r‖ {formatNumber(current.residual_norm, 1)} ·
              explained {formatPercent(current.explained_fraction ?? null, 2)}
              {current.final_coefficient_zero && (
                <span className="marker"> coefficient later refined to zero</span>
              )}
            </p>
          )
        )}
        <p className="pursuit-support">
          support:{" "}
          {trace.steps.slice(0, step).map((s) => (
            <code
              key={s.step}
              className={s.step === step ? "support-token support-token-new" : "support-token"}
            >
              {tokenLabel(s.added_label)}
            </code>
          ))}
          {step === 0 && <span className="muted">∅</span>}
        </p>
        <p className="muted small">
          stop reason: {trace.stop_reason} ·{" "}
          {trace.per_step_coefficients_available
            ? "per-step coefficients recorded"
            : "per-step coefficients were not recorded by this run — selection order and residual norms are exact; coefficient values are final-state only (see cone panel)"}
        </p>
      </div>
    </section>
  );
}
