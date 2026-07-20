/** Shared formatting helpers. */

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value !== 0 && Math.abs(value) < 10 ** -digits) {
    return value.toExponential(1);
  }
  return value.toFixed(digits);
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(100 * value).toFixed(digits)}%`;
}

export function shortSha(sha: string | null | undefined, length = 12): string {
  if (!sha) return "—";
  return sha.replace(/^sha256:/, "").slice(0, length);
}

/** Render a token label with visible whitespace so " is" and "is" differ. */
export function tokenLabel(label: string | null | undefined): string {
  if (label === null || label === undefined) return "—";
  return label.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}

export function formatRank(rank: number | null | undefined): string {
  if (rank === null || rank === undefined) return "not recorded";
  if (rank === 0) return "0 (exact top-1 match)";
  return rank.toLocaleString("en-US");
}
