import type { DataStatus } from "../types";

const LABELS: Record<DataStatus, string> = {
  measured: "Measured",
  imported: "Imported",
  synthetic_fixture: "Synthetic UI fixture",
};

export function StatusBadge({
  status,
  detail,
}: {
  status: DataStatus | null;
  detail?: string;
}) {
  if (status === null) {
    return (
      <span className="badge badge-none" role="status">
        No causal data available
      </span>
    );
  }
  return (
    <span className={`badge badge-${status}`} role="status" title={detail}>
      {LABELS[status]}
      {detail ? ` · ${detail}` : ""}
    </span>
  );
}
