import type { BundleProvenance, DataStatus, Modality } from "../types";
import { shortSha } from "../lib/format";
import { StatusBadge } from "./Badge";

export type ModalityTab = "text" | "image" | "audio";

export function modalityToTab(modality: Modality): ModalityTab {
  if (modality.startsWith("image")) return "image";
  if (modality.startsWith("audio")) return "audio";
  return "text";
}

export function Header({
  provenance,
  activeStatus,
  modality,
  onModalityChange,
  availableTabs,
}: {
  provenance: BundleProvenance;
  /** Data status of the currently viewed example (falls back to the
   * bundle-level status when nothing is selected). */
  activeStatus: DataStatus | null;
  modality: ModalityTab;
  onModalityChange: (tab: ModalityTab) => void;
  availableTabs: Set<ModalityTab>;
}) {
  const tabs: ModalityTab[] = ["text", "image", "audio"];
  return (
    <header className="app-header">
      <div className="app-header-title">
        <h1>Gemma 4 Multimodal J-Lens Explorer</h1>
        <p className="app-header-sub">
          {provenance.model_repo_id} @ {shortSha(provenance.model_revision, 12)} ·
          run {provenance.source_run_ids.join(", ")}
        </p>
      </div>
      <div className="app-header-right">
        <StatusBadge
          status={activeStatus ?? provenance.data_status}
          detail="current example"
        />
        <nav className="modality-tabs" aria-label="Modality">
          {tabs.map((tab) => (
            <button
              key={tab}
              className={modality === tab ? "tab tab-active" : "tab"}
              aria-pressed={modality === tab}
              onClick={() => onModalityChange(tab)}
            >
              {tab === "text" ? "Text" : tab === "image" ? "Image" : "Audio"}
              {!availableTabs.has(tab) && <span className="tab-empty"> (no data)</span>}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
