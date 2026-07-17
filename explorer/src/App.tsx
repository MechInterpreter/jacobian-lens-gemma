import { useEffect, useMemo, useState } from "react";
import type { ExplorerBundle } from "./types";
import { loadDefaultBundles, validateBundle, type LoadedData } from "./lib/loadBundle";
import { Header, modalityToTab, type ModalityTab } from "./components/Header";
import { ExampleBrowser } from "./components/ExampleBrowser";
import { InputViewer } from "./components/InputViewer";
import { LayerRail, summarizeLayers } from "./components/LayerRail";
import { PredictionPanel } from "./components/PredictionPanel";
import { ConePanel } from "./components/ConePanel";
import { PursuitPlayer } from "./components/PursuitPlayer";
import { TrajectoryView } from "./components/TrajectoryView";
import { CausalPanel } from "./components/CausalPanel";
import { ProvenancePanel } from "./components/ProvenancePanel";

type LoadState =
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "ready"; data: LoadedData };

export default function App({ preloaded }: { preloaded?: LoadedData }) {
  const [state, setState] = useState<LoadState>(
    preloaded ? { phase: "ready", data: preloaded } : { phase: "loading" },
  );
  const [modality, setModality] = useState<ModalityTab>("text");
  const [exampleId, setExampleId] = useState<string | null>(null);
  const [position, setPosition] = useState<number | null>(null);
  const [layer, setLayer] = useState<number | null>(null);

  useEffect(() => {
    if (preloaded) return;
    loadDefaultBundles()
      .then((data) => setState({ phase: "ready", data }))
      .catch((error: unknown) =>
        setState({
          phase: "error",
          message: error instanceof Error ? error.message : String(error),
        }),
      );
  }, [preloaded]);

  const bundle: ExplorerBundle | null =
    state.phase === "ready" ? state.data.bundle : null;

  const examplesForModality = useMemo(
    () =>
      (bundle?.examples ?? []).filter(
        (example) => modalityToTab(example.modality) === modality,
      ),
    [bundle, modality],
  );

  // Keep selection valid as modality / data changes.
  const example =
    examplesForModality.find((e) => e.example_id === exampleId) ??
    examplesForModality[0] ??
    null;
  const positions = example?.selected_positions ?? [];
  const activePosition =
    position !== null && positions.includes(position)
      ? position
      : (positions[positions.length - 1] ?? null);

  const layersAvailable = useMemo(() => {
    if (!bundle || !example) return [];
    return [
      ...new Set(
        bundle.layer_records
          .filter((r) => r.example_id === example.example_id)
          .map((r) => r.layer),
      ),
    ].sort((a, b) => a - b);
  }, [bundle, example]);
  const activeLayer =
    layer !== null && layersAvailable.includes(layer)
      ? layer
      : (layersAvailable[layersAvailable.length - 1] ?? null);

  if (state.phase === "loading") {
    return <main className="app-status">loading bundles…</main>;
  }
  if (state.phase === "error") {
    return (
      <main className="app-status app-error" role="alert">
        <h1>Gemma 4 Multimodal J-Lens Explorer</h1>
        <p>Could not load explorer data: {state.message}</p>
        <p className="muted">
          Export a bundle with <code>scripts/export_explorer_bundle.py</code> into{" "}
          <code>explorer/public/data/text_demo.json</code>, then reload.
        </p>
      </main>
    );
  }

  const { data } = state;
  const cones =
    bundle && example && activePosition !== null
      ? bundle.cones.filter(
          (c) => c.example_id === example.example_id && c.position === activePosition,
        )
      : [];
  const transitions =
    bundle && example && activePosition !== null
      ? bundle.trajectories.filter(
          (t) => t.example_id === example.example_id && t.position === activePosition,
        )
      : [];
  const layerRecord =
    bundle && example && activePosition !== null && activeLayer !== null
      ? (bundle.layer_records.find(
          (r) =>
            r.example_id === example.example_id &&
            r.position === activePosition &&
            r.layer === activeLayer,
        ) ?? null)
      : null;
  const cone = cones.find((c) => c.layer === activeLayer) ?? null;
  const trace =
    bundle && example && activePosition !== null && activeLayer !== null
      ? (bundle.pursuit_traces.find(
          (t) =>
            t.example_id === example.example_id &&
            t.position === activePosition &&
            t.layer === activeLayer,
        ) ?? null)
      : null;

  const availableTabs = new Set<ModalityTab>(
    (bundle?.examples ?? []).map((e) => modalityToTab(e.modality)),
  );

  return (
    <div className="app">
      <Header
        provenance={data.bundle.provenance}
        activeStatus={example?.data_status ?? null}
        modality={modality}
        onModalityChange={(tab) => {
          setModality(tab);
          setExampleId(null);
          setPosition(null);
        }}
        availableTabs={availableTabs}
      />
      <div className="app-body">
        <aside className="app-sidebar">
          <ExampleBrowser
            examples={data.bundle.examples}
            modality={modality}
            selectedId={example?.example_id ?? null}
            onSelect={(id) => {
              setExampleId(id);
              setPosition(null);
            }}
          />
          <ProvenancePanel
            provenance={data.bundle.provenance}
            layer={activeLayer}
            position={activePosition}
            warnings={data.warnings}
          />
        </aside>
        <main className="app-main">
          {!example ? (
            <section className="panel">
              <h2>No {modality} examples</h2>
              <p className="muted">
                {modality === "text"
                  ? "The text bundle is empty."
                  : `No ${modality}-conditioned records yet — run notebooks/gemma_4_e4b_multimodal_jlens_capture.ipynb and merge its bundle.`}
              </p>
            </section>
          ) : (
            <>
              <InputViewer
                example={example}
                selectedPosition={activePosition}
                onSelectPosition={setPosition}
              />
              <LayerRail
                summaries={summarizeLayers(layersAvailable, cones, transitions)}
                selectedLayer={activeLayer}
                onSelect={setLayer}
              />
              <div className="panel-row">
                <PredictionPanel
                  example={example}
                  record={layerRecord}
                  position={activePosition ?? 0}
                />
                <ConePanel cone={cone} />
              </div>
              <div className="panel-row">
                <PursuitPlayer trace={trace} />
                <TrajectoryView cones={cones} transitions={transitions} />
              </div>
              <CausalPanel
                records={data.bundle.causal_records}
                exampleId={example.example_id}
              />
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export { validateBundle };
