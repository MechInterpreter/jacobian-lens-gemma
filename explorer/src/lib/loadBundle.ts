import type { CausalRecord, DataStatus, ExplorerBundle } from "../types";

export class BundleError extends Error {}

/** Structural validation of the parts the UI depends on. Not a full JSON
 * Schema check (that runs in the Python exporter); enough to reject
 * malformed files with a clear message instead of a blank screen. */
export function validateBundle(raw: unknown): ExplorerBundle {
  if (typeof raw !== "object" || raw === null) {
    throw new BundleError("bundle is not a JSON object");
  }
  const bundle = raw as Record<string, unknown>;
  if (bundle.schema !== "jlens.explorer.bundle.v1") {
    throw new BundleError(
      `unsupported bundle schema: ${String(bundle.schema ?? "(missing)")}`,
    );
  }
  const provenance = bundle.provenance as Record<string, unknown> | undefined;
  if (!provenance || typeof provenance.schema_version !== "string") {
    throw new BundleError("bundle has no provenance.schema_version");
  }
  if (!/^1\./.test(provenance.schema_version)) {
    throw new BundleError(
      `bundle schema_version ${provenance.schema_version} is not 1.x`,
    );
  }
  for (const section of [
    "examples",
    "layer_records",
    "cones",
    "pursuit_traces",
    "trajectories",
    "causal_records",
  ]) {
    if (!Array.isArray(bundle[section])) {
      throw new BundleError(`bundle section '${section}' is missing or not a list`);
    }
  }
  for (const example of bundle.examples as Record<string, unknown>[]) {
    if (typeof example.example_id !== "string" || typeof example.modality !== "string") {
      throw new BundleError("an example record lacks example_id/modality");
    }
  }
  for (const record of bundle.causal_records as CausalRecord[]) {
    if (!record.status) {
      throw new BundleError(
        `causal record ${record.condition_id ?? "?"} has no measured/fixture status`,
      );
    }
  }
  return raw as ExplorerBundle;
}

const statusRank: Record<DataStatus, number> = {
  measured: 0,
  imported: 1,
  synthetic_fixture: 2,
};

/** Client-side merge with the same semantics as the Python exporter's
 * merge_bundles: extra wins on ID collision, modalities union, bundle-level
 * data_status degrades to the weakest input. Per-record statuses are kept. */
export function mergeBundles(
  base: ExplorerBundle,
  extra: ExplorerBundle,
): ExplorerBundle {
  const bySection = <T>(items: T[], extraItems: T[], key: (item: T) => string): T[] => {
    const map = new Map<string, T>();
    for (const item of items) map.set(key(item), item);
    for (const item of extraItems) map.set(key(item), item);
    return [...map.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(([, v]) => v);
  };
  const coord = (r: { example_id: string; layer: number; position: number }) =>
    `${r.example_id}|${r.layer}|${r.position}`;
  const merged: ExplorerBundle = {
    ...base,
    examples: bySection(base.examples, extra.examples, (e) => e.example_id),
    layer_records: bySection(base.layer_records, extra.layer_records, coord),
    cones: bySection(base.cones, extra.cones, coord),
    pursuit_traces: bySection(base.pursuit_traces, extra.pursuit_traces, coord),
    trajectories: bySection(
      base.trajectories,
      extra.trajectories,
      (t) => `${t.example_id}|${t.position}|${t.layer_from}|${t.layer_to}`,
    ),
    causal_records: bySection(
      base.causal_records,
      extra.causal_records,
      (c) => c.condition_id,
    ),
    causal_baseline_parity:
      extra.causal_baseline_parity ?? base.causal_baseline_parity,
    provenance: {
      ...base.provenance,
      source_run_ids: [
        ...new Set([
          ...base.provenance.source_run_ids,
          ...extra.provenance.source_run_ids,
        ]),
      ].sort(),
      modalities_present: [
        ...new Set([
          ...base.provenance.modalities_present,
          ...extra.provenance.modalities_present,
        ]),
      ].sort() as ExplorerBundle["provenance"]["modalities_present"],
      merged_bundles: [
        ...(base.provenance.merged_bundles ?? []),
        extra.provenance as unknown as Record<string, unknown>,
      ],
      data_status:
        statusRank[extra.provenance.data_status] >
        statusRank[base.provenance.data_status]
          ? extra.provenance.data_status
          : base.provenance.data_status,
    },
  };
  return merged;
}

export interface LoadedData {
  bundle: ExplorerBundle;
  /** Which files were actually loaded, in merge order. */
  sources: { path: string; status: DataStatus }[];
  warnings: string[];
}

async function fetchBundle(path: string): Promise<ExplorerBundle | null> {
  let response: Response;
  try {
    response = await fetch(path);
  } catch {
    return null;
  }
  if (!response.ok) return null;
  const raw: unknown = await response.json();
  return validateBundle(raw);
}

/** Load the default bundle set.
 *
 * Measured bundles are preferred automatically: fixtures are only loaded for
 * a section when no measured bundle provides it. Paths are relative to the
 * deployed site root. */
export async function loadDefaultBundles(baseUrl = "data"): Promise<LoadedData> {
  const warnings: string[] = [];
  const sources: LoadedData["sources"] = [];

  const text = await fetchBundle(`${baseUrl}/text_demo.json`);
  if (!text) {
    throw new BundleError(
      `could not load ${baseUrl}/text_demo.json — run scripts/export_explorer_bundle.py first`,
    );
  }
  sources.push({ path: `${baseUrl}/text_demo.json`, status: text.provenance.data_status });
  let bundle = text;

  const preferMeasured = async (measuredPath: string, fixturePath: string, what: string) => {
    const measured = await fetchBundle(`${baseUrl}/${measuredPath}`);
    if (measured) {
      bundle = mergeBundles(bundle, measured);
      sources.push({
        path: `${baseUrl}/${measuredPath}`,
        status: measured.provenance.data_status,
      });
      return;
    }
    const fixture = await fetchBundle(`${baseUrl}/${fixturePath}`);
    if (fixture) {
      if (fixture.provenance.data_status !== "synthetic_fixture") {
        throw new BundleError(
          `${fixturePath} must declare data_status synthetic_fixture`,
        );
      }
      bundle = mergeBundles(bundle, fixture);
      sources.push({ path: `${baseUrl}/${fixturePath}`, status: "synthetic_fixture" });
      warnings.push(
        `${what}: no measured bundle found; showing clearly-labelled synthetic fixtures`,
      );
      return;
    }
    warnings.push(`${what}: no data available`);
  };

  await preferMeasured("measured/causal.json", "fixtures/causal_fixture.json", "causal steering");
  await preferMeasured(
    "measured/multimodal.json",
    "fixtures/multimodal_fixture.json",
    "image/audio capture",
  );

  return { bundle, sources, warnings };
}
