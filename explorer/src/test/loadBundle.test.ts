import { afterEach, describe, expect, it, vi } from "vitest";
import {
  BundleError,
  defaultDataBaseUrl,
  loadDefaultBundles,
  mergeBundles,
  validateBundle,
} from "../lib/loadBundle";
import { testBundle } from "./fixture";

describe("validateBundle", () => {
  it("accepts a structurally valid bundle", () => {
    expect(validateBundle(testBundle())).toBeTruthy();
  });

  it("rejects a non-object", () => {
    expect(() => validateBundle("nope")).toThrow(BundleError);
  });

  it("rejects a wrong schema tag", () => {
    const bad = { ...testBundle(), schema: "something.else" };
    expect(() => validateBundle(bad)).toThrow(/unsupported bundle schema/);
  });

  it("rejects a missing section", () => {
    const bundle = testBundle() as unknown as Record<string, unknown>;
    delete bundle.cones;
    expect(() => validateBundle(bundle)).toThrow(/section 'cones'/);
  });

  it("rejects an incompatible schema_version", () => {
    const bundle = testBundle();
    bundle.provenance.schema_version = "2.0.0";
    expect(() => validateBundle(bundle)).toThrow(/not 1\.x/);
  });

  it("rejects causal records without a status", () => {
    const bundle = testBundle();
    // @ts-expect-error deliberately malformed
    bundle.causal_records[0].status = undefined;
    expect(() => validateBundle(bundle)).toThrow(/measured\/fixture status/);
  });
});

describe("defaultDataBaseUrl (local dev vs production GitHub Pages base)", () => {
  const originalBaseUrl = import.meta.env.BASE_URL;

  afterEach(() => {
    import.meta.env.BASE_URL = originalBaseUrl;
  });

  it("requests the site root in local development (Vite dev serves BASE_URL '/')", () => {
    import.meta.env.BASE_URL = "/";
    expect(defaultDataBaseUrl()).toBe("/data");
  });

  it("requests under the deployed subpath for a relative production build", () => {
    // vite.config.ts uses `base: "./"`; a client production build resolves
    // BASE_URL to the literal string "./" (relative to the deployed HTML
    // document), which is "/jacobian-lens-gemma/" once served from GitHub
    // Pages at that subpath.
    import.meta.env.BASE_URL = "./";
    expect(defaultDataBaseUrl()).toBe("./data");
  });

  it("requests under an explicit absolute deploy base", () => {
    import.meta.env.BASE_URL = "/jacobian-lens-gemma/";
    expect(defaultDataBaseUrl()).toBe("/jacobian-lens-gemma/data");
  });
});

describe("fetchBundle degradation on a 200-with-HTML response", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("treats a static-host not-found fallback (200 OK + HTML body) as missing, not a crash", async () => {
    const text = testBundle();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).endsWith("text_demo.json")) {
          return { ok: true, json: async () => text } as Response;
        }
        // Simulates Vite's dev-server SPA fallback / GitHub Pages'
        // 404->index.html rewrite: status 200, but the body is the app's
        // own index.html, not the requested JSON file.
        return {
          ok: true,
          json: async () => {
            throw new SyntaxError("Unexpected token '<'");
          },
        } as unknown as Response;
      }),
    );
    const data = await loadDefaultBundles("data");
    expect(data.bundle.examples.length).toBeGreaterThan(0);
    expect(data.warnings.some((w) => w.includes("no data available"))).toBe(
      true,
    );
  });
});

describe("mergeBundles", () => {
  it("unions modalities, dedupes by id, and degrades status", () => {
    const base = testBundle();
    const extra = testBundle();
    extra.provenance.data_status = "synthetic_fixture";
    extra.provenance.source_run_ids = ["fixture_run"];
    extra.causal_records = [
      { ...extra.causal_records[0], condition_id: "cond_new" },
    ];
    const merged = mergeBundles(base, extra);
    expect(merged.provenance.data_status).toBe("synthetic_fixture");
    expect(merged.provenance.source_run_ids).toContain("fixture_run");
    expect(merged.provenance.source_run_ids).toContain("jspace_test");
    const ids = merged.causal_records.map((r) => r.condition_id);
    expect(ids).toContain("cond_new");
    expect(ids).toContain("cond_target_plus");
    // Examples dedupe by example_id (same in both copies).
    expect(merged.examples.length).toBe(base.examples.length);
  });
});
