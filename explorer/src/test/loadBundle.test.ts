import { describe, expect, it } from "vitest";
import { BundleError, mergeBundles, validateBundle } from "../lib/loadBundle";
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
