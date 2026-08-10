# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The resumable preprocessing stage of the L32 convergence-resolution study.

What these tests are defending. Section 8a of the resolution notebook used to
walk every file of every completed run twice — once for a tree digest, once for
identities — with the whole result in Python memory. On a Drive mount that took
more than four hours and an interruption threw all of it away.

The replacement has to earn three claims, and each of them is only worth
anything if it is checked the hard way:

* **resumability**, checked across two genuinely separate interpreters rather
  than two calls in one kernel (:mod:`tests._prep_cache_harness`);
* **completeness**, checked against each completed run's own recorded
  population, including a fixture where the minimal sources really are
  insufficient and the fallback really is needed;
* **equivalence**, checked against the legacy whole-tree harvester on
  real-shaped fixtures — the optimization is only allowed if it changes nothing
  about which media are excluded.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jlens.mmpilot import prep_cache as prep
from jlens.mmpilot.l32_resolution import (
    RESOLUTION_FINGERPRINT_FIELDS,
    ExclusionSet,
    harvest_excluded_identities,
    run_tree_digest,
)
from jlens.mmpilot.mock import build_mock_completed_run
from jlens.mmpilot.store import payload_checksum
from tests._prep_cache_harness import build_fingerprint

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "_prep_cache_harness.py"
BUILDER = REPO_ROOT / "scripts" / "_build_l32_resolution_notebook.py"


# ------------------------------------------------------------------ fixtures


def _groups(n: int, *, prefix: str = "g") -> list[dict]:
    """Synchronized groups shaped like the real manifest's: two per photograph."""
    return [
        {
            "group_id": f"{prefix}{index:04d}",
            "image_id": f"img{index // 2:04d}",
            "audio_path": f"/audio/{prefix}{index:04d}.wav",
            "caption": f"a photograph number {index}",
            "split": "train" if index % 2 else "test",
        }
        for index in range(n)
    ]


def _completed_run(root: Path, groups, *, name: str = "mml32_l32_followup_20260808T182717"):
    run = root / name
    build_mock_completed_run(run, groups, layer=1)
    return run


@pytest.fixture
def run_dir(tmp_path):
    return _completed_run(tmp_path, _groups(24))


@pytest.fixture
def fingerprint(run_dir):
    return build_fingerprint([run_dir])


def _harness(cache, runs, **options) -> dict:
    arguments = [
        sys.executable,
        str(HARNESS),
        "--cache",
        str(cache),
        "--runs",
        os.pathsep.join(str(path) for path in runs),
    ]
    for name, value in options.items():
        flag = "--" + name.replace("_", "-")
        if value is True:
            arguments.append(flag)
        elif value is not None and value is not False:
            arguments += [flag, str(value)]
    result = subprocess.run(arguments, capture_output=True, text=True)
    assert result.stdout.strip(), result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _prepare(cache, runs, **options) -> dict:
    payload = _harness(cache, runs, **options)
    assert payload["ok"], payload
    return payload


def _shards(cache: Path, stage: str = "harvest_minimal") -> list[Path]:
    return sorted((cache / stage / "shards").glob("*.json.gz"))


# ========================================== 1-3. interruption and resumption


def test_an_interrupted_harvest_resumes_in_a_fresh_process(tmp_path, run_dir):
    """The second interpreter picks up at the last durable checkpoint.

    The abort lands *inside* a batch, which is the case that matters: a shard
    that was never written must be redone and a shard that was written must not.
    """
    cache = tmp_path / "cache"
    aborted = _harness(cache, [run_dir], batch=10, abort_after=25)
    assert aborted["aborted"] is True

    state = json.loads(
        (cache / "harvest_minimal" / "harvest_state.json").read_text(encoding="utf-8")
    )
    assert state["cursor"] == 20, "two full batches must have been committed"
    assert len(state["shards"]) == 2
    assert state["complete"] is False

    resumed = _prepare(cache, [run_dir], batch=10)
    assert resumed["files_reused"] == 20
    assert resumed["cursor"] == resumed["n_files_read"]
    assert resumed["complete"] is True


def test_completed_shards_are_never_recomputed(tmp_path, run_dir):
    cache = tmp_path / "cache"
    _harness(cache, [run_dir], batch=10, abort_after=25)
    before = {path.name: path.read_bytes() for path in _shards(cache)}
    assert before, "the interrupted session must have left durable checkpoints"

    _prepare(cache, [run_dir], batch=10)
    after = {path.name: path.read_bytes() for path in _shards(cache)}
    for name, payload in before.items():
        assert after[name] == payload, f"{name} was rewritten"


def test_only_the_single_in_flight_unit_may_be_repeated(tmp_path, run_dir):
    cache = tmp_path / "cache"
    aborted = _harness(cache, [run_dir], batch=10, abort_after=25)
    computed_before = sum(
        1 for line in aborted["progress_lines"] if "work=computed" in line
    )
    resumed = _prepare(cache, [run_dir], batch=10)
    total = resumed["n_files_read"]
    repeated = computed_before + resumed["files_computed"] - total
    assert 0 <= repeated <= 10, (
        f"{repeated} file(s) were repeated; at most one bounded unit of 10 may be"
    )


def test_an_uninterrupted_second_pass_reads_nothing(tmp_path, run_dir):
    cache = tmp_path / "cache"
    first = _prepare(cache, [run_dir], batch=10)
    second = _prepare(cache, [run_dir], batch=10)
    assert second["files_computed"] == 0
    assert second["files_reused"] == first["n_files_read"]
    assert second["exclusion_digest"] == first["exclusion_digest"]


# ================================================== 4. torn shard quarantine


def test_a_torn_shard_is_detected_and_quarantined(tmp_path, run_dir):
    """A shard that fails its own checksum is never treated as data."""
    cache = tmp_path / "cache"
    pristine = _prepare(cache, [run_dir], batch=10)
    shards = _shards(cache)
    assert len(shards) >= 3
    shards[1].write_bytes(gzip.compress(b'{"schema": "torn"}'))

    resumed = _prepare(cache, [run_dir], batch=10)
    assert resumed["quarantined"], "the torn shard was not recorded"
    assert (cache / "harvest_minimal" / "quarantine").is_dir()
    assert resumed["exclusion_digest"] == pristine["exclusion_digest"], (
        "recovering from a torn shard must reproduce the same exclusion set"
    )
    # Only the torn shard's own batch and what followed it is recomputed; the
    # shards before it survive.
    assert resumed["files_computed"] < resumed["n_files_read"]


def test_a_torn_shard_can_be_refused_instead_of_quarantined(tmp_path, run_dir):
    cache = tmp_path / "cache"
    _prepare(cache, [run_dir], batch=10)
    _shards(cache)[1].write_bytes(b"not gzip at all")
    payload = _harness(cache, [run_dir], batch=10, on_corrupt="refuse")
    assert payload["ok"] is False
    assert "CorruptShard" in payload["error"]


def test_read_shard_refuses_a_body_that_does_not_match_its_checksum(tmp_path):
    path = tmp_path / "shard_00000.json.gz"
    body = {"schema": prep.SHARD_SCHEMA, "cursor_start": 0, "cursor_end": 1}
    path.write_bytes(
        gzip.compress(
            json.dumps({**body, "shard_checksum": "sha256:wrong"}).encode("utf-8")
        )
    )
    with pytest.raises(prep.CorruptShard):
        prep.read_shard(path)


# ============================================ 5-6. what invalidates a cache


def test_a_changed_completed_run_fingerprint_changes_the_cache(tmp_path, run_dir):
    """A different completed run is a different preparation, not a resume."""
    before = build_fingerprint([run_dir])
    (run_dir / "fingerprint.json").write_text(
        json.dumps({"fingerprint_digest": "sha256:moved"}), encoding="utf-8"
    )
    after = build_fingerprint([run_dir])
    assert before["preparation_digest"] != after["preparation_digest"]
    assert prep.preparation_cache_dir(tmp_path, before) != prep.preparation_cache_dir(
        tmp_path, after
    )


def test_a_changed_source_file_refuses_the_resume_rather_than_mixing(
    tmp_path, run_dir
):
    """A new or altered identity-bearing file voids the harvest state."""
    cache = tmp_path / "cache"
    _harness(cache, [run_dir], batch=10, abort_after=25)
    extra = run_dir / "units" / "activation" / "zzz_extra.json"
    extra.write_text(
        json.dumps({"payload": {"group_id": "gNEW", "image_id": "imgNEW"}}),
        encoding="utf-8",
    )
    payload = _harness(cache, [run_dir], batch=10)
    assert payload["ok"] is False
    assert "PreparationIncompatible" in payload["error"]
    assert "inventory_digest" in payload["error"]
    assert "zzz_extra.json" in payload["error"]


def test_a_changed_source_file_invalidates_only_its_own_preparation(
    tmp_path, run_dir
):
    """One void preparation does not void an unrelated one.

    The second preparation differs only in its selection seed, so it has its own
    cache directory and its own inventory — and it is unaffected by the first
    one's state being void.
    """
    voided = tmp_path / "voided"
    _harness(voided, [run_dir], batch=10, abort_after=25)
    (run_dir / "units" / "activation" / "zzz_extra.json").write_text(
        json.dumps({"payload": {"group_id": "gNEW"}}), encoding="utf-8"
    )
    assert _harness(voided, [run_dir], batch=10)["ok"] is False

    other = tmp_path / "other"
    assert _prepare(other, [run_dir], batch=10, salt="_different")["complete"] is True


def test_a_content_only_edit_is_caught_by_the_harvest_time_digests(
    tmp_path, run_dir
):
    """Same size, same mtime, different bytes — invisible to the old scan.

    The whole-tree digest this replaced compared names, sizes and mtimes only.
    A content digest taken during the harvest read catches an edit that
    preserves both.
    """
    cache = tmp_path / "cache"
    _prepare(cache, [run_dir], batch=10)
    inventory = json.loads(
        (cache / "harvest_minimal" / "source_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    reduction = prep.reduce_shards([cache / "harvest_minimal"])
    families = {row["run"]: prep.families_used(row) for row in
                json.loads((cache / "source_plan.json").read_text(encoding="utf-8"))["runs"]}

    victim = run_dir / inventory["entries"][-1]["relpath"]
    stat = victim.stat()
    original = victim.read_bytes()
    victim.write_bytes(original[:-1] + b" ")
    os.utime(victim, (stat.st_atime, stat.st_mtime))
    assert victim.stat().st_size == stat.st_size

    unaware = prep.verify_sources_unchanged(
        [run_dir], inventory, families, rehash=False
    )
    assert unaware["unchanged"] is True, "size and mtime cannot see this edit"

    aware = prep.verify_sources_unchanged(
        [run_dir],
        inventory,
        families,
        rehash=True,
        file_checksums=reduction["file_checksums"],
    )
    assert aware["unchanged"] is False
    assert aware["content_mismatches"]
    with pytest.raises(prep.PreparationRefused):
        prep.assert_sources_unchanged(aware)


def test_a_completed_cache_that_disagrees_with_itself_refuses(tmp_path, run_dir):
    cache = tmp_path / "cache"
    _prepare(cache, [run_dir], batch=10, finalize=True)
    marker = json.loads(
        (cache / "preparation_complete.json").read_text(encoding="utf-8")
    )
    with pytest.raises(prep.PreparationIncompatible):
        prep.finalize_preparation(
            cache, {**marker, "exclusion_digest": "sha256:different"}
        )


def test_a_tampered_exclusion_set_is_refused_on_load(tmp_path, run_dir):
    cache = tmp_path / "cache"
    _prepare(cache, [run_dir], batch=10)
    path = cache / "exclusion_set.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["image_ids"] = payload["image_ids"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(prep.PreparationRefused):
        prep.load_exclusion_set(cache)


# ================================================== 7-8. determinism, parity


def test_the_same_inputs_produce_the_same_exclusion_digest(tmp_path, run_dir):
    one = _prepare(tmp_path / "a", [run_dir], batch=7)
    two = _prepare(tmp_path / "b", [run_dir], batch=13)
    assert one["exclusion_digest"] == two["exclusion_digest"]
    assert one["exclusion_counts"] == two["exclusion_counts"]


def test_the_optimized_harvest_equals_the_legacy_full_harvester(tmp_path, run_dir):
    """The optimization is only allowed if it excludes exactly the same media."""
    legacy = harvest_excluded_identities([run_dir], require=True)
    optimized = _prepare(tmp_path / "cache", [run_dir], batch=9)
    assert optimized["exclusion_digest"] == legacy.digest
    assert optimized["exclusion_counts"] == legacy.counts()


def test_the_optimized_harvest_matches_the_legacy_one_on_two_runs(tmp_path):
    """Two completed runs of different shapes, as the real study has."""
    followup = _completed_run(tmp_path, _groups(20))
    audio = _completed_run(
        tmp_path,
        _groups(14, prefix="a"),
        name="mmaudio_native_audio_transfer_20260806T144822",
    )
    legacy = harvest_excluded_identities([followup, audio], require=True)
    optimized = _prepare(tmp_path / "cache", [followup, audio], batch=11)
    assert optimized["exclusion_digest"] == legacy.digest
    assert optimized["exclusion_counts"] == legacy.counts()


def test_the_minimal_harvest_skips_the_redundant_unit_families(tmp_path, run_dir):
    optimized = _prepare(tmp_path / "cache", [run_dir], batch=25)
    skipped = optimized["runs"][0]["families_skipped"]
    assert "capability" in skipped
    assert "intervention" in skipped
    assert set(optimized["files_by_family"]) <= {
        "activation",
        "run_documents",
        "population_manifest",
    }
    assert optimized["fallback_required"] is False


def test_capability_units_are_never_treated_as_a_population_source(tmp_path):
    plan = prep.plan_sources([_completed_run(tmp_path, _groups(8))])
    assert "capability" not in plan["runs"][0]["minimal_families"]
    assert "capability" in plan["capability_units_are_not_a_population_source"]


# ========================================== 9-10. completeness and fallback


@pytest.fixture
def run_with_a_hidden_identity(tmp_path):
    """A completed run whose activation units do NOT cover the population.

    One group's only trace is a capability unit — the family the minimal
    strategy deliberately skips — while ``split_provenance.json`` still says the
    run spent every group. That is the shape a real pipeline change could take,
    and the completeness proof has to catch it rather than quietly excluding one
    photograph fewer.
    """
    groups = _groups(12)
    run = _completed_run(tmp_path, groups[:-1])
    hidden = groups[-1]
    capability = run / "units" / "capability" / "hidden.json"
    capability.parent.mkdir(parents=True, exist_ok=True)
    capability.write_text(
        json.dumps(
            {
                "schema": "jlens.mmpilot.unit.v1",
                "stage": "capability",
                "payload": {
                    "group_id": hidden["group_id"],
                    "image_id": hidden["image_id"],
                    "sample_id": f"{hidden['group_id']}:text",
                },
            }
        ),
        encoding="utf-8",
    )
    provenance = json.loads(
        (run / "split_provenance.json").read_text(encoding="utf-8")
    )
    provenance["n_groups"] = len(groups)
    provenance["n_distinct_images"] = len({g["image_id"] for g in groups})
    (run / "split_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    # The fingerprint's sample-id lists would otherwise leak the hidden group
    # back in through a bulk document.
    (run / "fingerprint.json").write_text(
        json.dumps({"fingerprint_digest": "sha256:mock-completed-followup"}),
        encoding="utf-8",
    )
    return run, hidden


def test_the_completeness_proof_fails_when_an_identity_hides_in_a_skipped_family(
    tmp_path, run_with_a_hidden_identity
):
    run, hidden = run_with_a_hidden_identity
    payload = _prepare(tmp_path / "cache", [run], batch=9, no_fallback=True)
    assert payload["complete"] is False, (
        "the minimal sources missed a group and said so"
    )
    row = payload["runs"][0]
    assert row["group_ids_recovered"] < row["expected_group_count"]
    assert row["shortfall"]["group_ids"] == 1
    assert "capability" in row["families_skipped"]

    exclusion = prep.load_exclusion_set(Path(payload["cache_dir"]))
    assert hidden["group_id"] not in exclusion.group_ids

    with pytest.raises(prep.CompletenessNotProven):
        prep.assert_complete(
            {"complete": False, "runs": [{**row, "complete": False}]}
        )


def test_the_fallback_scan_recovers_the_hidden_identity(
    tmp_path, run_with_a_hidden_identity
):
    run, hidden = run_with_a_hidden_identity
    payload = _prepare(tmp_path / "cache", [run], batch=9)
    assert payload["complete"] is True
    assert payload["fallback_required"] is True
    assert "capability" in payload["files_by_family"]

    exclusion = prep.load_exclusion_set(Path(payload["cache_dir"]))
    assert hidden["group_id"] in exclusion.group_ids
    assert hidden["image_id"] in exclusion.image_ids
    legacy = harvest_excluded_identities([run], require=True)
    assert exclusion.digest == legacy.digest


def test_the_fallback_scan_is_itself_resumable(
    tmp_path, run_with_a_hidden_identity
):
    run, _hidden = run_with_a_hidden_identity
    cache = tmp_path / "cache"
    # The abort has to land inside the *fallback* stage, so it is placed a few
    # files past the end of the minimal one rather than at a guessed number.
    plan = prep.plan_sources([run])
    minimal = prep.build_source_inventory(
        [run], {row["run"]: row["minimal_families"] for row in plan["runs"]}
    )
    aborted = _harness(
        cache, [run], batch=4, abort_after=minimal["n_files"] + 6
    )
    assert aborted["aborted"] is True
    assert (cache / "harvest_fallback" / "harvest_state.json").is_file()

    resumed = _prepare(cache, [run], batch=4)
    assert resumed["complete"] is True
    assert resumed["fallback_required"] is True
    assert resumed["files_reused"] >= minimal["n_files"]


def test_an_unanchored_run_is_never_called_complete(tmp_path):
    """No recorded population means no proof, whatever was recovered."""
    run = tmp_path / "mml32_l32_followup_unanchored"
    build_mock_completed_run(run, _groups(6), layer=1, write_split_provenance=False)
    plan = prep.plan_sources([run])
    assert plan["runs"][0]["expected"]["n_groups"] is None
    payload = _prepare(tmp_path / "cache", [run], batch=5)
    assert payload["complete"] is False
    assert payload["runs"][0]["anchored"] is False
    with pytest.raises(prep.CompletenessNotProven):
        prep.assert_complete(
            {"complete": False, "runs": [{**payload["runs"][0], "complete": False}]}
        )


def test_a_bulk_population_manifest_makes_the_activation_scan_unnecessary(tmp_path):
    """The preferred source: one file that enumerates every used group."""
    groups = _groups(10)
    run = _completed_run(tmp_path, groups)
    (run / "population_manifest.json").write_text(
        json.dumps(
            {
                "schema": "jlens.mmpilot.l32_resolution_population_manifest.v1",
                "units": [
                    {
                        "group_id": group["group_id"],
                        "image_id": group["image_id"],
                        "audio_path": group["audio_path"],
                        "caption": group["caption"],
                    }
                    for group in groups
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = prep.plan_sources([run])
    assert plan["runs"][0]["strategy"] == "bulk_population_manifest"
    assert "activation" not in plan["runs"][0]["minimal_families"]
    payload = _prepare(tmp_path / "cache", [run], batch=5)
    assert payload["complete"] is True
    assert "activation" not in payload["files_by_family"]


# =========================================== 11-13. read-only and reuse


def test_no_completed_run_is_modified(tmp_path, run_dir):
    before = run_tree_digest(run_dir)
    _prepare(tmp_path / "cache", [run_dir], batch=6, finalize=True)
    assert run_tree_digest(run_dir)["tree_digest"] == before["tree_digest"]


def test_a_write_into_a_completed_run_namespace_is_refused(tmp_path):
    protected = ("mml32_l32_followup", "mmaudio_")
    with pytest.raises(prep.PreparationRefused):
        prep.assert_write_allowed(
            tmp_path / "mml32_l32_followup_20260808T182717" / "x.json",
            protected_prefixes=protected,
        )
    with pytest.raises(prep.PreparationRefused):
        prep.atomic_write_json(
            tmp_path / "mmaudio_run" / "nested" / "x.json",
            {},
            protected_prefixes=protected,
        )
    assert prep.atomic_write_json(
        tmp_path / "derived" / "x.json", {"ok": True}, protected_prefixes=protected
    ).is_file()


def test_a_completed_preparation_reads_no_source_unit_at_all(tmp_path, run_dir):
    """Proven by taking the units away, not by counting reads.

    Renaming ``units/`` is the strongest available statement of "nothing under
    here was opened": if the reload path touched a single one it would fail.
    """
    cache = tmp_path / "cache"
    first = _prepare(cache, [run_dir], batch=8, finalize=True)
    os.rename(run_dir / "units", run_dir / "units_moved_away")

    second = _prepare(cache, [run_dir], batch=8)
    assert second["files_computed"] == 0
    assert second["exclusion_digest"] == first["exclusion_digest"]
    assert second["reused_complete_cache"] is True


def test_the_reloaded_exclusion_set_is_verified_against_its_shards(
    tmp_path, run_dir
):
    cache = tmp_path / "cache"
    _prepare(cache, [run_dir], batch=8, finalize=True)
    path = cache / "exclusion_set.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["group_ids"] = payload["group_ids"][:-2]
    payload["exclusion_digest"] = ExclusionSet(
        image_ids=set(payload["image_ids"]),
        group_ids=set(payload["group_ids"]),
        sample_ids=set(payload["sample_ids"]),
        audio_paths=set(payload["audio_paths"]),
        captions=set(payload["captions"]),
        media_checksums=set(payload["media_checksums"]),
        run_dirs=list(payload["run_dirs"]),
    ).digest
    path.write_text(json.dumps(payload), encoding="utf-8")

    payload = _harness(cache, [run_dir], batch=8)
    assert payload["ok"] is False
    assert "PreparationIncompatible" in payload["error"]


# ================================================== 14. progress reporting


def test_progress_is_printed_during_a_long_scan(tmp_path):
    """No 30-second silence, checked on a synthetic clock rather than by waiting."""
    run = _completed_run(tmp_path, _groups(60))
    lines: list[str] = []
    ticks = {"t": 0.0}

    def clock():
        ticks["t"] += 5.0  # five simulated seconds per clock read
        return ticks["t"]

    reporter = prep.ProgressReporter(
        interval=30.0, printer=lines.append, clock=clock
    )
    record = prep.run_exclusion_preparation(
        tmp_path / "cache",
        [run],
        fingerprint=build_fingerprint([run]),
        batch_files=25,
        progress=reporter,
    )
    file_lines = [line for line in lines if "work=computed" in line]
    assert len(file_lines) >= 3, lines
    for line in file_lines:
        assert "elapsed" in line and "eta" in line
        assert "run=" in line and "family=" in line and "shard=" in line
        assert "new_ids=" in line
    assert record["completeness"]["complete"] is True


def test_a_resume_announces_itself_prominently(tmp_path, run_dir):
    cache = tmp_path / "cache"
    _harness(cache, [run_dir], batch=10, abort_after=25)
    resumed = _prepare(cache, [run_dir], batch=10)
    banner = "\n".join(resumed["progress_lines"])
    assert "RESUMING PREPROCESSING" in banner
    assert "completed shards" in banner
    assert "remaining files" in banner
    assert "last durable checkpoint" in banner
    assert "reused from Drive" in banner


def test_a_complete_cache_says_it_was_reused(tmp_path, run_dir):
    cache = tmp_path / "cache"
    _prepare(cache, [run_dir], batch=10, finalize=True)
    again = _prepare(cache, [run_dir], batch=10)
    banner = "\n".join(again["progress_lines"])
    assert "PREPROCESSING REUSED FROM DRIVE — no source unit was read" in banner


def test_the_reporter_stays_quiet_between_intervals(tmp_path):
    lines: list[str] = []
    now = {"t": 0.0}
    reporter = prep.ProgressReporter(
        interval=30.0, printer=lines.append, clock=lambda: now["t"]
    )
    reporter.begin("start")
    assert reporter.tick(done=1, total=100) is False
    now["t"] = 31.0
    assert reporter.tick(done=2, total=100) is True


# =============================== 22. every preparation input moves the digest


@pytest.mark.parametrize("field", prep.PREPARATION_FINGERPRINT_FIELDS)
def test_changing_any_preparation_input_changes_the_digest(field, tmp_path, run_dir):
    base = build_fingerprint([run_dir])
    changed = prep.preparation_fingerprint(
        **{
            **{k: v for k, v in base.items() if k != "preparation_digest"},
            field: "moved",
        }
    )
    assert changed["preparation_digest"] != base["preparation_digest"]


def test_a_missing_preparation_field_refuses(tmp_path, run_dir):
    base = {k: v for k, v in build_fingerprint([run_dir]).items()
            if k != "preparation_digest"}
    base.pop("selection_seed")
    with pytest.raises(prep.PreparationRefused):
        prep.preparation_fingerprint(**base)


def test_an_unknown_preparation_field_refuses(tmp_path, run_dir):
    base = {k: v for k, v in build_fingerprint([run_dir]).items()
            if k != "preparation_digest"}
    with pytest.raises(prep.PreparationRefused):
        prep.preparation_fingerprint(**base, extra="x")


def test_the_preparation_is_bound_into_the_scientific_fingerprint():
    for name in (
        "preparation_version",
        "preparation_digest",
        "exclusion_completeness_digest",
        "independent_pool_digest",
        "concept_ranking_digest",
        "frozen_concept_feasibility_digest",
    ):
        assert name in RESOLUTION_FINGERPRINT_FIELDS, name


# ============================================ the configured real paths


def _builder_source() -> str:
    return BUILDER.read_text(encoding="utf-8")


def test_the_configured_completed_run_paths_are_the_real_ones():
    source = _builder_source()
    assert (
        '"/content/drive/MyDrive/jacobian-lens-gemma/runs/"\n'
        '    "mml32_l32_followup_20260808T182717"' in source
    )
    assert (
        '"/content/drive/MyDrive/jacobian-lens-gemma/runs/"\n'
        '    "mmaudio_native_audio_transfer_20260806T144822"' in source
    )


def test_the_expanded_manifest_is_derived_from_the_completed_l32_run():
    source = _builder_source()
    assert 'f"{COMPLETED_RUN_DIRS[0]}/expanded_manifest.json"' in source
    assert "jlens_mmpilot_v1" not in source


def test_the_preparation_cache_root_is_configured_and_deterministic(tmp_path, run_dir):
    source = _builder_source()
    assert (
        'PREP_CACHE_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco_derived"'
        in source
    )
    fingerprint = build_fingerprint([run_dir])
    cache = prep.preparation_cache_dir(
        "/content/drive/MyDrive/datasets/cstf_spokencoco_derived", fingerprint
    )
    assert prep.PREP_CACHE_NAMESPACE in cache.parts
    assert cache.name.startswith("prep_")
    assert cache == prep.preparation_cache_dir(
        "/content/drive/MyDrive/datasets/cstf_spokencoco_derived", fingerprint
    )
    # A preparation cache is never inside a run namespace.
    assert "runs" not in cache.parts


def test_the_cache_directory_keeps_paths_inside_the_windows_limit(tmp_path, run_dir):
    cache = prep.preparation_cache_dir(
        "/content/drive/MyDrive/datasets/cstf_spokencoco_derived",
        build_fingerprint([run_dir]),
    )
    deepest = cache / "harvest_fallback" / "shards" / "shard_00000.json.gz.tmp.999999"
    assert len(str(deepest)) < 260, str(deepest)


# ================================================= reporting and artifacts


def test_the_preprocessing_report_states_what_was_proven(tmp_path, run_dir):
    record = prep.run_exclusion_preparation(
        tmp_path / "cache",
        [run_dir],
        fingerprint=build_fingerprint([run_dir]),
        batch_files=8,
    )
    report = prep.render_preprocessing_report(
        {
            "cache_dir": str(tmp_path / "cache"),
            "preparation_digest": "sha256:x",
            "exclusion_digest": record["exclusion"].digest,
            "completeness": record["completeness"],
            "files_computed_this_session": record["files_computed_this_session"],
            "files_reused_from_drive": record["files_reused_from_drive"],
        }
    )
    assert "Checkpoint and resume semantics" in report
    assert "at most the single in-flight unit" in report
    assert "completeness proof" in report.lower()
    assert "no model output" in report
    assert run_dir.name in report


def test_the_shards_are_gzip_and_carry_their_own_checksums(tmp_path, run_dir):
    cache = tmp_path / "cache"
    _prepare(cache, [run_dir], batch=6)
    for path in _shards(cache):
        record = prep.read_shard(path)
        assert record["schema"] == prep.SHARD_SCHEMA
        body = {k: v for k, v in record.items() if k != "shard_checksum"}
        assert payload_checksum(body) == record["shard_checksum"]
        assert record["cursor_end"] > record["cursor_start"]


def test_shard_cursors_tile_the_inventory_exactly(tmp_path, run_dir):
    cache = tmp_path / "cache"
    payload = _prepare(cache, [run_dir], batch=7)
    state = json.loads(
        (cache / "harvest_minimal" / "harvest_state.json").read_text(encoding="utf-8")
    )
    cursor = 0
    for shard in state["shards"]:
        assert shard["cursor_start"] == cursor
        cursor = shard["cursor_end"]
    assert cursor == state["n_files_total"] == payload["n_files_read"]


def test_the_prepared_selection_round_trips(tmp_path, run_dir):
    cache = tmp_path / "cache"
    _prepare(cache, [run_dir], batch=8)
    prep.save_prepared_selection(cache, {"pool_digest": "sha256:pool", "ranking": []})
    loaded = prep.load_prepared_selection(cache)
    assert loaded["pool_digest"] == "sha256:pool"
    assert loaded["schema"] == prep.SELECTION_SCHEMA


def test_an_absent_prepared_selection_is_none_not_an_error(tmp_path):
    assert prep.load_prepared_selection(tmp_path) is None
    assert prep.preparation_is_complete(tmp_path) is None


def test_an_unreadable_source_file_is_recorded_and_never_silently_dropped(
    tmp_path, run_dir
):
    victim = next((run_dir / "units" / "activation").glob("*.json"))
    victim.write_text("{not json", encoding="utf-8")
    record = prep.run_exclusion_preparation(
        tmp_path / "cache",
        [run_dir],
        fingerprint=build_fingerprint([run_dir]),
        batch_files=8,
        allow_fallback=False,
    )
    assert record["reduction"]["unreadable"], "the torn unit was not recorded"
    assert record["completeness"]["n_missing_or_invalid_units"] >= 1


def test_shutil_is_not_needed_to_clean_a_quarantine(tmp_path, run_dir):
    """Quarantined shards stay on disk for inspection rather than being deleted."""
    cache = tmp_path / "cache"
    _prepare(cache, [run_dir], batch=6)
    _shards(cache)[0].write_bytes(b"torn")
    _prepare(cache, [run_dir], batch=6)
    quarantine = cache / "harvest_minimal" / "quarantine"
    assert quarantine.is_dir()
    assert list(quarantine.iterdir())
    shutil.rmtree(quarantine)
