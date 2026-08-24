"""Execute the notebook's REAL_MODE follow-up stages against a fake model.

Why this file exists
--------------------
``test_multimodal_lens_notebook.test_mock_notebook_executes_end_to_end`` runs
the notebook with ``REAL_MODE = False``. Every follow-up stage is guarded by
``if REAL_MODE and <stage>_ENABLED:``, so that test executes **none** of
stages 5A/5B0/5B1/5B2/5B3/5C. Three separate bugs reached real GPU runs
through that hole:

1. capability scoring called ``property_answer_matches`` against an empirical
   concept whose answer had not been resolved yet (crash);
2. ``assert_property_pair_changes_answer`` compared an unresolved concept's
   empty alias set and silently admitted the pair instead of refusing it;
3. dropping ``clean_capability`` from the audit call to break that cycle also
   disabled the capability gate for every declared concept, because a missing
   dict yields ``None`` rather than ``False``.

Each was a logic error in a code path no test executed. This harness closes
that hole: it runs the notebook's *real* stage cells with ``REAL_MODE = True``
against a synthetic 42-layer backend, so the follow-up stages execute their
actual production code.

What is faked, and what is not
------------------------------
Faked: the model (a deterministic mock backend), Drive media loading, and the
checksum-pinned prior-run loaders. Those have their own tests and were never
where the bugs were.

Real: every line of the Stage 5A/5B0/5B1/5B2/5B3/5C cells, the property audit,
the dominant-answer rule, the capability gate, the freeze gate, the exclusion
audit, the unit store and its resume fingerprint, and every verdict function.

**Nothing here is a scientific result.** The backend is synthetic.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import torch

from jlens.lens import JacobianLens
from jlens.mmpilot import multimodal_followup as followup
from jlens.mmpilot import multimodal_lens as mmlens
from jlens.mmpilot.mock import MockPilotBackend, MockWorld
from jlens.mmpilot.store import payload_checksum

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "multimodal_jspace_matched_jlens_colab.ipynb"

#: Concepts the follow-up stages name, plus the two unrelated-control concepts.
_CONCEPTS = (
    "bird", "cat", "cow", "dog", "sheep", "zebra", "giraffe",
    "microwave", "toilet",
)
_BAND = tuple(range(16, 41))


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _override_config(source: str, overrides: dict[str, str]) -> str:
    """Rewrite top-level ``NAME = ...`` assignments in the config cell.

    The config cell also computes the stage-gating booleans at its end, so a
    switch cannot be set from outside between the two; rewriting the literal
    assignment is what lets a test choose a stage.
    """
    for name, literal in overrides.items():
        # a lambda replacement, because a Windows path literal contains
        # backslashes that re.sub would otherwise read as escape sequences
        source, count = re.subn(
            rf"^{re.escape(name)} = .*$",
            lambda _m, _r=f"{name} = {literal}": _r,
            source,
            flags=re.M,
        )
        assert count == 1, f"config override {name!r} matched {count} lines"
    return source


def _fake_groups(n_per_concept: int = 200) -> list[dict]:
    groups = []
    for concept in _CONCEPTS:
        for index in range(n_per_concept):
            key = f"{concept}-{index:04d}"
            groups.append({
                "group_id": f"g_{key}",
                "image_id": f"img_{key}",
                # the selector requires the literal concept word in the caption
                "caption": f"A {concept} in a scene number {index}",
                "concept_annotations": [concept],
                "image_path": f"/fake/{concept}/{index:04d}.jpg",
                "audio_path": f"/fake/{concept}/{index:04d}.wav",
            })
    return groups


def _synthetic_lens(d_model: int, path: Path) -> str:
    """A JacobianLens over L16-L40 with the mock backend's width."""
    jacobians = {
        layer: torch.eye(d_model, dtype=torch.float32) for layer in _BAND
    }
    lens = JacobianLens(jacobians, n_prompts=99, d_model=d_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(path))
    return str(path)


class _Harness:
    """Builds the fixture tree and runs the notebook for one stage."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        self.world = MockWorld({name: (name,) for name in _CONCEPTS})
        self.backend = MockPilotBackend(self.world, n_layers=42)
        # The mock's encode_candidate deliberately appends a suffix token so
        # complete-sequence scoring is exercised elsewhere; the coordinate
        # swap needs the single vocabulary row a concept occupies, which is
        # exactly what the real tokenizer gives it and what encode_token is.
        self.backend.encode_candidate = self.backend.encode_token
        self.d_model = self.world.d_model
        self.groups = _fake_groups()
        self.runs = tmp_path / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.manifest = tmp_path / "expanded_manifest.json"
        self.manifest.write_text(
            json.dumps({"groups": self.groups}), encoding="utf-8"
        )
        self.lens_path = _synthetic_lens(self.d_model, tmp_path / "lens.pooled.pt")
        # image ids the pinned prior runs are pretending to have spent
        self.spent = [f"img_spent_{i:04d}" for i in range(64)]

    # ---------------------------------------------------------------- fakes
    def _install(self) -> None:
        mp = self.monkeypatch
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=lambda *_a, **_k: None)
        google = types.ModuleType("google")
        google.colab = colab
        mp.setitem(sys.modules, "google", google)
        mp.setitem(sys.modules, "google.colab", colab)

        lens_path, spent, band = self.lens_path, self.spent, list(_BAND)

        def fake_causal_source(*_a, **_k):
            payload = {
                "run_dir": "fake", "lens_paths": {
                    arm: lens_path
                    for arm in ("text", "image", "spoken_audio", "pooled")
                },
                "excluded_image_ids": [f"img_prior_{i:04d}" for i in range(32)],
                "fit_plan_digest": None,
            }
            return {**payload, "source_digest": payload_checksum(payload)}

        def fake_broad_source(*_a, **_k):
            payload = {
                "run_dir": "fake", "lens_path": lens_path, "layers": band,
                "excluded_image_ids": [f"img_dev_{i:04d}" for i in range(32)],
                "direction": ["bird", "cat"], "alpha": 1.0,
            }
            return {**payload, "source_digest": payload_checksum(payload)}

        def fake_spent_confirmation(*_a, **_k):
            payload = {
                "run_dir": "fake", "verdict": "FRESH_MULTIMODAL_CONFIRMATION_GO",
                "candidate_image_ids": spent, "n_candidates": len(spent),
                "recruited_image_ids": spent[:16], "n_recruited": 16,
                "n_capability_rows": 192, "all_candidates_spent": True,
                "excluding_only_recruits_would_be_wrong": True,
            }
            return {**payload, "spent_digest": payload_checksum(payload)}

        mp.setattr(mmlens, "load_completed_causal_source", fake_causal_source)
        mp.setattr(
            mmlens, "load_broad_pooled_development_source", fake_broad_source
        )
        mp.setattr(
            followup, "load_spent_confirmation_population", fake_spent_confirmation
        )

        def fake_localization_population(*_a, **_k):
            # the spent development photographs Stage 5A reuses; these must
            # exist in the manifest so the notebook can resolve them
            payload = {
                "run_dir": "fake", "direction": ["bird", "cat"],
                "groups": [
                    {"group_id": f"g_bird-{i:04d}", "image_id": f"img_bird-{i:04d}"}
                    for i in range(3)
                ],
                "n_groups": 3,
                "population_status": "spent_development_population",
                "reuse_licence": "descriptive and exploratory analyses only",
                "lens_refitted": False,
            }
            return {**payload, "population_digest": payload_checksum(payload)}

        mp.setattr(
            followup, "load_localization_population", fake_localization_population
        )
        mp.setattr(followup, "load_extra_spent_image_ids", lambda paths: {
            "image_ids": [], "n_image_ids": 0, "checksum_verified": False,
            "digest": payload_checksum({"empty": True}),
        })

        # media loading: return the mock world's own evidence vectors, keyed
        # off the concept encoded in the fake path
        world = self.world

        def evidence_for(path: str, modality: str):
            concept = str(path).split("/")[2]
            return world.evidence(
                concepts_present=(concept,), modality=modality, nuisance_key=str(path)
            )

        # the media files themselves do not exist; the backend only takes
        # their checksum for provenance, so a deterministic stand-in keeps the
        # fixture from having to materialise thousands of empty files
        from jlens.mmpilot import mock as mock_module
        mp.setattr(
            mock_module, "file_checksum",
            lambda path: "sha256:" + payload_checksum(str(path)).split(":")[1],
        )

        # the mock tokenizer has no real digit lexicalization, so the leg-count
        # endpoint resolver cannot round-trip "2"/"4" through it. Stage 5A and
        # 5C only need the ids to exist; whether a trial then succeeds is not
        # what this harness asserts.
        from jlens.mmpilot import digit_reasoning_confirmation as digits
        mp.setattr(
            digits, "resolve_digit_endpoints",
            lambda _backend: {"token_ids": {"2": 2, "4": 4}},
        )

        from jlens.mmpilot import media_io
        mp.setattr(media_io, "drive_media_loaders", lambda **_k: {
            "load_image": lambda p: evidence_for(p, "image"),
            "load_audio": lambda p: (evidence_for(p, "spoken_audio"), 16000),
        })

    # ------------------------------------------------- scripted model output
    def script_model(self) -> None:
        """Make the fake model answer the sound question correctly.

        Without this the mock's generations are noise, capability never
        clears, and stages 5B1/5B2/5B3 stay gated shut -- which is exactly the
        blind spot that let three bugs through. Scripting the *model output*
        (not the logic under test) lets the real audit, gate, freeze and
        confirmation code execute on data that passes.
        """
        mp = self.monkeypatch
        sounds = {
            "bird": "chirp", "cat": "meow", "cow": "moo",
            "dog": "bark", "sheep": "baa",
        }
        seen = {"concept": "cat"}
        kinds: dict[int, str] = {}

        original_build = self.backend.build_inputs

        def build_inputs(**kwargs):
            text = str(kwargs.get("prompt") or "")
            path = str(kwargs.get("media_path") or "")
            for concept in sounds:
                if f" {concept} " in f" {text} " or f"/{concept}/" in path:
                    seen["concept"] = concept
                    break
            return original_build(**kwargs)

        mp.setattr(self.backend, "build_inputs", build_inputs)

        from jlens.mmpilot import coordinate_swap as cswap
        from jlens.mmpilot import multimodal_lens as ml
        from jlens.mmpilot import workspace_replication as wr

        original_bases = ml.build_swap_bases_for_lens

        def tagged_bases(*args, **kwargs):
            result = original_bases(*args, **kwargs)
            source = kwargs.get("source")
            target = kwargs.get("target")
            names = (
                getattr(source, "concept", ""), getattr(target, "concept", "")
            )
            kind = "unrelated" if "microwave" in names or "toilet" in names else "exact"
            for basis in result.values():
                kinds[id(basis)] = kind
            return result

        original_random = cswap.random_two_direction_basis

        def tagged_random(basis, **kwargs):
            result = original_random(basis, **kwargs)
            kinds[id(result)] = "random"
            return result

        mp.setattr(ml, "build_swap_bases_for_lens", tagged_bases)
        mp.setattr(cswap, "random_two_direction_basis", tagged_random)

        def fake_completion(_backend, _inputs, *, answer="", max_new_tokens=6):
            return {
                "generated_text": sounds.get(seen["concept"], "unknown"),
                "n_forward_passes": 1,
            }

        def fake_swap_trial(
            _backend, _inputs, *, bases, alpha, answer, max_new_tokens=6, **_k
        ):
            kind = next(
                (kinds.get(id(b), "exact") for b in bases.values()), "exact"
            )
            moved = float(alpha) == 1.0 and kind == "exact"
            return {
                "generated_text": str(answer) if moved else sounds[seen["concept"]],
                "alpha": float(alpha),
                "layers_patched": sorted(bases),
                "all_prompt_positions_patched": True,
                "n_forward_passes": 1,
                "intervention_diagnostics": {
                    "by_layer": [{
                        "max_after_to_before_activation_ratio": 1.02,
                        "max_update_to_activation_ratio": 0.2,
                    }],
                    "max_post_cast_relative_residual_drift": 0.0,
                    "max_post_cast_relative_coordinate_error": 0.0,
                    "all_hooks_fired": True,
                },
            }

        mp.setattr(wr, "unrestricted_greedy_completion", fake_completion)
        mp.setattr(wr, "unrestricted_greedy_swap_trial", fake_swap_trial)

    # ------------------------------------------------------------------ run
    def run(self, **switches: bool) -> dict:
        self._install()
        overrides = {name: repr(value) for name, value in switches.items()}
        overrides["RUN_REAL_MATCHED_JLENS"] = "True"
        overrides["CONFIRM_MODEL_LOAD"] = "True"
        # keep the fixture small enough to run in seconds
        overrides["NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT"] = "12"
        overrides["NEW_PROPERTY_DEV_IMAGES_PER_DIRECTION"] = "3"
        overrides["NEW_PROPERTY_CONFIRM_CANDIDATES"] = "12"
        overrides["NEW_PROPERTY_CONFIRM_IMAGES"] = "3"
        overrides["ASYMMETRY_CANDIDATES"] = "12"
        overrides["ASYMMETRY_IMAGES"] = "3"
        overrides["BROAD_POOLED_IMAGES_PER_DIRECTION"] = "2"

        ns: dict = {"__name__": "__main__"}
        for source in _code_cells():
            if "REPO_URL" in source:
                # Bootstrap is environment setup only (clone/pull/pip). Replace
                # it with the names later cells need, so the harness neither
                # touches the network nor reinstalls the package under test.
                exec(
                    "import hashlib, json, os, subprocess, sys, tempfile\n"
                    "from pathlib import Path\n"
                    "IN_COLAB = False\n"
                    f"REPO_DIR = Path({str(ROOT)!r})\n"
                    "COMMIT = 'realpath-harness'\n",
                    ns,
                )
                continue
            if "RUN_REAL_MATCHED_JLENS = " in source:
                source = _override_config(source, overrides)
            if "build_real_backend" in source:
                # substitute the model load with the synthetic backend
                ns["BACKEND"] = self.backend
                ns["BUNDLE"] = None
                ns["AUDIO_RECORD"] = {"protocol_fingerprint": "fake"}
                continue
            exec(compile(source, "cell", "exec"), ns)
            if "EXPANDED_MANIFEST_CACHE" in source and ns.get("REAL_MODE"):
                ns["RUNS_ROOT"] = self.runs
                ns["EXPANDED_MANIFEST_CACHE"] = self.manifest
                ns["IMAGE_MEDIA_ROOT"] = Path("/fake")
        return ns


# ---------------------------------------------------------------- the tests


def test_stage_5b0_property_audit_runs_for_real(tmp_path, monkeypatch) -> None:
    """The whole B0 path executes: capability, resolution, gate, report."""
    ns = _Harness(tmp_path, monkeypatch).run(RUN_STAGE5B0_PROPERTY_AUDIT=True)
    report = ns["PROPERTY_AUDIT_REPORT"]
    assert report is not None
    assert report["family"] == "animal_sound"
    assert report["verdict"].startswith("PROPERTY_AUDIT_")
    # bug 1 regression: capability rows were scored without crashing even
    # though bird's answer is empirical
    assert report["capability_rows"], "no capability rows were collected"
    assert all("pass" in row for row in report["capability_rows"])
    # every concept got a real per-modality rate
    rates = report["clean_capability_by_concept"]
    for concept in ns["NEW_PROPERTY_CONCEPTS"]:
        assert set(rates[concept]) == {"text", "image", "spoken_audio"}


def test_capability_gate_is_actually_applied_in_the_real_stage(
    tmp_path, monkeypatch
) -> None:
    """Bug 3 regression, at the notebook level rather than the module level.

    Every concept whose measured rate misses the threshold in any modality
    must be absent from usable_concepts. Previously the notebook's single
    audit call omitted clean_capability, so this held vacuously for nobody.
    """
    ns = _Harness(tmp_path, monkeypatch).run(RUN_STAGE5B0_PROPERTY_AUDIT=True)
    report = ns["PROPERTY_AUDIT_REPORT"]
    rates = report["clean_capability_by_concept"]
    threshold = ns["NEW_PROPERTY_MIN_CLEAN_CAPABILITY_RATE"]
    for concept, by_modality in rates.items():
        insufficient = any(rate < threshold for rate in by_modality.values())
        if insufficient:
            assert concept not in report["usable_concepts"], (
                f"{concept} scored {by_modality} against threshold {threshold} "
                "but was still marked usable"
            )
    # and the two audit passes must agree about bird
    rows = {row["concept"]: row for row in report["concepts"]}
    bird = rows["bird"]
    assert bird["empirical_resolution"] is not None
    if bird["empirical_resolution"]["resolved"]:
        assert bird["clean_capability"] == (
            bird["empirical_resolution"]["rates_by_modality"]
        )


def test_stage_5a_localization_runs_for_real(tmp_path, monkeypatch) -> None:
    ns = _Harness(tmp_path, monkeypatch).run(
        RUN_STAGE5A_BAND_LOCALIZATION=True, CONFIRM_LOCALIZATION_BUDGET=True
    )
    report = ns["LOCALIZATION_REPORT"]
    assert report is not None
    assert report["label"] == "exploratory"
    assert report["is_confirmation"] is False
    assert report["claim_boundary"]["onset_layer_claimed"] is False
    assert len(report["bands_tested"]) == ns["LOCALIZATION_GRID"]["n_bands"]
    assert report["rows"], "no localization trials ran"


def test_stage_5c_asymmetry_runs_for_real(tmp_path, monkeypatch) -> None:
    ns = _Harness(tmp_path, monkeypatch).run(
        RUN_STAGE5C_ASYMMETRY_REPLICATION=True, CONFIRM_ASYMMETRY_BUDGET=True
    )
    report = ns["ASYMMETRY_REPORT"]
    assert report is not None
    assert report["framing"] == "asymmetry replication test"
    assert report["cause_of_asymmetry_identified"] is False
    assert report["verdict"].startswith("ASYMMETRY_")


def test_every_followup_stage_is_reachable_under_real_mode(
    tmp_path, monkeypatch
) -> None:
    """Each stage switch actually enters its guarded block.

    This is the specific hole the mock end-to-end test left: with
    REAL_MODE False every one of these blocks is skipped, so a stage could
    be arbitrarily broken and still "pass".
    """
    cases = {
        "5A": ({"RUN_STAGE5A_BAND_LOCALIZATION": True,
                "CONFIRM_LOCALIZATION_BUDGET": True}, "LOCALIZATION_ENABLED"),
        "5B0": ({"RUN_STAGE5B0_PROPERTY_AUDIT": True}, "PROPERTY_AUDIT_ENABLED"),
        "5C": ({"RUN_STAGE5C_ASYMMETRY_REPLICATION": True,
                "CONFIRM_ASYMMETRY_BUDGET": True}, "ASYMMETRY_ENABLED"),
    }
    for label, (switches, flag) in cases.items():
        ns = _Harness(tmp_path / label, monkeypatch).run(**switches)
        assert ns[flag] is True, f"{label} did not enable under REAL_MODE"
        assert ns["REAL_MODE"] is True


def test_stage_5b1_development_executes_when_the_audit_passes(
    tmp_path, monkeypatch
) -> None:
    """Stage 5B1's intervention loop runs for real, end to end.

    This is the stage that would spend the next GPU budget. Before this
    harness it had never been executed by any test.
    """
    harness = _Harness(tmp_path, monkeypatch)
    harness.script_model()
    ns = harness.run(
        RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT=True,
        CONFIRM_NEW_PROPERTY_DEVELOPMENT_BUDGET=True,
    )
    audit = ns["PROPERTY_AUDIT_REPORT"]
    assert audit["verdict"] == "PROPERTY_AUDIT_GO", audit["usable_concepts"]
    # bird's answer came from the rule, not the declared table
    rows = {row["concept"]: row for row in audit["concepts"]}
    assert rows["bird"]["empirical_resolution"]["resolved"] is True
    assert rows["bird"]["answer"] == "chirp"

    report = ns["NEW_PROPERTY_DEVELOPMENT_REPORT"]
    assert report is not None
    assert report["rows"], "no intervention trials ran"
    assert report["verdict"].startswith("NEW_PROPERTY_DEVELOPMENT_")
    assert report["lens_refitted"] is False
    assert report["teacher_forcing_used"] is False
    assert report["candidate_list_supplied"] is False
    # scoring used the resolved answer, so a bird-target direction is scorable
    directions = {row["direction"] for row in report["rows"]}
    assert directions, "no directions were scored"


def test_full_5b1_to_5b3_chain_executes(tmp_path, monkeypatch) -> None:
    """Development -> freeze -> fresh confirmation, all through real cells.

    The scripted model makes the exact condition move the answer and every
    control leave it alone, so development passes and the later stages become
    reachable. What is asserted is that each stage *runs* and produces a
    well-formed artifact -- not that any effect is real.
    """
    # --- Stage 5B1
    dev_harness = _Harness(tmp_path / "dev", monkeypatch)
    dev_harness.script_model()
    dev_ns = dev_harness.run(
        RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT=True,
        CONFIRM_NEW_PROPERTY_DEVELOPMENT_BUDGET=True,
    )
    development = dev_ns["NEW_PROPERTY_DEVELOPMENT_REPORT"]
    assert development["verdict"] == "NEW_PROPERTY_DEVELOPMENT_GO", (
        development["failure_modes"]
    )
    assert development["passing_directions"]

    dev_dir = Path(dev_ns["NEW_PROPERTY_DEV_RUN_DIR"])
    assert (dev_dir / "new_property_development_report.json").is_file()
    assert (dev_dir / "new_property_audit_report.json").is_file()

    # --- Stage 5B2: freeze, on CPU, reading the development run by checksum
    freeze_harness = _Harness(tmp_path / "freeze", monkeypatch)
    freeze_harness.script_model()
    freeze_ns = freeze_harness.run(
        RUN_STAGE5B2_FREEZE_NEW_PROPERTY_DESIGN=True,
        NEW_PROPERTY_DEVELOPMENT_RUN_DIR=str(dev_dir),
        EXPECTED_NEW_PROPERTY_DEVELOPMENT_CHECKSUM=development["report_checksum"],
    )
    design = freeze_ns["FROZEN_NEW_PROPERTY_DESIGN"]
    assert design is not None
    assert design["frozen_before_fresh_population_opened"] is True
    assert design["lens_refitted"] is False
    # the frozen aliases are the resolved ones, never the empty declared set
    for concept, aliases in design["answer_aliases"].items():
        assert aliases, f"{concept} was frozen with an empty alias set"
    design_path = dev_dir / "frozen_new_property_design.json"
    assert design_path.is_file()

    # --- Stage 5B3: fresh confirmation under the frozen design
    confirm_harness = _Harness(tmp_path / "confirm", monkeypatch)
    confirm_harness.script_model()
    confirm_ns = confirm_harness.run(
        RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION=True,
        CONFIRM_NEW_PROPERTY_CONFIRMATION_BUDGET=True,
        NEW_PROPERTY_FROZEN_DESIGN_PATH=str(design_path),
        NEW_PROPERTY_DEVELOPMENT_RUN_DIR=str(dev_dir),
    )
    confirmation = confirm_ns["NEW_PROPERTY_CONFIRMATION_REPORT"]
    assert confirmation is not None
    assert confirmation["verdict"].startswith("NEW_PROPERTY_CONFIRMATION_")
    assert confirmation["exclusion_audit"]["disjoint"] is True
    assert confirmation["design_digest"] == design["design_digest"]


def test_capability_failure_reports_a_verdict_instead_of_crashing(
    tmp_path, monkeypatch
) -> None:
    """A recruitment failure must not surface as a 'missing control' error.

    Found by this harness: confirmation_verdict fell through to the pairing
    check with zero rows when capability failed, raising a misleading refusal
    for what is an ordinary expected outcome.
    """
    from jlens.mmpilot.multimodal_followup import (
        asymmetry_replication_design,
        asymmetry_replication_verdict,
        exclusion_universe,
    )

    design = asymmetry_replication_design(
        lens_checksum="sha256:" + "0" * 64, exclusions=exclusion_universe()
    )
    report = asymmetry_replication_verdict(
        [], design=design, capability_go=False,
        exclusion_audit={"disjoint": True},
    )
    assert report["verdict"] == "ASYMMETRY_REPLICATION_CAPABILITY_NO_GO"
    assert report["failure_mode"] == "no_trials"
    assert report["cells"] == []
