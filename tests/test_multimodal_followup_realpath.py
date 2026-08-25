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

import hashlib
import json
import re
import sys
import types
from pathlib import Path

import pytest
import torch

from jlens.lens import JacobianLens
from jlens.mmpilot import multimodal_followup as followup
from jlens.mmpilot import multimodal_lens as mmlens
from jlens.mmpilot.mock import MockPilotBackend, MockWorld
from jlens.mmpilot.multimodal_instrument import (
    INSTRUMENT_STATES,
    INTEGRITY_CLAUSES,
    MODEL_DTYPE_REALIZATION,
)
from jlens.mmpilot.store import payload_checksum

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "multimodal_jspace_matched_jlens_colab.ipynb"

#: Concepts the follow-up stages name, plus the two unrelated-control concepts.
_CONCEPTS = (
    "bird", "cat", "cow", "dog", "sheep", "zebra", "giraffe",
    "microwave", "toilet",
)
_BAND = tuple(range(16, 41))


#: Defect switches. Every one is off by default, so the harness's baseline is a
#: faithfully realized intervention; a test turns exactly one on.
_CLEAN_DEFECTS: dict = {
    "uncorrected_relative_error": 0.21,   # what the completed flawed run saw
    "corrected_relative_error": 0.004,
    "relative_residual_drift": 0.001,
    "direct_answer_match_error": 0.002,
    "nonfinite": False,
    "hooks_do_not_fire": False,
    "controls_move": False,
    "exact_never_moves": False,
    "direct_answer_never_moves": False,
    "correction_changes_outcome": False,
}


def _synthetic_swap_stats(
    layers,
    *,
    alpha: float,
    n_passes: int,
    relative_error: float,
    residual_drift: float,
    converged: bool,
    finite: bool = True,
    policy=None,
) -> dict:
    """Per-layer hook stats in the exact shape ``swap_coordinates`` records.

    The point of building *these* and letting
    :func:`jlens.mmpilot.workspace_replication.summarize_swap_diagnostics`
    compact them is that the harness then cannot disagree with production about
    the summary's shape. The previous fake asserted its own shape by hand and
    got ``by_layer`` wrong -- a list where production returns a dict keyed by
    string layer number -- which is precisely why the verdict bug survived.
    """
    bad = float("nan")
    stats = {}
    for layer in sorted(map(int, layers)):
        record = {
            "alpha": float(alpha),
            "n_positions": 4,
            "update_norm": [1.0, 1.0, 1.0, 1.0],
            "activation_norm_before": [10.0] * 4,
            "activation_norm_after": [10.1] * 4,
            "alpha_one_is_exact_exchange": float(alpha) == 1.0,
            "model_dtype_realization": None if policy is None else policy.to_dict(),
            "model_dtype_realization_converged": bool(converged),
            "model_dtype_corrections_applied": 0 if policy is None else 2,
            "max_coordinate_update_error": bad if not finite else 1e-12,
            "max_post_cast_coordinate_update_error": float(relative_error) * 3.0,
            "max_post_cast_relative_coordinate_update_error": float(relative_error),
            "max_orthogonal_residual_drift": 1e-12,
            "max_post_cast_relative_orthogonal_residual_drift": float(residual_drift),
        }
        stats[layer] = {
            "layer": layer,
            "n_forward_passes": int(n_passes),
            "n_positions": 4,
            "positions": [0, 1, 2, 3],
            "basis": {"diagnostics": {"condition_number": 2.0, "numerical_rank": 2}},
            "swap_history": [record] * int(n_passes),
        }
    return stats


def _synthetic_direct_answer_stats(
    layers, *, n_passes: int, match_error: float, policy=None
) -> dict:
    """The direct-answer control's hook stats, same reasoning as above."""
    stats = {}
    for layer in sorted(map(int, layers)):
        stats[layer] = {
            "layer": layer,
            "n_forward_passes": int(n_passes),
            "positions": [0, 1, 2, 3],
            "answer_vector_norm": 1.0,
            "answer_vector_checksum": "sha256:" + "0" * 64,
            "history": [
                {
                    "n_positions": 4,
                    "all_finite": True,
                    "max_relative_norm_match_error": float(match_error),
                    "model_dtype_realization_converged": True,
                    "model_dtype_corrections_applied": 0 if policy is None else 1,
                    "model_dtype_realization": (
                        None if policy is None else policy.to_dict()
                    ),
                    "max_update_to_activation_ratio": 0.2,
                }
            ] * int(n_passes),
        }
    return stats


def _with_hooks_not_fired(diagnostics: dict) -> dict:
    """One hook silently missed a forward pass."""
    return {**diagnostics, "all_hooks_fired": False}


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

    def __init__(self, tmp_path: Path, monkeypatch, **defects) -> None:
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        unknown = sorted(set(defects) - set(_CLEAN_DEFECTS))
        assert not unknown, f"unknown defect switch(es) {unknown}"
        self.defects = {**_CLEAN_DEFECTS, **defects}
        #: Every swap / direct-answer call the notebook made, so a test can
        #: assert the realization policy actually reached production code
        #: rather than assert on a printed string.
        self.calls: list[dict] = []
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
        mp.setattr(
            followup,
            "audio_metadata_linkage_audit",
            lambda *_a, **_k: {
                "version": followup.AUDIO_LINKAGE_AUDIT_VERSION,
                "concept": "cow",
                "failed_only": True,
                "n_rows_audited": 1,
                "records": [{"passed": True}],
                "metadata_linkage_verified": True,
                "waveform_content_independently_transcribed": False,
                "causal_outcomes_used": False,
                "model_forwards": 0,
                "backward_passes": 0,
                "verdict": "AUDIO_METADATA_LINKAGE_GO",
                "audit_digest": payload_checksum({"synthetic": "linkage-go"}),
            },
        )

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

        defects = self.defects

        def fake_swap_trial(
            _backend, _inputs, *, bases, alpha, answer, max_new_tokens=6,
            realization_policy=None, **_k
        ):
            kind = next(
                (kinds.get(id(b), "exact") for b in bases.values()), "exact"
            )
            moved = float(alpha) == 1.0 and (
                kind == "exact" if not defects["controls_move"] else alpha != 0.0
            )
            if defects["exact_never_moves"] and kind == "exact":
                moved = False
            self.calls.append({
                "trial": "swap",
                "alpha": float(alpha),
                "kind": kind,
                "realization_policy": (
                    None if realization_policy is None
                    else realization_policy.to_dict()
                ),
            })
            # An uncorrected cast leaves the intended coordinates unrealized.
            # This is what the completed run actually did, and the harness
            # reproduces it whenever the caller omits the policy.
            error = (
                float(defects["uncorrected_relative_error"])
                if realization_policy is None
                else float(defects["corrected_relative_error"])
            )
            stats = _synthetic_swap_stats(
                sorted(bases),
                alpha=float(alpha),
                n_passes=1,
                relative_error=error,
                residual_drift=float(defects["relative_residual_drift"]),
                converged=(
                    realization_policy is None
                    or error <= float(realization_policy.relative_coordinate_tolerance)
                ),
                finite=not defects["nonfinite"],
                policy=realization_policy,
            )
            passes = 0 if defects["hooks_do_not_fire"] else 1
            return {
                "generated_text": str(answer) if moved else sounds[seen["concept"]],
                "alpha": float(alpha),
                "alpha_role": "exact_exchange" if alpha == 1.0 else "nonexact",
                "layers_patched": sorted(bases),
                "all_prompt_positions_patched": True,
                "n_forward_passes": 1,
                "teacher_forcing_used": False,
                "candidate_list_supplied": False,
                # The real summariser, not a hand-written stand-in. The bug this
                # harness missed was a fake that returned ``by_layer`` as a list
                # while production returned a dict keyed by layer; calling the
                # production summariser makes that class of drift impossible.
                "intervention_diagnostics": wr.summarize_swap_diagnostics(
                    stats, expected_forward_passes=passes or 1
                ) if passes else _with_hooks_not_fired(
                    wr.summarize_swap_diagnostics(stats, expected_forward_passes=1)
                ),
            }

        def fake_direct_answer_trial(
            _backend, _inputs, *, bases, answer_vectors, answer,
            max_new_tokens=6, realization_policy=None, alpha=1.0, **_k
        ):
            if set(map(int, bases)) != set(map(int, answer_vectors)):
                raise AssertionError(
                    "the direct-answer control must cover exactly the band"
                )
            self.calls.append({
                "trial": "direct_answer",
                "alpha": float(alpha),
                "realization_policy": (
                    None if realization_policy is None
                    else realization_policy.to_dict()
                ),
            })
            moved = not defects["direct_answer_never_moves"]
            stats = _synthetic_direct_answer_stats(
                sorted(map(int, bases)),
                n_passes=1,
                match_error=float(defects["direct_answer_match_error"]),
                policy=realization_policy,
            )
            return {
                "generated_text": str(answer) if moved else sounds[seen["concept"]],
                "condition": "direct_answer_norm_matched",
                "alpha": float(alpha),
                "layers_patched": sorted(map(int, bases)),
                "all_prompt_positions_patched": True,
                "n_forward_passes": 1,
                "teacher_forcing_used": False,
                "candidate_list_supplied": False,
                "intervention_diagnostics": wr.summarize_direct_answer_diagnostics(
                    stats, expected_forward_passes=1
                ),
            }

        def fake_single_token_swap_trial(
            _backend, _inputs, *, bases, alpha=1.0, realization_policy=None,
            compact_positions=False, **_k
        ):
            """The Stage 3D/3DA single-token endpoint, with real diagnostics."""
            kind = next(
                (kinds.get(id(b), "exact") for b in bases.values()), "exact"
            )
            self.calls.append({
                "trial": "single_token_swap",
                "alpha": float(alpha),
                "kind": kind,
                "realization_policy": (
                    None if realization_policy is None
                    else realization_policy.to_dict()
                ),
            })
            error = (
                float(defects["uncorrected_relative_error"])
                if realization_policy is None
                else float(defects["corrected_relative_error"])
            )
            stats = _synthetic_swap_stats(
                sorted(bases), alpha=float(alpha), n_passes=1,
                relative_error=error,
                residual_drift=float(defects["relative_residual_drift"]),
                converged=(
                    realization_policy is None
                    or error <= float(realization_policy.relative_coordinate_tolerance)
                ),
                policy=realization_policy,
            )
            summary = wr.summarize_swap_diagnostics(stats, expected_forward_passes=1)
            moved = float(alpha) == 1.0 and kind == "exact"
            if realization_policy is not None and defects["correction_changes_outcome"]:
                moved = not moved
            return {
                "alpha": float(alpha),
                "layers_patched": sorted(bases),
                "all_prompt_positions_patched": True,
                "positions_patched": {},
                "clean_top_token_id": 11,
                "patched_top_token_id": 4 if moved else 2,
                "prediction_changed": moved,
                "max_activation_norm_ratio": 1.02,
                "min_activation_norm_ratio": 1.0,
                "max_update_to_activation_norm_ratio": 0.2,
                "max_orthogonal_residual_drift": 1e-12,
                "max_coordinate_update_error": 1e-12,
                "coordinate_error_basis": "float64_pre_cast_solve",
                "max_post_cast_relative_coordinate_error": summary[
                    "max_post_cast_relative_coordinate_error"
                ],
                "max_post_cast_relative_residual_drift": summary[
                    "max_post_cast_relative_residual_drift"
                ],
                "all_layers_are_exact_alpha_one_exchange_before_cast": summary[
                    "all_layers_are_exact_alpha_one_exchange_before_cast"
                ],
                "all_model_dtype_realizations_converged": summary[
                    "all_model_dtype_realizations_converged"
                ],
                "max_model_dtype_corrections_applied": summary[
                    "max_model_dtype_corrections_applied"
                ],
                "model_dtype_realization_policy": summary[
                    "model_dtype_realization_policy"
                ],
                "teacher_forcing_used": False,
                "candidate_list_supplied": False,
            }

        mp.setattr(wr, "unrestricted_greedy_completion", fake_completion)
        mp.setattr(wr, "unrestricted_greedy_swap_trial", fake_swap_trial)
        mp.setattr(
            wr, "unrestricted_greedy_direct_answer_trial", fake_direct_answer_trial
        )
        mp.setattr(ml, "unrestricted_swap_trial", fake_single_token_swap_trial)
        mp.setattr(
            self.backend, "decode_token",
            lambda token_id: {4: "4", 2: "2", 11: "?"}.get(int(token_id), "?"),
        )

    def _prepare_prompt_screen_source(self, ns: dict) -> tuple[Path, str, str]:
        """Materialise the checksum-pinned failed B0 source for Stage 5B00."""

        root = self.tmp / "prompt-screen-source"
        root.mkdir(parents=True, exist_ok=True)
        n = int(ns["NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT"])
        population = {
            concept: [
                {"group_id": f"g_{concept}-{index:04d}",
                 "image_id": f"img_{concept}-{index:04d}"}
                for index in range(n)
            ]
            for concept in ("cat", "cow")
        }
        (root / "development_population.json").write_text(
            json.dumps({"population": population}), encoding="utf-8"
        )
        (root / "scientific_config.json").write_text(
            json.dumps({
                "model_repo_id": ns["MODEL_REPO_ID"],
                "model_revision": ns["MODEL_REVISION"],
                "manifest_checksum": ns["MANIFEST_CHECKSUM"],
            }),
            encoding="utf-8",
        )
        rows = []
        for concept in ("cat", "cow"):
            for index in range(n):
                for modality in ("text", "image", "spoken_audio"):
                    # Keep one baseline cell below the gate so this faithfully
                    # represents the completed NO_GO source audit.
                    passed = not (
                        concept == "cat" and modality == "text" and index >= n // 2
                    )
                    rows.append({
                        "concept": concept,
                        "group_id": f"g_{concept}-{index:04d}",
                        "image_id": f"img_{concept}-{index:04d}",
                        "modality": modality,
                        "generated": concept,
                        "pass": passed,
                    })
        audit_digest = payload_checksum({"synthetic": "failed-audit"})
        path = root / "new_property_audit_report.json"
        path.write_text(
            json.dumps({
                "family": "animal_sound",
                "verdict": "PROPERTY_AUDIT_NO_GO",
                "audit_digest": audit_digest,
                "prompt_id": "identity_explicit_v1",
                "capability_rows": rows,
            }),
            encoding="utf-8",
        )
        file_sha = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        return root, file_sha, audit_digest

    # ------------------------------------------------------------------ run
    def run(self, **switches) -> dict:
        self._install()
        # Values that are not plain booleans are multi-line parenthesised
        # literals in the config cell, which a line-oriented rewrite cannot
        # replace. Those are applied to the namespace right after the config
        # cell instead, which is equally real: the stage cell reads them then.
        after_config = {
            name: value for name, value in switches.items()
            if not isinstance(value, bool)
        }
        switches = {
            name: value for name, value in switches.items()
            if isinstance(value, bool)
        }
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
        overrides["RECRUITED_EXPLORATORY_IMAGES_PER_DIRECTION"] = "3"

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
                # The cat->dog study's own model-load cells (Stage 6C/6E) set
                # a flag their next cell checks before proceeding, since that
                # split is what lets a fp32-specific load sit in its own cell
                # while the rest of the stage still runs under this same
                # "build_real_backend" substitution. Setting both here is a
                # no-op for every other stage's cells.
                ns["CATDOG_MODEL_LOADED_DTYPE"] = "float32"
                ns["CATDOG_CONFIRM_MODEL_LOADED"] = True
                continue
            if "PROPERTY_PROMPT_SCREEN_REPORT = None" in source:
                root, file_sha, audit_digest = self._prepare_prompt_screen_source(ns)
                ns["PROPERTY_PROMPT_SCREEN_SOURCE_RUN_DIR"] = str(root)
                ns["EXPECTED_PROPERTY_PROMPT_SCREEN_SOURCE_FILE_SHA256"] = file_sha
                ns["EXPECTED_PROPERTY_PROMPT_SCREEN_SOURCE_AUDIT_DIGEST"] = audit_digest
                ns["RECRUITED_EXPLORATORY_SOURCE_RUN_DIR"] = str(root)
                ns["EXPECTED_RECRUITED_SOURCE_FILE_SHA256"] = file_sha
                ns["EXPECTED_RECRUITED_SOURCE_AUDIT_DIGEST"] = audit_digest
            exec(compile(source, "cell", "exec"), ns)
            if "RUN_REAL_MATCHED_JLENS = " in source:
                ns.update(after_config)
            if "EXPANDED_MANIFEST_CACHE" in source and ns.get("REAL_MODE"):
                ns["RUNS_ROOT"] = self.runs
                ns["EXPANDED_MANIFEST_CACHE"] = self.manifest
                # A caller that pinned a real IMAGE_MEDIA_ROOT via after_config
                # (the cat->dog study reads raw COCO annotation files under it
                # in Stage 6A, so it needs a real directory, not this
                # placeholder) gets that value applied here -- the cell just
                # ran and unconditionally set its own hardcoded Drive path, so
                # merely *skipping* this assignment would leave that hardcoded
                # value in place instead of the override. Every other caller
                # gets the placeholder exactly as before.
                ns["IMAGE_MEDIA_ROOT"] = (
                    Path(after_config["IMAGE_MEDIA_ROOT"])
                    if "IMAGE_MEDIA_ROOT" in after_config
                    else Path("/fake")
                )
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
        "5B00": ({"RUN_STAGE5B00_PROPERTY_PROMPT_SCREEN": True,
                   "CONFIRM_PROPERTY_PROMPT_SCREEN_BUDGET": True},
                  "PROPERTY_PROMPT_SCREEN_ENABLED"),
        "5C": ({"RUN_STAGE5C_ASYMMETRY_REPLICATION": True,
                "CONFIRM_ASYMMETRY_BUDGET": True}, "ASYMMETRY_ENABLED"),
    }
    for label, (switches, flag) in cases.items():
        ns = _Harness(tmp_path / label, monkeypatch).run(**switches)
        assert ns[flag] is True, f"{label} did not enable under REAL_MODE"
        assert ns["REAL_MODE"] is True


def test_stage_5b00_prompt_screen_runs_real_cell_without_causal_work(
    tmp_path, monkeypatch
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    harness.script_model()
    ns = harness.run(
        RUN_STAGE5B00_PROPERTY_PROMPT_SCREEN=True,
        CONFIRM_PROPERTY_PROMPT_SCREEN_BUDGET=True,
    )
    report = ns["PROPERTY_PROMPT_SCREEN_REPORT"]
    assert report is not None
    assert report["verdict"] == "PROPERTY_PROMPT_SCREEN_GO"
    assert report["selected_prompt_id"] in {
        "identity_explicit_v1", "knowledge_cloze_v1"
    }
    assert report["causal_outcomes_used_for_selection"] is False
    assert report["causal_spending_licensed"] is False
    assert report["lens_fitted"] is False
    assert report["backward_passes"] == 0


def test_nonbaseline_prompt_requires_and_uses_the_pinned_screen_winner(
    tmp_path, monkeypatch
) -> None:
    screen_harness = _Harness(tmp_path / "screen", monkeypatch)
    screen_harness.script_model()
    screen_ns = screen_harness.run(
        RUN_STAGE5B00_PROPERTY_PROMPT_SCREEN=True,
        CONFIRM_PROPERTY_PROMPT_SCREEN_BUDGET=True,
    )
    screen = screen_ns["PROPERTY_PROMPT_SCREEN_REPORT"]
    prompt_id = screen["selected_prompt_id"]

    audit_harness = _Harness(tmp_path / "audit", monkeypatch)
    audit_harness.script_model()
    audit_ns = audit_harness.run(
        RUN_STAGE5B0_PROPERTY_AUDIT=True,
        NEW_PROPERTY_PROMPT_ID=prompt_id,
        NEW_PROPERTY_PROMPT_SCREEN_RUN_DIR=str(
            screen_ns["PROPERTY_PROMPT_SCREEN_RUN_DIR"]
        ),
        EXPECTED_NEW_PROPERTY_PROMPT_SCREEN_CHECKSUM=screen["report_checksum"],
    )
    audit = audit_ns["PROPERTY_AUDIT_REPORT"]
    assert audit["prompt_id"] == prompt_id
    assert audit["prompt_screen_report_checksum"] == screen["report_checksum"]
    assert audit_ns["_dev_config"]["prompt_id"] == prompt_id


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


def _run_corrected(tmp_path, monkeypatch, **defects):
    """Execute the real Stage 5B01 + 5B1RC cells against the synthetic backend."""
    harness = _Harness(tmp_path, monkeypatch, **defects)
    harness.script_model()
    ns = harness.run(
        RUN_STAGE5B01_AUDIO_LINKAGE_AUDIT=True,
        RUN_STAGE5B1RC_CORRECTED_EXPLORATORY=True,
        CONFIRM_CORRECTED_EXPLORATORY_BUDGET=True,
    )
    return harness, ns


def test_stage_5b1rc_passes_the_realization_policy_into_every_condition(
    tmp_path, monkeypatch
) -> None:
    """The root defect, asserted where it happened rather than in a docstring.

    Stage 5B1R called ``unrestricted_greedy_swap_trial`` with no
    ``realization_policy=``. This asserts the corrected cell passes the frozen
    policy on *every* generated condition -- exact, zero, random, unrelated and
    the direct-answer control -- by inspecting the calls production code
    actually made.
    """
    harness, ns = _run_corrected(tmp_path, monkeypatch)
    assert ns["AUDIO_LINKAGE_REPORT"]["verdict"] == "AUDIO_METADATA_LINKAGE_GO"
    assert harness.calls, "the corrected stage ran no trials"
    expected = MODEL_DTYPE_REALIZATION.to_dict()
    for call in harness.calls:
        assert call["realization_policy"] == expected, call
    # every declared condition really ran, including the alpha=0 parity arm
    kinds = {call["kind"] for call in harness.calls if call["trial"] == "swap"}
    assert kinds == {"exact", "random", "unrelated"}
    assert {call["alpha"] for call in harness.calls} == {0.0, 1.0}
    assert any(call["trial"] == "direct_answer" for call in harness.calls)


def test_stage_5b1rc_reports_the_real_diagnostic_shape_and_a_valid_instrument(
    tmp_path, monkeypatch
) -> None:
    """The corrected cell produces scorable rows and a named instrument state."""
    _harness, ns = _run_corrected(tmp_path, monkeypatch)
    report = ns["CORRECTED_EXPLORATORY_REPORT"]
    assert report is not None
    assert report["rows"]
    assert report["source_aggregate_verdict_unchanged"] is True
    assert report["selection_uses_only_clean_capability"] is True
    assert report["causal_outcomes_used_for_selection"] is False
    assert report["outcome_informed_stage_design"] is True
    assert report["is_confirmation"] is False
    assert report["lens_refitted"] is False
    assert report["alpha_sweep_run"] is False
    assert report["primary_alpha"] == 1.0
    assert report["design_changed_from_superseded_run"] is False
    assert report["supersedes_report_checksum"] == (
        ns["EXPECTED_RECRUITED_EXPLORATORY_REPORT_CHECKSUM"]
    )
    # the exact bug the old fake hid: production returns by_layer as a dict
    # keyed by string layer number, and every flattened row must carry the
    # integrity evidence rather than an absent-means-pass default
    for row in report["rows"]:
        if row["condition"] == "direct_answer":
            continue
        assert isinstance(row["max_coordinate_update_error"], float)
        assert isinstance(row["max_orthogonal_residual_drift"], float)
        assert row["all_hooks_fired"] is True
        assert row["all_finite"] is True
        assert row["all_model_dtype_realizations_converged"] is True
        assert row["model_dtype_realization_policy"] == (
            MODEL_DTYPE_REALIZATION.to_dict()
        )
    assert report["instrument_state"] in INSTRUMENT_STATES
    assert report["verdict"] == (
        f"CORRECTED_RECRUITED_EXPLORATORY_{report['instrument_state']}"
    )


def test_favorable_valid_primary_effect_is_an_exploratory_go(
    tmp_path, monkeypatch
) -> None:
    """The scripted world moves the exact arm and nothing else."""
    _harness, ns = _run_corrected(tmp_path, monkeypatch)
    report = ns["CORRECTED_EXPLORATORY_REPORT"]
    assert report["instrument_state"] == "EFFECT_GO"
    assert report["verdict"] == "CORRECTED_RECRUITED_EXPLORATORY_EFFECT_GO"
    assert report["passing_directions"]
    # a GO here is still exploratory and still needs a fresh population
    assert report["is_confirmation"] is False
    assert report["fresh_confirmation_licensed_directly"] is False


def test_direct_answer_passes_while_the_primary_fails_is_a_scientific_null(
    tmp_path, monkeypatch
) -> None:
    """The one combination that licenses reading a null as a null."""
    _harness, ns = _run_corrected(tmp_path, monkeypatch, exact_never_moves=True)
    report = ns["CORRECTED_EXPLORATORY_REPORT"]
    assert report["instrument_state"] == "SCIENTIFIC_NULL"
    assert not report["passing_directions"]
    for row in report["directions"]:
        assert row["direct_answer_positive_control"]["passed"] is True
        assert row["failure_mode"] == "no_effect_in_every_modality"


def test_primary_and_direct_answer_both_failing_is_inconclusive(
    tmp_path, monkeypatch
) -> None:
    """No causal leverage on this path at all: never reported as a null."""
    _harness, ns = _run_corrected(
        tmp_path, monkeypatch,
        exact_never_moves=True, direct_answer_never_moves=True,
    )
    report = ns["CORRECTED_EXPLORATORY_REPORT"]
    assert report["instrument_state"] == "INCONCLUSIVE"
    assert report["verdict"] == "CORRECTED_RECRUITED_EXPLORATORY_INCONCLUSIVE"
    for row in report["directions"]:
        assert row["direct_answer_positive_control"]["passed"] is False
        assert row["failure_mode"] == "no_effect_and_positive_control_also_failed"


def test_controls_that_move_are_a_control_failure_not_a_null(
    tmp_path, monkeypatch
) -> None:
    _harness, ns = _run_corrected(tmp_path, monkeypatch, controls_move=True)
    report = ns["CORRECTED_EXPLORATORY_REPORT"]
    assert report["instrument_state"] == "CONTROL_FAILURE"
    assert report["verdict"] == "CORRECTED_RECRUITED_EXPLORATORY_CONTROL_FAILURE"


def test_an_uncorrected_cast_is_instrument_failure_not_a_null(
    tmp_path, monkeypatch
) -> None:
    """The completed flawed run's actual numbers, scored by the fixed verdict.

    0.21 is the worst post-cast relative coordinate error that run recorded.
    Under the old two-clause check this produced ``integrity_pass = true`` and
    a printed 0/8 null.
    """
    _harness, ns = _run_corrected(
        tmp_path, monkeypatch,
        corrected_relative_error=0.21, exact_never_moves=True,
    )
    report = ns["CORRECTED_EXPLORATORY_REPORT"]
    assert report["instrument_state"] == "INSTRUMENT_FAILURE"
    assert report["verdict"] == "CORRECTED_RECRUITED_EXPLORATORY_INSTRUMENT_FAILURE"
    for row in report["directions"]:
        assert "coordinate_integrity_failed" in row["failure_mode"]
        assert "post_cast_coordinate_error_within_tolerance" in row["failure_mode"]
        for cell in row["cells"]:
            assert cell["integrity_pass"] is False


def test_nonconvergent_realization_is_instrument_failure(
    tmp_path, monkeypatch
) -> None:
    """The bounded correction ran out of budget: refuse, do not score."""
    _harness, ns = _run_corrected(
        tmp_path, monkeypatch, corrected_relative_error=0.5,
    )
    report = ns["CORRECTED_EXPLORATORY_REPORT"]
    assert report["instrument_state"] == "INSTRUMENT_FAILURE"
    failures = {
        clause
        for row in report["directions"]
        for cell in row["cells"]
        for clause in cell["integrity_failed_clauses"]
    }
    assert "model_dtype_realization_converged" in failures


def test_residual_drift_out_of_tolerance_is_instrument_failure(
    tmp_path, monkeypatch
) -> None:
    _harness, ns = _run_corrected(
        tmp_path, monkeypatch, relative_residual_drift=0.03,
    )
    report = ns["CORRECTED_EXPLORATORY_REPORT"]
    assert report["instrument_state"] == "INSTRUMENT_FAILURE"
    failures = {
        clause
        for row in report["directions"]
        for cell in row["cells"]
        for clause in cell["integrity_failed_clauses"]
    }
    assert "post_cast_residual_drift_within_tolerance" in failures


def test_hooks_that_do_not_fire_are_instrument_failure(
    tmp_path, monkeypatch
) -> None:
    _harness, ns = _run_corrected(tmp_path, monkeypatch, hooks_do_not_fire=True)
    report = ns["CORRECTED_EXPLORATORY_REPORT"]
    assert report["instrument_state"] == "INSTRUMENT_FAILURE"
    failures = {
        clause
        for row in report["directions"]
        for cell in row["cells"]
        for clause in cell["integrity_failed_clauses"]
    }
    assert "all_hooks_fired" in failures


def test_nonfinite_diagnostics_are_instrument_failure(
    tmp_path, monkeypatch
) -> None:
    _harness, ns = _run_corrected(tmp_path, monkeypatch, nonfinite=True)
    report = ns["CORRECTED_EXPLORATORY_REPORT"]
    assert report["instrument_state"] == "INSTRUMENT_FAILURE"
    failures = {
        clause
        for row in report["directions"]
        for cell in row["cells"]
        for clause in cell["integrity_failed_clauses"]
    }
    assert "all_finite" in failures


def test_corrected_rerun_fits_nothing_and_runs_no_backward_pass(
    tmp_path, monkeypatch
) -> None:
    """Zero fitting, zero gradients, and the pinned lens reused as-is."""
    harness = _Harness(tmp_path, monkeypatch)
    harness.script_model()
    fits: list[str] = []
    backwards: list[str] = []
    from jlens import fitting as fitting_module
    from jlens.lens import JacobianLens

    for name in dir(fitting_module):
        attribute = getattr(fitting_module, name)
        if callable(attribute) and name.startswith("fit"):
            monkeypatch.setattr(
                fitting_module, name,
                lambda *_a, _n=name, **_k: fits.append(_n),
            )
    original_backward = torch.Tensor.backward

    def refuse_backward(self, *args, **kwargs):
        backwards.append("backward")
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", refuse_backward)
    original_save = JacobianLens.save
    saves: list[str] = []
    monkeypatch.setattr(
        JacobianLens, "save",
        lambda self, path, *a, **k: (saves.append(str(path)), original_save(self, path, *a, **k))[1],
    )
    ns = harness.run(
        RUN_STAGE5B01_AUDIO_LINKAGE_AUDIT=True,
        RUN_STAGE5B1RC_CORRECTED_EXPLORATORY=True,
        CONFIRM_CORRECTED_EXPLORATORY_BUDGET=True,
    )
    assert ns["CORRECTED_EXPLORATORY_REPORT"] is not None
    assert fits == [], f"the corrected rerun fitted something: {fits}"
    assert backwards == [], "the corrected rerun ran a backward pass"
    assert saves == [], f"the corrected rerun wrote a lens: {saves}"


def test_interrupted_corrected_rerun_resumes_without_recomputing(
    tmp_path, monkeypatch
) -> None:
    """Every completed condition survives a disconnect and is not recomputed."""
    first_harness, first_ns = _run_corrected(tmp_path / "a", monkeypatch)
    run_dir = Path(first_ns["CORRECTED_EXPLORATORY_RUN_DIR"])
    n_units = len(list((run_dir / "units" / "intervention").glob("*.json")))
    assert n_units == len(first_ns["CORRECTED_EXPLORATORY_REPORT"]["rows"])
    first_calls = len(first_harness.calls)
    assert first_calls == n_units

    # Second session, same configuration, same run directory: nothing is
    # recomputed and the report is identical.
    second = _Harness(tmp_path / "a", monkeypatch)
    second.script_model()
    second.runs = first_harness.runs
    second_ns = second.run(
        RUN_STAGE5B01_AUDIO_LINKAGE_AUDIT=True,
        RUN_STAGE5B1RC_CORRECTED_EXPLORATORY=True,
        CONFIRM_CORRECTED_EXPLORATORY_BUDGET=True,
    )
    assert second.calls == [], "a resumed session recomputed completed units"
    assert (
        second_ns["CORRECTED_EXPLORATORY_REPORT"]["report_checksum"]
        == first_ns["CORRECTED_EXPLORATORY_REPORT"]["report_checksum"]
    )


def test_a_changed_realization_policy_changes_the_fingerprint(
    tmp_path, monkeypatch
) -> None:
    """A different instrument must refuse the old units, not mix with them."""
    from jlens.mmpilot import multimodal_instrument as instrument
    from jlens.mmpilot.coordinate_swap import ModelDtypeRealizationPolicy

    baseline = instrument.realization_policy_digest()
    looser = instrument.realization_policy_digest(
        ModelDtypeRealizationPolicy(
            max_corrections=8,
            relative_coordinate_tolerance=0.05,
            relative_residual_tolerance=0.05,
            minimum_scale=1.0,
        )
    )
    assert baseline != looser

    _harness, first_ns = _run_corrected(tmp_path / "a", monkeypatch)
    first_dir = Path(first_ns["CORRECTED_EXPLORATORY_RUN_DIR"])
    config = json.loads((first_dir / "scientific_config.json").read_text())
    assert config["model_dtype_realization_policy_digest"] == baseline
    fingerprint = json.loads(
        (first_dir / "fingerprint.json").read_text()
    )
    assert baseline in json.dumps(fingerprint)

    # A run directory written under a different policy is refused, not merged.
    stale = json.loads(json.dumps(fingerprint))
    stale["intervention_config"]["model_dtype_realization_policy_digest"] = looser
    second_dir = tmp_path / "b" / "run"
    second_dir.mkdir(parents=True)
    (second_dir / "fingerprint.json").write_text(json.dumps(stale))
    from jlens.mmpilot.store import IncompatibleStateError, RunFingerprint, UnitStore

    requested = RunFingerprint(**{
        **{k: v for k, v in fingerprint.items() if k != "layers"},
        "layers": tuple(fingerprint["layers"]),
    })
    with pytest.raises(IncompatibleStateError):
        UnitStore(second_dir, requested).open()


def test_stage_5b1a_amends_read_only_and_leaves_the_source_untouched(
    tmp_path, monkeypatch
) -> None:
    """The historical amendment: pinned, reclassified, and byte-for-byte safe."""
    harness = _Harness(tmp_path, monkeypatch)
    flawed_dir = tmp_path / "flawed-run"
    flawed_dir.mkdir(parents=True)
    flawed_path = flawed_dir / "recruited_new_property_exploratory_report.json"
    flawed_payload = {
        "verdict": "RECRUITED_NEW_PROPERTY_EXPLORATORY_NO_GO",
        "rows": [{"condition": "exact", "success": False}],
    }
    flawed_path.write_text(json.dumps(flawed_payload, indent=2), encoding="utf-8")
    before = hashlib.sha256(flawed_path.read_bytes()).hexdigest()

    harness.script_model()
    ns = harness.run(
        RUN_STAGE5B1A_INSTRUMENT_AMENDMENT=True,
        RECRUITED_EXPLORATORY_FLAWED_RUN_DIR=str(flawed_dir),
    )
    amendment = ns["INSTRUMENT_AMENDMENT"]
    assert amendment["scientific_recompute"] == 0
    assert amendment["corrected_classification"] == "INSTRUMENT_INCONCLUSIVE"
    assert amendment["original_report_modified"] is False
    assert amendment["original_units_modified"] is False
    assert amendment["original_verdict"] == (
        "RECRUITED_NEW_PROPERTY_EXPLORATORY_NO_GO"
    )
    assert amendment["original_report_checksum"] == (
        ns["EXPECTED_RECRUITED_EXPLORATORY_REPORT_CHECKSUM"]
    )
    assert set(amendment["omitted_integrity_clauses"]) <= set(INTEGRITY_CLAUSES)
    assert "post_cast_coordinate_error_within_tolerance" in (
        amendment["omitted_integrity_clauses"]
    )
    assert amendment["observed_post_cast_relative_errors"]["exact"] == 0.21
    # the historical run is byte-for-byte unchanged
    after = hashlib.sha256(flawed_path.read_bytes()).hexdigest()
    assert before == after
    assert json.loads(flawed_path.read_text()) == flawed_payload

    # the leg-count confirmation audit, in the same CPU-only stage
    legacy = ns["LEGACY_CONFIRMATION_AUDIT"]
    assert legacy["scientific_recompute"] == 0
    assert legacy["reaffirms_original_result"] is False
    assert legacy["invalidates_original_result"] is False
    assert legacy["realization_policy_passed"] is False
    assert legacy["verdict"] == "ARTIFACTS_INSUFFICIENT_REPLICATION_REQUIRED"
    assert legacy["required_replication"]["reproduces_stored_outcome_required"]
    assert legacy["required_replication"]["writes_to_original_run"] is False


def _completed_confirmation_fixture(tmp_path: Path, groups) -> tuple[Path, str]:
    """A stand-in for the checksum-pinned completed leg-count confirmation."""
    rows = []
    for group in groups:
        for modality in ("text", "image", "spoken_audio"):
            for condition in ("exact", "zero", "random", "unrelated"):
                moved = condition == "exact"
                rows.append({
                    "group_id": str(group["group_id"]),
                    "image_id": str(group["image_id"]),
                    "modality": modality,
                    "condition": condition,
                    "expected": "4",
                    "patched_top_token_id": 4 if moved else 2,
                    "patched_surface": "4" if moved else "2",
                    "success": moved,
                    "all_prompt_positions_patched": True,
                    "layers_patched": list(_BAND),
                    "max_activation_norm_ratio": 1.02,
                    "max_update_to_activation_norm_ratio": 0.2,
                    "max_orthogonal_residual_drift": 0.0,
                    "max_coordinate_update_error": 0.0,
                })
    payload = {
        "schema": "jlens.mmpilot.broad_pooled_multimodal_confirmation.v1",
        "verdict": "FRESH_MULTIMODAL_CONFIRMATION_GO",
        "rows": rows,
    }
    checksum = payload_checksum(payload)
    run_dir = tmp_path / "completed-confirmation"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fresh_multimodal_confirmation_report.json").write_text(
        json.dumps({**payload, "report_checksum": checksum}, indent=2),
        encoding="utf-8",
    )
    return run_dir, checksum


def _run_stage_3da(tmp_path, monkeypatch, **defects):
    harness = _Harness(tmp_path, monkeypatch, **defects)
    harness.script_model()
    run_dir, checksum = _completed_confirmation_fixture(
        tmp_path, harness.groups[:2]
    )
    ns = harness.run(
        RUN_STAGE3DA_CONFIRMATION_REALIZATION_REPLICATION=True,
        CONFIRM_CONFIRMATION_REPLICATION_BUDGET=True,
        CONFIRM_MODEL_LOAD=True,
        FRESH_CONFIRMATION_RUN_DIR=str(run_dir),
        EXPECTED_FRESH_CONFIRMATION_REPORT_CHECKSUM=checksum,
    )
    return harness, ns, run_dir, checksum


def test_stage_3da_replays_both_arms_and_never_writes_to_the_confirmed_run(
    tmp_path, monkeypatch
) -> None:
    """The smallest exact replication of the confirmed result's realization."""
    harness, ns, run_dir, checksum = _run_stage_3da(tmp_path, monkeypatch)
    report = ns["CONFIRMATION_REALIZATION_REPLICATION"]
    assert report is not None
    assert report["original_report_checksum"] == checksum
    assert report["original_report_modified"] is False
    assert report["original_units_modified"] is False
    assert report["original_verdict_relabelled"] is False
    # both arms ran, and only the corrected one carried a policy
    policies = [
        call["realization_policy"] for call in harness.calls
        if call["trial"] == "single_token_swap"
    ]
    assert policies.count(None) == len(policies) / 2
    assert policies.count(MODEL_DTYPE_REALIZATION.to_dict()) == len(policies) / 2
    assert report["n_uncorrected_rows"] == report["n_corrected_rows"]
    # nothing was written into the completed run directory
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "fresh_multimodal_confirmation_report.json"
    ]


def test_stage_3da_reports_repaired_and_preserved_when_the_outcome_holds(
    tmp_path, monkeypatch
) -> None:
    _harness, ns, _dir, _sum = _run_stage_3da(tmp_path, monkeypatch)
    report = ns["CONFIRMATION_REALIZATION_REPLICATION"]
    assert report["uncorrected_reproduced_stored_tokens"] is True
    assert report["original_within_tolerance"] is False
    assert report["corrected_realizations_converged"] is True
    assert report["n_outcomes_changed"] == 0
    assert report["verdict"] == "CONFIRMATION_REALIZATION_REPAIRED_AND_PRESERVED"


def test_stage_3da_reports_a_changed_outcome_without_relabelling_the_original(
    tmp_path, monkeypatch
) -> None:
    """A changed outcome licenses a fresh confirmation, not a retroactive verdict."""
    _harness, ns, _dir, _sum = _run_stage_3da(
        tmp_path, monkeypatch, correction_changes_outcome=True
    )
    report = ns["CONFIRMATION_REALIZATION_REPLICATION"]
    assert report["verdict"] == "CONFIRMATION_REALIZATION_REPAIRED_AND_CHANGED"
    assert report["n_outcomes_changed"] > 0
    assert report["original_verdict_relabelled"] is False
    assert "FRESH_MULTIMODAL_CONFIRMATION_GO" not in json.dumps(
        {k: v for k, v in report.items() if k != "rows"}
    )


def test_stage_3da_reports_clean_when_the_original_cast_was_within_tolerance(
    tmp_path, monkeypatch
) -> None:
    """If the confirmed run was already faithful, nothing needed repairing."""
    _harness, ns, _dir, _sum = _run_stage_3da(
        tmp_path, monkeypatch, uncorrected_relative_error=0.001
    )
    report = ns["CONFIRMATION_REALIZATION_REPLICATION"]
    assert report["original_within_tolerance"] is True
    assert report["verdict"] == "CONFIRMATION_REALIZATION_CLEAN"


def test_stage_5b1a_needs_no_model_and_spends_no_forward(
    tmp_path, monkeypatch
) -> None:
    """CPU-only really means CPU-only."""
    harness = _Harness(tmp_path, monkeypatch)
    flawed_dir = tmp_path / "flawed-run"
    flawed_dir.mkdir(parents=True)
    harness.script_model()
    ns = harness.run(
        RUN_STAGE5B1A_INSTRUMENT_AMENDMENT=True,
        RECRUITED_EXPLORATORY_FLAWED_RUN_DIR=str(flawed_dir),
    )
    assert ns["MODEL_ENABLED"] is False
    assert ns["INSTRUMENT_AMENDMENT"] is not None
    assert ns["CORRECTED_EXPLORATORY_REPORT"] is None


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
