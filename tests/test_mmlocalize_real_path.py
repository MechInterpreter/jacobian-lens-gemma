# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The real path, executed without weights: lens loading and preflight.

MOCK execution cannot catch a defect in a branch it never enters, and
string-matching tests cannot catch one that looks fine as a string. So these
tests run the **actual** real-path code — the same lens loading, the same
signature binding, the same preflight — with only the network touched by a stub.

The specific thing being protected here is the separation between *fitted* and
*certified*. The robustness study's loader requires every requested layer to
appear in the manifest's passing list, which is right for a study that uses one
already-validated layer and wrong for this one, whose whole purpose is to test
layers the manifest has not certified.
"""

import json

import pytest
import torch

from jlens.lens import JacobianLens
from jlens.mmlocalize.layers import LOCALIZATION_LAYERS
from jlens.mmlocalize.real_path import (
    EXPECTED_HOOK_SITE,
    LocalizationLensError,
    LocalizationPreflightError,
    check_localization_call_contracts,
    format_preflight,
    load_lens_for_localization,
    localization_preflight,
)
from jlens.mmpilot.real_backend import LensManifestError, load_validated_lens

D_MODEL = 8
REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"


@pytest.fixture
def published_lens(tmp_path):
    """A lens fitted at all four layers, but certified only at 38."""
    from jlens.metadata import file_sha256

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    lens_path = artifacts / "lens.validated.pt"
    JacobianLens(
        jacobians={layer: torch.eye(D_MODEL) for layer in LOCALIZATION_LAYERS},
        n_prompts=32,
        d_model=D_MODEL,
    ).save(str(lens_path))
    checksum = file_sha256(str(lens_path))
    (artifacts / "validated_lens_manifest.json").write_text(
        json.dumps(
            {
                "status": "validated_text_only",
                "lens_checksum": checksum,
                "model_revision": REVISION,
                "source_layers": list(LOCALIZATION_LAYERS),
                "native_readout_layers_passing": [38],
                "prompt_protocol": "text-only WikiText chat rendering",
                "native_validation_path": "artifacts/native_readout_validation.json",
            }
        ),
        encoding="utf-8",
    )
    return lens_path, checksum


# ------------------------------------------------------------ lens loading


def test_the_robustness_loader_refuses_the_uncertified_layers(published_lens):
    """Establishes the premise: the older loader cannot ask this question."""
    lens_path, checksum = published_lens
    with pytest.raises(LensManifestError, match="natively validated"):
        load_validated_lens(
            lens_path,
            expect_checksum=checksum,
            layers=LOCALIZATION_LAYERS,
            model_revision=REVISION,
        )


def test_the_localization_loader_accepts_fitted_but_uncertified_layers(published_lens):
    lens_path, checksum = published_lens
    loaded = load_lens_for_localization(
        lens_path,
        expect_checksum=checksum,
        layers=LOCALIZATION_LAYERS,
        model_revision=REVISION,
        expect_d_model=D_MODEL,
    )
    assert loaded.fitted_layers == [20, 26, 32, 38]
    assert loaded.natively_validated_layers == [38]
    assert loaded.layers_under_test == [20, 26, 32]
    assert "carry no causal claim" in loaded.to_dict()["reading"]


def test_an_unfitted_layer_is_a_missing_artifact_not_a_testable_one(published_lens):
    lens_path, checksum = published_lens
    with pytest.raises(LocalizationLensError, match="no fitted Jacobian"):
        load_lens_for_localization(
            lens_path,
            expect_checksum=checksum,
            layers=(14, 20, 26, 32, 38),
            model_revision=REVISION,
        )


def test_a_lens_other_than_the_pinned_one_is_refused(published_lens):
    lens_path, _ = published_lens
    with pytest.raises(LocalizationLensError, match="refusing to use a lens"):
        load_lens_for_localization(
            lens_path,
            expect_checksum="sha256:not-this-one",
            layers=LOCALIZATION_LAYERS,
            model_revision=REVISION,
        )


def test_a_missing_lens_names_the_missing_artifact(tmp_path):
    with pytest.raises(LocalizationLensError, match="does not fit a lens"):
        load_lens_for_localization(
            tmp_path / "absent.pt",
            expect_checksum="sha256:x",
            layers=LOCALIZATION_LAYERS,
            model_revision=REVISION,
        )


def test_a_lens_without_its_manifest_is_an_unvalidated_lens(published_lens):
    lens_path, checksum = published_lens
    (lens_path.parent / "validated_lens_manifest.json").unlink()
    with pytest.raises(LocalizationLensError, match="unvalidated lens"):
        load_lens_for_localization(
            lens_path,
            expect_checksum=checksum,
            layers=LOCALIZATION_LAYERS,
            model_revision=REVISION,
        )


def test_a_lens_calibrated_on_anything_but_text_is_refused(published_lens):
    lens_path, checksum = published_lens
    manifest_path = lens_path.parent / "validated_lens_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_protocol"] = "text-only plus SpokenCOCO image captions"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LocalizationLensError, match="non-text data"):
        load_lens_for_localization(
            lens_path,
            expect_checksum=checksum,
            layers=LOCALIZATION_LAYERS,
            model_revision=REVISION,
        )


def test_a_lens_validated_against_another_revision_is_refused(published_lens):
    lens_path, checksum = published_lens
    with pytest.raises(LocalizationLensError, match="model revision"):
        load_lens_for_localization(
            lens_path,
            expect_checksum=checksum,
            layers=LOCALIZATION_LAYERS,
            model_revision="0000000000000000000000000000000000000000",
        )


def test_a_width_mismatch_is_refused(published_lens):
    lens_path, checksum = published_lens
    with pytest.raises(LocalizationLensError, match="d_model"):
        load_lens_for_localization(
            lens_path,
            expect_checksum=checksum,
            layers=LOCALIZATION_LAYERS,
            model_revision=REVISION,
            expect_d_model=2560,
        )


# --------------------------------------------------------- call contracts


def test_every_real_path_call_binds_against_the_installed_signatures():
    """The class of drift that reached three L4 starts instead of CI."""
    assert check_localization_call_contracts() == []


# -------------------------------------------------------------- preflight


def _manifest_payload(layers=LOCALIZATION_LAYERS, **overrides):
    payload = {
        "concepts": ["cat", "toilet"],
        "policy": "fresh_image_disjoint_from_completed_run.v1",
        "layers": [int(x) for x in layers],
        "target_checksum": "sha256:frozen",
        "frozen_before_any_layer_result": True,
        "same_targets_at_every_layer": True,
        "image_exclusion_audit": {
            "source_target_overlap": [],
            "fresh_policy_satisfied": True,
            "n_overlap_all": 0,
            "n_overlap_causal_targets": 0,
        },
    }
    payload.update(overrides)
    return payload


def _fingerprint_fields(**overrides):
    fields = {
        "model_repo_id": "google/gemma-4-E4B-it",
        "model_revision": REVISION,
        "processor_revision": REVISION,
        "lens_checksum": "sha256:x",
        "calibration_protocol": "text-only",
        "layers": [20, 26, 32, 38],
        "validity_gate_digest": "sha256:gate",
        "manifest_checksum": "sha256:manifest",
        "target_checksum": "sha256:frozen",
        "source_image_ids": ["a"],
        "target_image_ids": ["b"],
        "concepts": ["cat", "toilet"],
        "prompt_protocol": "v1",
        "alphas": [0.0, 0.25, 0.5],
        "controls": ["permuted"],
        "pursuit_config": {"k": 25},
    }
    fields.update(overrides)
    return fields


def _preflight(published_lens, tmp_path, **overrides):
    lens_path, checksum = published_lens
    kwargs = {
        "model_repo_id": "google/gemma-4-E4B-it",
        "model_revision": REVISION,
        "lens_path": lens_path,
        "lens_expect_checksum": checksum,
        "layers": LOCALIZATION_LAYERS,
        "expect_d_model": D_MODEL,
        "target_manifest": _manifest_payload(),
        "completed_run": {
            "fingerprint_matches_pin": True,
            "run_dir": "/runs/completed",
            "fingerprint": "sha256:completed",
            "verdict": "ROBUSTNESS_GO",
        },
        "fingerprint_fields": _fingerprint_fields(),
        "runs_root": tmp_path,
        "resolve_hub_revision": False,
    }
    kwargs.update(overrides)
    return localization_preflight(**kwargs)


def test_a_complete_configuration_passes_preflight(published_lens, tmp_path):
    report = _preflight(published_lens, tmp_path)
    assert report["passed"] is True
    names = {check["check"] for check in report["checks"]}
    for required in (
        "layer_set_is_immutable",
        "lens_identity_and_layers_fitted",
        "lens_d_model",
        "reference_layer_is_already_certified",
        "hook_site_convention",
        "model_depth_expectation",
        "hub_revision_resolves",
        "call_signatures_bind",
        "completed_run_verified",
        "target_manifest_frozen",
        "target_manifest_covers_every_layer",
        "targets_concepts_match",
        "source_target_images_disjoint",
        "target_policy_satisfied",
        "run_fingerprint_complete",
        "storage_available",
    ):
        assert required in names, required
    assert "REAL PATH PREFLIGHT: PASS" in format_preflight(report)


def test_preflight_reports_every_problem_at_once(published_lens, tmp_path):
    """One Colab round trip should surface all of them, not the first."""
    with pytest.raises(LocalizationPreflightError) as error:
        _preflight(
            published_lens,
            tmp_path,
            target_manifest=_manifest_payload(
                layers=(20, 26),
                concepts=["cat", "dog"],
                image_exclusion_audit={
                    "source_target_overlap": ["shared"],
                    "fresh_policy_satisfied": False,
                    "n_overlap_all": 3,
                    "n_overlap_causal_targets": 1,
                },
            ),
            fingerprint_fields=_fingerprint_fields(target_checksum=None, concepts=[]),
        )
    message = str(error.value)
    assert "the model download must not start" in message
    for expected in (
        "target_manifest_covers_every_layer",
        "targets_concepts_match",
        "source_target_images_disjoint",
        "target_policy_satisfied",
        "run_fingerprint_complete",
    ):
        assert expected in message


def test_a_changed_layer_set_fails_preflight(published_lens, tmp_path):
    with pytest.raises(LocalizationPreflightError, match="layer_set_is_immutable"):
        _preflight(
            published_lens,
            tmp_path,
            layers=(32,),
            target_manifest=_manifest_payload(layers=(32,)),
            fingerprint_fields=_fingerprint_fields(layers=[32]),
        )


def test_an_incomplete_fingerprint_fails_preflight(published_lens, tmp_path):
    with pytest.raises(LocalizationPreflightError, match="run_fingerprint_complete"):
        _preflight(
            published_lens,
            tmp_path,
            fingerprint_fields=_fingerprint_fields(validity_gate_digest=None),
        )


def test_an_unverified_completed_run_fails_preflight(published_lens, tmp_path):
    with pytest.raises(LocalizationPreflightError, match="completed_run_verified"):
        _preflight(
            published_lens,
            tmp_path,
            completed_run={"fingerprint_matches_pin": False, "run_dir": "/x"},
        )


def test_a_missing_lens_fails_preflight_before_any_download(published_lens, tmp_path):
    lens_path, checksum = published_lens
    with pytest.raises(LocalizationPreflightError, match="lens_file_exists"):
        _preflight(published_lens, tmp_path, lens_path=tmp_path / "absent.pt")


def test_preflight_never_loads_model_weights(published_lens, tmp_path, monkeypatch):
    """The point of the whole cell: nothing that costs 16 GB may run here."""
    import jlens.gemma4

    def _explode(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("preflight attempted to load model weights")

    monkeypatch.setattr(jlens.gemma4, "load_gemma4", _explode, raising=False)
    assert _preflight(published_lens, tmp_path)["passed"] is True


def test_the_hub_check_reports_a_failure_rather_than_downloading(
    published_lens, tmp_path, monkeypatch
):
    import jlens.mmpilot.preflight as preflight_module

    def _unreachable(repo_id, revision, token):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(preflight_module, "_hub_model_info", _unreachable)
    with pytest.raises(LocalizationPreflightError, match="hub_revision_resolves"):
        _preflight(published_lens, tmp_path, resolve_hub_revision=True)


def test_the_hook_site_convention_is_pinned():
    assert EXPECTED_HOOK_SITE == "block_output"
