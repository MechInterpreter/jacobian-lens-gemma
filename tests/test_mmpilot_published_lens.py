# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The published-lens loader: what it accepts, and everything it refuses.

The accept path is one test. The rest are refusals, because the loader's job is
to be the thing that stops an incompatible artifact — a lens that was never
confirmed, one fitted at another scale, one for another model, one for another
layer, or the layer-32 record that exists precisely so it can be refused.

Every fixture is written by :func:`jlens.mmpilot.mock.build_mock_published_lenses`
in the *real* calibration artifact format, so these run on CPU in milliseconds
without a 16 GB checkpoint and still exercise the schema the real run reads.
"""

import json
from pathlib import Path

import pytest

from jlens.mmpilot.mock import (
    MOCK_FAILED_CONFIRMATION_LAYER,
    MOCK_PUBLISHED_LAYERS,
    build_mock_published_lenses,
)
from jlens.mmpilot.published_lens import (
    CONFIRMED_LAYERS,
    EXPECTED_ARTIFACT_FORMAT,
    FAILED_CONFIRMATION_LAYERS,
    REQUIRED_ARTIFACT_FIELDS,
    PublishedLensRefused,
    PublishedLensSpec,
    artifact_schema_report,
    combined_lens_checksum,
    format_lens_report,
    load_published_lenses,
    read_artifact_sidecar,
)
from jlens.mmpilot.store import payload_checksum


@pytest.fixture
def published(tmp_path):
    return build_mock_published_lenses(tmp_path / "published")


def _rewrite(spec: PublishedLensSpec, **changes) -> None:
    """Edit an artifact's sidecar in place and re-seal its own checksum.

    Re-sealing matters: an edit that left ``artifact_checksum`` stale would be
    caught by the intactness clause, and then the test would not be testing the
    clause it names.
    """
    sidecar = Path(spec.path).with_suffix(".json")
    artifact = json.loads(sidecar.read_text(encoding="utf-8"))
    artifact.update(changes)
    artifact.pop("artifact_checksum", None)
    artifact["artifact_checksum"] = payload_checksum(artifact)
    sidecar.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


# ------------------------------------------------------------- the accept path


def test_the_three_published_layers_load_and_assemble(published):
    loaded = load_published_lenses(published["specs"], published["expectations"])

    assert loaded.layers == tuple(MOCK_PUBLISHED_LAYERS)
    assert sorted(loaded.lens.jacobians) == list(MOCK_PUBLISHED_LAYERS)
    assert loaded.n_prompts == 100
    assert set(loaded.checksums) == set(MOCK_PUBLISHED_LAYERS)
    assert loaded.combined_checksum == combined_lens_checksum(loaded.checksums)
    assert all(value.startswith("sha256:") for value in loaded.checksums.values())
    payload = loaded.to_dict()
    assert payload["frozen"] is True
    assert payload["fitted_here"] is False


def test_every_commissioned_property_is_actually_checked(published):
    loaded = load_published_lenses(published["specs"], published["expectations"])
    checks = {
        entry["check"]
        for record in loaded.validations.values()
        for entry in record["checks"]
    }
    for name in (
        "publication_status",
        "confirmation_status",
        "lens_checksum_matches_pin",
        "model_repo_id",
        "model_revision",
        "tokenizer_revision",
        "fitted_scale",
        "calibration_modality",
        "physical_layer",
        "d_model",
        "hook_site",
        "residual_convention",
        "vector_orientation",
        "normalization_convention",
        "frozen_status",
        "layer_independently_confirmed",
        "no_multimodal_calibration",
    ):
        assert name in checks, name
    assert all(record["passed"] for record in loaded.validations.values())


def test_the_report_names_the_confirmed_layers_and_the_failed_one(published):
    loaded = load_published_lenses(published["specs"], published["expectations"])
    text = format_lens_report(loaded)
    assert "nothing is fitted here" in text
    assert "text-only" in text
    for layer in MOCK_PUBLISHED_LAYERS:
        assert f"layer {layer}" in text
    assert str(FAILED_CONFIRMATION_LAYERS[0]) in text


def test_the_confirmed_layer_set_is_the_calibration_run_s(published):
    """35/38/40 passed; 32 and everything earlier failed. Stated, not inferred."""
    assert CONFIRMED_LAYERS == (35, 38, 40)
    assert FAILED_CONFIRMATION_LAYERS == (32,)
    assert 32 not in CONFIRMED_LAYERS


# ------------------------------------------------------------- the refusals


def test_a_failed_confirmation_artifact_is_refused(published):
    """The layer-32 case: a record that exists so it can be refused."""
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(
            [published["failed_spec"]], published["expectations"]
        )
    problems = " ".join(error.value.problems)
    assert "publication_status" in problems
    assert "confirmation_status" in problems
    assert "layer_independently_confirmed" in problems


def test_a_layer_outside_the_confirmed_set_is_refused(tmp_path):
    built = build_mock_published_lenses(tmp_path / "published")
    expectations = built["expectations"]
    narrowed = type(expectations)(
        model_repo_id=expectations.model_repo_id,
        model_revision=expectations.model_revision,
        scale_point=expectations.scale_point,
        d_model=expectations.d_model,
        confirmed_layers=(MOCK_PUBLISHED_LAYERS[0],),
        failed_confirmation_layers=(MOCK_PUBLISHED_LAYERS[-1],),
    )
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses([built["specs"][-1]], narrowed)
    assert "layer_independently_confirmed" in " ".join(error.value.problems)
    assert "failed confirmation" in str(error.value)


def test_a_wrong_checksum_pin_is_refused(published):
    spec = published["specs"][0]
    wrong = PublishedLensSpec(
        layer=spec.layer, path=spec.path, expect_sha256="sha256:" + "0" * 64
    )
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses([wrong], published["expectations"])
    assert "lens_checksum_matches_pin" in " ".join(error.value.problems)


def test_a_wrong_layer_is_refused(published):
    spec = published["specs"][0]
    _rewrite(spec, physical_layer=spec.layer + 100)
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses([spec], published["expectations"])
    assert "physical_layer" in " ".join(error.value.problems)


def test_a_wrong_model_revision_is_refused(published):
    _rewrite(published["specs"][0], model_revision="deadbeef" * 5)
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    assert "model_revision" in " ".join(error.value.problems)


def test_a_wrong_tokenizer_revision_is_refused(published):
    _rewrite(published["specs"][0], tokenizer_revision="another-revision")
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    assert "tokenizer_revision" in " ".join(error.value.problems)


def test_a_wrong_fitted_scale_is_refused(published):
    _rewrite(published["specs"][0], scale_point=250, n_fitting_prompts=250)
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    assert "fitted_scale" in " ".join(error.value.problems)


def test_an_unvalidated_artifact_is_refused(published):
    _rewrite(published["specs"][0], validated=False)
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    assert "publication_status" in " ".join(error.value.problems)


def test_a_multimodal_calibration_is_refused(published):
    """A lens that saw SpokenCOCO cannot answer 'does a text lens transfer'."""
    _rewrite(published["specs"][0], spokencoco_used=True, multimodal_data_used=True)
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    assert "no_multimodal_calibration" in " ".join(error.value.problems)


def test_a_non_text_calibration_modality_is_refused(published):
    _rewrite(published["specs"][0], calibration_modality="multimodal")
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    assert "calibration_modality" in " ".join(error.value.problems)


def test_a_tampered_record_is_refused(published):
    """An edit that does not re-seal the record is caught by the record itself."""
    sidecar = Path(published["specs"][0].path).with_suffix(".json")
    artifact = json.loads(sidecar.read_text(encoding="utf-8"))
    artifact["hook_site"] = "something_else"
    sidecar.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    problems = " ".join(error.value.problems)
    assert "artifact_record_is_intact" in problems
    assert "hook_site" in problems


def test_a_missing_sidecar_is_refused(published):
    Path(published["specs"][0].path).with_suffix(".json").unlink()
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    assert "sidecar_present" in error.value.problems
    assert "no publication or confirmation status" in str(error.value)


def test_a_missing_required_field_is_refused_not_defaulted(published):
    """Absence is never a False. It is an artifact this code has not seen."""
    sidecar = Path(published["specs"][0].path).with_suffix(".json")
    artifact = json.loads(sidecar.read_text(encoding="utf-8"))
    del artifact["confirmation_protocol"]
    artifact.pop("artifact_checksum", None)
    artifact["artifact_checksum"] = payload_checksum(artifact)
    sidecar.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    problems = " ".join(error.value.problems)
    assert "artifact_schema_complete" in problems
    assert "confirmation_protocol" in problems


def test_an_unknown_artifact_format_is_refused(published):
    _rewrite(published["specs"][0], artifact_format_version="some.other.format.v9")
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    assert "artifact_format_version" in " ".join(error.value.problems)


def test_a_missing_lens_file_is_refused(published):
    Path(published["specs"][0].path).unlink()
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses(published["specs"], published["expectations"])
    assert "lens_file_exists" in error.value.problems
    assert "does not fit a lens" in str(error.value)


def test_a_duplicate_layer_is_refused(published):
    spec = published["specs"][0]
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses([spec, spec], published["expectations"])
    assert "specs_distinct" in error.value.problems


def test_artifacts_fitted_at_different_scales_do_not_form_one_lens(tmp_path):
    a = build_mock_published_lenses(
        tmp_path / "a", layers=(2,), scale=100, include_failed_layer=False
    )
    b = build_mock_published_lenses(
        tmp_path / "b", layers=(3,), scale=50, include_failed_layer=False
    )
    # Both sidecars now claim scale 100; only the lens *files* disagree, so the
    # per-artifact clauses all pass and the assembly clause is what has to fire.
    _rewrite(b["specs"][0], scale_point=100, n_fitting_prompts=100)
    expectations = type(a["expectations"])(
        model_repo_id=a["expectations"].model_repo_id,
        model_revision=a["expectations"].model_revision,
        scale_point=100,
        d_model=a["expectations"].d_model,
        confirmed_layers=(2, 3),
        failed_confirmation_layers=(),
    )
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses([*a["specs"], *b["specs"]], expectations)
    assert "scale_agreement" in error.value.problems


def test_a_file_that_does_not_hold_its_recorded_layer_is_refused(tmp_path):
    built = build_mock_published_lenses(
        tmp_path / "published", layers=(2,), include_failed_layer=False
    )
    other = build_mock_published_lenses(
        tmp_path / "other", layers=(3,), include_failed_layer=False
    )
    # The layer-2 record, pointing at the layer-3 bytes, with both checksums
    # honest — only the *contents* disagree.
    victim = built["specs"][0]
    Path(victim.path).write_bytes(Path(other["specs"][0].path).read_bytes())
    from jlens.metadata import file_sha256

    checksum = file_sha256(str(victim.path))
    _rewrite(victim, lens_checksum=checksum)
    spec = PublishedLensSpec(
        layer=victim.layer, path=victim.path, expect_sha256=checksum
    )
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses([spec], built["expectations"])
    assert "file_contains_recorded_layer" in error.value.problems


def test_no_specs_is_refused():
    with pytest.raises(PublishedLensRefused) as error:
        load_published_lenses([], build_mock_published_lenses.__defaults__ and None)
    assert "specs_present" in error.value.problems


# ------------------------------------------------------------------ schema


def test_the_schema_report_lists_what_is_actually_present(published):
    _sidecar, artifact = read_artifact_sidecar(published["specs"][0].path)
    report = artifact_schema_report(artifact)
    assert report["artifact_format_version"] == EXPECTED_ARTIFACT_FORMAT
    assert report["missing_required_fields"] == []
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(report["observed_keys"])
    assert report["n_keys"] == len(report["observed_keys"])


def test_the_mock_fixture_is_written_in_the_real_publication_format(published):
    """If the fixture drifts from `build_artifact`, these tests stop meaning much."""
    from jlens.calibration.publication import ARTIFACT_FORMAT_VERSION

    _sidecar, artifact = read_artifact_sidecar(published["specs"][0].path)
    assert artifact["artifact_format_version"] == ARTIFACT_FORMAT_VERSION
    assert artifact["objective"] == "not_applicable_estimator_is_a_sample_mean"
    # `build_artifact` writes no `published` key at all; only
    # `record_failed_layer` does, and it writes False. Absence must therefore
    # never be read as a refusal, and False always must be.
    assert "published" not in artifact
    _sidecar, failed = read_artifact_sidecar(published["failed_spec"].path)
    assert failed["published"] is False
    assert failed["physical_layer"] == MOCK_FAILED_CONFIRMATION_LAYER
