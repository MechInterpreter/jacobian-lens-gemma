# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and MOCK execution of the three-modality native-audio notebook.

The execution tests run every code cell in one namespace from a working
directory where ``jlens`` is not importable, so they check the bootstrap and the
whole study at once — and, at the committed defaults, that opening the notebook
starts nothing at all.

**A passing MOCK run is plumbing evidence.** The synthetic world contains a
concept direction that is identical across text, image and audio by
construction, so a GO there says the code can find a signal that is present. It
is never evidence about Gemma, and no assertion here treats it as any.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = (
    REPO_ROOT
    / "notebooks"
    / "multimodal_jspace_spokencoco_native_audio_colab.ipynb"
)
RUNNER = Path(__file__).resolve().parent / "_mmpilot_native_audio_notebook_runner.py"
BUILDER = REPO_ROOT / "scripts" / "_build_native_audio_transfer_notebook.py"

REQUIRED_SECTIONS = [
    "bootstrap repository",
    "configuration and the staged switches",
    "mount google drive",
    "install and verify dependencies",
    "reuse the derived cache",
    "select six concepts and build the unique-image subset",
    "staged budget",
    "load gemma and resolve the **native** spoken-audio protocol",
    "load the three published, independently confirmed lenses",
    "architecture, spans and the invariance gate",
    "capability gate",
    "extract final-prompt-token activations",
    "j-space codes",
    "all six directional pairs",
    "stage a is complete",
    "stage b",
    "stage c",
    "overall verdict and report",
    "resume state",
]

COMMITTED_SWITCHES = (
    "RUN_REAL_AUDIO_TRANSFER",
    "RUN_MODEL_STAGES",
    "CONFIRM_MODEL_LOAD",
    "CONFIRM_REPRESENTATION_BUDGET",
    "RUN_L35_CAUSAL_STAGE",
    "CONFIRM_L35_CAUSAL_BUDGET",
    "RUN_L38_L40_REPLICATION",
    "CONFIRM_REPLICATION_BUDGET",
)

ALL_STAGES_ON = [f"{name}=True" for name in COMMITTED_SWITCHES[1:]]

PUBLISHED_PINS = {
    35: "sha256:64fb02d718ac48adc1bced99e2eff3c2215052ba144d5dedac05f17936a96ed1",
    38: "sha256:c8508fbf2b916e5d9aaeb8711a30f76414ee16478c5f6cc321e57e2fe846d1c0",
    40: "sha256:8a90f67eeb9bb5db14e6715b8bc516a899da1c3210d0662ec7fa177b5409f7d7",
}
AUDIO_FINGERPRINT = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells(payload):
    return [cell for cell in payload["cells"] if cell["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def _code_source(payload):
    return "\n".join("".join(cell["source"]) for cell in _code_cells(payload))


# --------------------------------------------------------------- structure


def test_notebook_is_valid_and_output_free(notebook):
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(_code_cells(notebook)):
        assert cell["outputs"] == [], f"code cell {index} has stored outputs"
        assert cell["execution_count"] is None, f"code cell {index} has an exec count"


def test_all_code_cells_parse(notebook):
    for index, cell in enumerate(_code_cells(notebook)):
        try:
            ast.parse("".join(cell["source"]))
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_the_nineteen_commissioned_stages_appear_in_order(notebook):
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(1, 20)), headings
    for (_, title), expected in zip(headings, REQUIRED_SECTIONS, strict=True):
        assert expected in title.lower(), (expected, title)


def test_the_committed_notebook_matches_its_generator():
    before = NOTEBOOK_PATH.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(BUILDER)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert NOTEBOOK_PATH.read_text(encoding="utf-8") == before


def test_every_switch_is_committed_false(notebook):
    """An *assignment* at column zero, which is what the reader edits by hand.

    The instructional blocks print `NAME = True` lines on purpose, so this
    matches assignments rather than the substring.
    """
    source = _code_source(notebook)
    for name in COMMITTED_SWITCHES:
        assignments = re.findall(
            rf"^{re.escape(name)} = (True|False)$", source, re.MULTILINE
        )
        assert assignments == ["False"], (name, assignments)


def test_real_audio_runtime_pins_the_audited_transformers_release(notebook):
    source = _code_source(notebook)
    assert '"transformers==5.13.1"' in source
    assert 'TRANSFORMERS_VERSION_EXPECTED = "5.13.1"' in source
    assert "TRANSFORMERS_VERSION != TRANSFORMERS_VERSION_EXPECTED" in source


def test_the_three_published_lens_pins_are_the_confirmed_ones(notebook):
    source = _code_source(notebook)
    for layer, checksum in PUBLISHED_PINS.items():
        assert f"lens.layer{layer}.scale100.validated.pt" in source
        assert checksum in source
    assert "rgcalib_real_7e3736b4de8f" in source


def test_layer_32_is_never_loaded_or_called_validated(notebook):
    source = _source(notebook)
    assert "lens.layer32" not in source
    assert re.search(r"layer[ -]?32\b.{0,40}validated", source, re.IGNORECASE) is None
    assert "Layer 32 and every earlier tested layer" in source


def test_the_audio_protocol_is_pinned_by_fingerprint(notebook):
    source = _code_source(notebook)
    assert AUDIO_FINGERPRINT in source
    assert "jlens.mmpilot.native_spoken_audio.v1" in source
    assert "assert_audio_protocol" in source
    assert "resolve_audio=True" in source


def test_the_real_branch_calls_the_tested_package_entry_points(notebook):
    """The real branch is package code, not notebook-only code."""
    from jlens.mmpilot import published_lens, real_backend, tri_modal

    source = _source(notebook)
    assert "from jlens.mmpilot.real_backend import build_real_backend" in source
    assert "from jlens.mmpilot.published_lens import (" in source
    assert "Gemma4PilotBackend" not in source
    assert callable(real_backend.build_real_backend)
    assert callable(published_lens.load_published_lenses)
    assert callable(tri_modal.overall_verdict)


def test_the_notebook_fits_nothing(notebook):
    source = _source(notebook)
    assert "does not fit a lens" in source
    for forbidden in ("JacobianLens.fit", "fit_lens(", "train(", "optimizer"):
        assert forbidden not in _code_source(notebook), forbidden


def test_the_model_load_is_guarded_by_both_flags(notebook):
    source = _code_source(notebook)
    assert "MODEL_STAGES_ENABLED = bool(RUN_MODEL_STAGES and CONFIRM_MODEL_LOAD)" in source
    assert source.index("if not MODEL_STAGES_ENABLED") < source.index(
        "allow_model_load=True"
    )


def test_completed_runs_are_protected_by_name(notebook):
    source = _code_source(notebook)
    for prefix in ("mmpilot_pilot_", "mmrobust_", "mmlocalize_", "rgcalib_", "audioaudit_"):
        assert f'"{prefix}"' in source
    assert "Completed runs are evidence and are never written into" in source


def test_the_scope_note_appears_in_the_notebook_prose(notebook):
    source = _source(notebook)
    assert "spoken captions" in source
    assert "environmental sound" in source
    assert "not evidence of J-space transfer" in source or (
        "not evidence of speech understanding" in source
    )


# ------------------------------------------------------------- MOCK execution


def _run(tmp_path, *overrides):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    environment["MMPILOT_REPO_DIR"] = str(REPO_ROOT)
    environment["MMPILOT_SCRATCH"] = str(tmp_path / "scratch")
    environment["MMPILOT_RUNS_ROOT"] = str(tmp_path / "runs")
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(NOTEBOOK_PATH), *overrides],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=environment,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout[-4000:]}\n\nstderr:\n{result.stderr[-2000:]}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def committed_run(tmp_path_factory):
    """Every cell at the committed defaults: nothing may run."""
    return _run(tmp_path_factory.mktemp("committed"))


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """MOCK, every stage confirmed by hand. Plumbing evidence only."""
    return _run(tmp_path_factory.mktemp("full"), *ALL_STAGES_ON)


def test_opening_the_notebook_starts_nothing(committed_run):
    assert committed_run["ok"] is True
    assert committed_run["run_real"] is False
    assert committed_run["model_stages_enabled"] is False
    assert committed_run["stage_a_enabled"] is False
    assert committed_run["stage_b_requested"] is False
    assert committed_run["stage_c_requested"] is False
    assert committed_run["model_is_none"] is True
    assert committed_run["capability_verdict"] is None
    assert committed_run["overall_verdict"] is None
    assert committed_run["completed_units"] is None


def test_the_selection_happens_without_a_model(committed_run):
    """Concepts, focal set and controls are all fixed before anything loads."""
    assert len(committed_run["selected_concepts"]) == 6
    assert committed_run["focal_concepts"] == committed_run["selected_concepts"][:3]
    assert committed_run["non_focal_concepts"] == committed_run["selected_concepts"][3:]
    assert set(committed_run["unrelated_controls"]) == set(
        committed_run["focal_concepts"]
    )
    assert not set(committed_run["unrelated_controls"].values()) & set(
        committed_run["focal_concepts"]
    )


def test_the_budget_is_printed_before_any_confirmation(committed_run):
    stages = {budget["stage"]: budget for budget in committed_run["budgets"]}
    assert sorted(stages) == ["A", "B", "C"]
    assert stages["A"]["total_passes"] > 0
    assert stages["B"]["causal_intervention_passes"] > 0
    assert stages["C"]["n_causal_cells"] == 2 * stages["B"]["n_causal_cells"]
    for budget in stages.values():
        assert budget["estimated_drive_bytes"] > 0


def test_one_group_per_photograph_and_one_recording_per_group(committed_run):
    assert committed_run["n_groups"] == committed_run["n_distinct_images"]
    assert committed_run["n_groups"] == committed_run["n_distinct_recordings"]
    assert committed_run["n_siblings_excluded"] > 0
    assert committed_run["train_heldout_image_overlap"] == []
    assert committed_run["leakage_ok"] is True


def test_the_mock_run_reaches_a_verdict_through_every_stage(full_run):
    assert full_run["run_real"] is False
    assert full_run["stage_a_enabled"] is True
    assert full_run["stage_b_requested"] is True
    assert full_run["stage_c_requested"] is True
    assert full_run["model_is_none"] is True  # the MOCK backend, never Gemma
    assert full_run["overall_verdict"] is not None
    assert full_run["report_written"] is True


def test_three_modalities_are_available_and_none_is_blocked(full_run):
    assert full_run["available_modalities"] == ["text", "image", "spoken_audio"]
    assert full_run["blocked_modalities"] == []


def test_three_lenses_are_loaded_and_each_carries_its_own_checksum(full_run):
    assert full_run["lens_layers"] == full_run["layers"]
    assert len(full_run["lens_checksums"]) == 3
    assert len(set(full_run["lens_checksums"].values())) == 3
    assert full_run["lens_combined_checksum"].startswith("sha256:")


def test_the_architecture_and_invariance_gates_run_in_every_modality(full_run):
    assert full_run["architecture_passed"] is True
    assert full_run["architecture_modalities"] == ["image", "spoken_audio", "text"]
    assert full_run["invariance_passed"] is True
    assert full_run["invariance_modalities"] == ["image", "spoken_audio", "text"]


def test_all_six_pairs_are_computed_at_every_validated_layer(full_run):
    assert full_run["representational_layers"] == full_run["layers"]
    for pairs in full_run["representational_pairs"].values():
        assert len(pairs) == 6
        assert sorted(pairs) == sorted(
            [
                "image->spoken_audio",
                "image->text",
                "spoken_audio->image",
                "spoken_audio->text",
                "text->image",
                "text->spoken_audio",
            ]
        )
    assert full_run["representational_rows"] == 6 * len(full_run["layers"])


def test_the_causal_cells_are_off_diagonal_with_every_control_and_alpha(full_run):
    assert full_run["primary_causal_layer_in_verdict"] == full_run[
        "primary_causal_layer"
    ]
    assert sorted(full_run["intervention_pairs"]) == sorted(
        [
            "image->spoken_audio",
            "image->text",
            "spoken_audio->image",
            "spoken_audio->text",
            "text->image",
            "text->spoken_audio",
        ]
    )
    # Off-diagonal only: no same-modality cell was spent.
    assert not any(
        pair.split("->")[0] == pair.split("->")[1]
        for pair in full_run["intervention_pairs"]
    )
    assert sorted(full_run["intervention_control_kinds"]) == [
        "random_norm_matched",
        "raw_residual_difference",
        "source_concept",
        "unrelated_concept",
        "zero",
    ]
    assert full_run["intervention_alphas"] == [0.0, 0.25, 0.5, 1.0]


def test_every_intervention_unit_records_its_independent_unit(full_run):
    for field in (
        "image_id",
        "image_identity_rule_version",
        "target_is_positive",
        "target_selection_version",
        "source_split",
        "target_split",
        "unrelated_control_concept",
        "direction_checksum",
    ):
        assert field in full_run["intervention_fields"], field


def test_the_five_verdicts_are_all_produced(full_run):
    assert full_run["capability_verdict"] in (
        "AUDIO_CAPABILITY_GO",
        "AUDIO_CAPABILITY_NO_GO",
    )
    assert full_run["representational_verdict"] is not None
    assert full_run["primary_causal_verdict"] is not None
    assert full_run["replication_verdict"] is not None
    assert full_run["overall_verdict"] is not None
    assert set(full_run["criteria_status"]) >= {
        "three_modality_capability",
        "audio_jspace_beats_shuffled",
        "audio_causal_cell_with_expected_sign",
        "effects_are_specific",
        "activation_norms_sane",
        "two_distinct_images_support_each_claim",
        "not_text_image_replication_alone",
        "invariance_gate",
        "environmental_audio_not_claimed",
    }


def test_the_replication_layers_ran_and_are_reported_separately(full_run):
    assert full_run["causal_layers_run"] == [
        full_run["primary_causal_layer"],
        *full_run["replication_layers"],
    ]
    assert sorted(int(k) for k in full_run["replication_per_layer"]) == sorted(
        full_run["replication_layers"]
    )


def test_the_completed_units_match_the_printed_budget(full_run):
    """The budget is derived from the configuration; this checks it was right."""
    stages = {budget["stage"]: budget for budget in full_run["budgets"]}
    units = full_run["completed_units"]
    assert units["capability"] == stages["A"]["estimated_units"]["capability"]
    assert units["activation"] == stages["A"]["estimated_units"]["activation"]
    assert units["jspace"] == stages["A"]["estimated_units"]["jspace"]
    assert units["intervention"] == (
        stages["B"]["estimated_units"]["intervention"]
        + stages["C"]["estimated_units"]["intervention"]
    )


def test_the_fingerprint_binds_the_selection_and_the_modalities(full_run):
    fingerprint = full_run["selection_fingerprint"]
    assert fingerprint["selected_concepts"] == full_run["selected_concepts"]
    assert fingerprint["focal_concepts"] == full_run["focal_concepts"]
    assert fingerprint["n_candidates_scored"] == 6
    assert fingerprint["off_diagonal_causal_only"] is True
    assert fingerprint["subset_profile"]["max_groups_per_image"] == 1
    assert full_run["fingerprint_digest"].startswith("sha256:")


# --------------------------------------------------------- staged execution


@pytest.fixture(scope="module")
def stage_a_only(tmp_path_factory):
    return _run(
        tmp_path_factory.mktemp("stage-a"),
        "RUN_MODEL_STAGES=True",
        "CONFIRM_MODEL_LOAD=True",
        "CONFIRM_REPRESENTATION_BUDGET=True",
    )


def test_stage_a_alone_spends_no_causal_pass(stage_a_only):
    assert stage_a_only["stage_a_enabled"] is True
    assert stage_a_only["stage_b_requested"] is False
    assert stage_a_only["stage_c_requested"] is False
    assert stage_a_only["completed_units"]["intervention"] == 0
    assert stage_a_only["completed_units"]["direction"] == 0
    assert stage_a_only["causal_layers_run"] == []
    assert stage_a_only["primary_causal_verdict"] is None
    assert stage_a_only["replication_verdict"] == "NOT_EVALUATED"
    # A representational result still reaches a verdict; the causal criteria
    # are NOT_EVALUATED rather than failed.
    assert stage_a_only["overall_verdict"] is not None
    assert (
        stage_a_only["criteria_status"]["audio_causal_cell_with_expected_sign"]
        == "NOT_EVALUATED"
    )


def test_confirming_the_budget_without_running_the_stage_does_nothing(tmp_path):
    """Both flags are required; either alone is inert."""
    only_confirmed = _run(
        tmp_path / "confirm-only", "CONFIRM_MODEL_LOAD=True"
    )
    assert only_confirmed["model_stages_enabled"] is False
    only_requested = _run(tmp_path / "request-only", "RUN_MODEL_STAGES=True")
    assert only_requested["model_stages_enabled"] is False


def test_stage_c_cannot_run_without_stage_b(tmp_path):
    result = _run(
        tmp_path / "c-without-b",
        "RUN_MODEL_STAGES=True",
        "CONFIRM_MODEL_LOAD=True",
        "CONFIRM_REPRESENTATION_BUDGET=True",
        "RUN_L38_L40_REPLICATION=True",
        "CONFIRM_REPLICATION_BUDGET=True",
    )
    assert result["stage_b_requested"] is False
    assert result["stage_c_requested"] is False
    assert result["completed_units"]["intervention"] == 0


# ----------------------------------------------------------------- resume


def test_a_rerun_reuses_every_unit_it_can_verify(tmp_path):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    environment["MMPILOT_REPO_DIR"] = str(REPO_ROOT)
    environment["MMPILOT_SCRATCH"] = str(tmp_path / "scratch")
    environment["MMPILOT_RUNS_ROOT"] = str(tmp_path / "runs")
    environment["MMPILOT_RUN_DIR"] = str(tmp_path / "runs" / "mmaudio_resume")

    def once():
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                str(NOTEBOOK_PATH),
                "RUN_MODEL_STAGES=True",
                "CONFIRM_MODEL_LOAD=True",
                "CONFIRM_REPRESENTATION_BUDGET=True",
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env=environment,
        )
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
        return json.loads(result.stdout.strip().splitlines()[-1])

    first = once()
    second = once()
    assert first["resume_status"] == "starting"
    assert second["resume_status"] == "resuming"
    assert first["fingerprint_digest"] == second["fingerprint_digest"]
    assert first["completed_units"] == second["completed_units"]


def test_an_incompatible_fingerprint_refuses_to_reuse_the_directory(tmp_path):
    """A run directory holds one configuration's results, or it refuses."""
    from jlens.mmpilot.store import IncompatibleStateError, RunFingerprint, UnitStore

    def fingerprint(**changes):
        base = dict(
            mode="native_audio_transfer",
            model_repo_id="google/gemma-4-E4B-it",
            model_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
            processor_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
            layers=(35, 38, 40),
            lens_checksum="sha256:combined",
            manifest_checksum="sha256:manifest",
            split_id="spokencoco-native-audio-v1",
            intervention_config={"alphas": [0.0, 0.25, 0.5, 1.0]},
            selection_config={"selected_concepts": ["a", "b"]},
            extra={
                "modalities": ["text", "image", "spoken_audio"],
                "audio_protocol_fingerprint": AUDIO_FINGERPRINT,
                "per_layer_lens_checksums": {"35": "sha256:a"},
            },
        )
        base.update(changes)
        return RunFingerprint(**base)

    root = tmp_path / "run"
    assert UnitStore(root, fingerprint()).open() == "starting"
    assert UnitStore(root, fingerprint()).open() == "resuming"

    for change in (
        {"extra": {"modalities": ["text", "image"]}},
        {"extra": {"audio_protocol_fingerprint": "sha256:" + "0" * 64}},
        {"layers": (35, 38)},
        {"lens_checksum": "sha256:another"},
        {"selection_config": {"selected_concepts": ["a", "c"]}},
    ):
        with pytest.raises(IncompatibleStateError):
            UnitStore(root, fingerprint(**change)).open()


# ------------------------------------------------------- condition isolation


def test_no_condition_ever_sees_another_channel_s_evidence():
    """Checked in package code, not promised in the notebook's prose."""
    from jlens.mmpilot.audio import assert_no_text_leakage
    from jlens.mmpilot.capability import build_prompt, build_question

    question = build_question(["bus", "cat"])
    caption = "a photo showing a cat near a window"
    text_prompt = build_prompt(question, modality="text", caption=caption)
    image_prompt = build_prompt(question, modality="image", caption=caption)
    audio_prompt = build_prompt(question, modality="spoken_audio", caption=caption)

    assert caption in text_prompt
    assert caption not in image_prompt
    assert caption not in audio_prompt
    assert image_prompt == audio_prompt == question
    # The audio condition additionally refuses filenames and image ids.
    assert_no_text_leakage(
        audio_prompt, forbidden=[caption, "img0001", "COCO_train2014_000001"]
    )
    with pytest.raises(ValueError, match="leaks non-audio evidence"):
        assert_no_text_leakage(text_prompt, forbidden=[caption])


def test_the_question_and_candidate_set_are_identical_across_modalities():
    from jlens.mmpilot.capability import build_ordered_questions, build_prompt

    concepts = ["bus", "cat", "clock"]
    questions = build_ordered_questions(concepts)
    assert len(questions) == 2  # canonical and reversed option order
    for question in questions:
        image = build_prompt(question, modality="image")
        audio = build_prompt(question, modality="spoken_audio")
        text = build_prompt(question, modality="text", caption="a cat")
        assert image == audio == question
        assert text.endswith(question)
        for concept in concepts:
            assert concept in question
