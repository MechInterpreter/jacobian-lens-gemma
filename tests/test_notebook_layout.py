# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The shape of ``notebooks/``, enforced.

The archive only helps if it stays an archive. Two things can quietly undo it:
a builder still writing to the old top-level path (so re-running it recreates a
duplicate), and a doc link still pointing at a file that has moved. Both are
cheap to check and neither is visible in review.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO_ROOT / "notebooks"
ARCHIVE = NOTEBOOKS / "archive"

#: The active research workflow. Exactly these, at the top level.
ACTIVE = {
    "research_grade_multilayer_jlens_calibration_colab.ipynb",
    "research_grade_early_layer_jlens_extension_colab.ipynb",
    "multimodal_jspace_spokencoco_native_audio_colab.ipynb",
    "multimodal_jspace_spokencoco_l32_followup_colab.ipynb",
    "multimodal_jspace_coordinate_swap_mock_colab.ipynb",
    "multimodal_jspace_l32_convergence_resolution_colab.ipynb",
    "research_grade_l27_l31_preconvergence_study_colab.ipynb",
    "multimodal_jspace_anthropic_reasoning_swap_colab.ipynb",
    "multimodal_jspace_anthropic_band_swap_colab.ipynb",
    "multimodal_jspace_anthropic_band33_40_swap_colab.ipynb",
    "multimodal_jspace_full_vocabulary_causal_validation_colab.ipynb",
    "multimodal_jspace_digit_reasoning_confirmation_colab.ipynb",
}

#: The archive categories, and what each is for.
CATEGORIES = {
    "completed_studies": {
        "multimodal_jspace_spokencoco_pilot_colab.ipynb",
        "multimodal_jspace_spokencoco_robustness_colab.ipynb",
        "multimodal_jspace_layer_localization_colab.ipynb",
        "multimodal_jspace_output_convergence_audit_colab.ipynb",
        "gemma4_native_spoken_audio_feasibility_colab.ipynb",
        "gemma_4_e4b_layer32_confirmatory_validation_colab.ipynb",
    },
    "engineering_audits": {
        "mmpilot_image_independence_audit_colab.ipynb",
        "gemma_4_e4b_layer21_diagnostic.ipynb",
    },
    "legacy_prototypes": {
        "gemma_4_e4b_text_jlens.ipynb",
        "gemma_4_e4b_jspace_pursuit.ipynb",
        "gemma_4_e4b_jspace_causal_smoke.ipynb",
        "gemma_4_e4b_multimodal_jlens_capture.ipynb",
        "gemma_4_e4b_text_jlens_recalibration_colab.ipynb",
    },
    "protocol_mocks": {
        "open_prompt_protocol_mock_colab.ipynb",
    },
}

ARCHIVED = {name for names in CATEGORIES.values() for name in names}


# ------------------------------------------------------------------- layout


def test_only_the_active_workflow_sits_at_the_top_level():
    found = {path.name for path in NOTEBOOKS.glob("*.ipynb")}
    assert found == ACTIVE


@pytest.mark.parametrize("category", sorted(CATEGORIES))
def test_each_archive_category_holds_exactly_what_it_should(category):
    found = {path.name for path in (ARCHIVE / category).glob("*.ipynb")}
    assert found == CATEGORIES[category]


def test_no_notebook_exists_in_two_places():
    everywhere: dict[str, list[str]] = {}
    for path in NOTEBOOKS.rglob("*.ipynb"):
        everywhere.setdefault(path.name, []).append(
            str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        )
    duplicates = {name: paths for name, paths in everywhere.items() if len(paths) > 1}
    assert not duplicates, f"a notebook exists at two paths: {duplicates}"


def test_the_readme_documents_every_notebook():
    readme = (NOTEBOOKS / "README.md").read_text(encoding="utf-8")
    for name in ACTIVE | ARCHIVED:
        assert name in readme, f"{name} is not documented in notebooks/README.md"


def test_the_readme_warns_against_rerunning_historical_studies():
    readme = (NOTEBOOKS / "README.md").read_text(encoding="utf-8").lower()
    assert "read this before running anything in `archive/`" in readme
    assert "different measurements" in readme
    assert "older protocol" in readme


def test_the_readme_names_the_archive_categories():
    readme = (NOTEBOOKS / "README.md").read_text(encoding="utf-8")
    for category in CATEGORIES:
        assert f"archive/{category}/" in readme


def test_the_readme_states_which_notebook_is_canonical_for_which_claim():
    readme = (NOTEBOOKS / "README.md").read_text(encoding="utf-8")
    assert "canonical" in readme.lower()
    assert "Which notebook is canonical for which claim" in readme


# ------------------------------------------------------------------ builders


def _builders():
    return sorted(REPO_ROOT.glob("scripts/_build_*_notebook.py"))


def test_every_builder_targets_the_path_its_notebook_actually_lives_at():
    for builder in _builders():
        source = builder.read_text(encoding="utf-8")
        name = re.search(r'"([^"]+\.ipynb)"', source)
        assert name, f"{builder.name} names no notebook"
        matches = list(NOTEBOOKS.rglob(name.group(1)))
        assert len(matches) == 1, f"{name.group(1)} is not at exactly one path"
        expected = matches[0].relative_to(NOTEBOOKS).parts[:-1]
        for part in expected:
            assert f'"{part}"' in source, (
                f"{builder.name} does not write into notebooks/{'/'.join(expected)}; "
                "re-running it would recreate a duplicate at the old path"
            )


def test_no_builder_writes_to_a_bare_top_level_archived_name():
    for builder in _builders():
        source = builder.read_text(encoding="utf-8")
        for name in ARCHIVED:
            if f'"{name}"' not in source:
                continue
            assert '"archive"' in source, (
                f"{builder.name} still writes {name} to the top level"
            )


# ---------------------------------------------------------------- doc links


def _markdown_files():
    return [REPO_ROOT / "README.md", *sorted(REPO_ROOT.glob("docs/*.md"))]


def test_every_notebook_path_named_in_the_docs_resolves():
    pattern = re.compile(r"notebooks/[A-Za-z0-9_./]+\.ipynb")
    broken: list[str] = []
    for document in _markdown_files():
        for reference in pattern.findall(document.read_text(encoding="utf-8")):
            if not (REPO_ROOT / reference).is_file():
                broken.append(f"{document.name}: {reference}")
    assert not broken, f"stale notebook links: {broken}"


def test_every_notebook_path_named_in_the_package_resolves():
    pattern = re.compile(r"notebooks/[A-Za-z0-9_./]+\.ipynb")
    broken: list[str] = []
    for source in sorted(REPO_ROOT.glob("jlens/**/*.py")):
        text = source.read_text(encoding="utf-8")
        # Rejoin paths that were wrapped across two adjacent string literals.
        text = re.sub(r'"\s*\n\s*"', "", text)
        for reference in pattern.findall(text):
            if not (REPO_ROOT / reference).is_file():
                broken.append(f"{source.name}: {reference}")
    assert not broken, f"stale notebook links in the package: {broken}"
