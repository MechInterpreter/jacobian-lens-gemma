# SPDX-License-Identifier: Apache-2.0
"""Enforce the small, supported public notebook surface."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO_ROOT / "notebooks"
SUPPORTED = {
    "multimodal_jspace_coordinate_swap_mock_colab.ipynb",
    "multimodal_jspace_workspace_replication_colab.ipynb",
    "multimodal_jspace_matched_jlens_colab.ipynb",
}


def test_only_supported_notebooks_are_public_entry_points():
    found = {path.name for path in NOTEBOOKS.glob("*.ipynb")}
    assert found == SUPPORTED
    assert not list((NOTEBOOKS / "archive").rglob("*.ipynb"))


def test_readme_documents_every_supported_notebook():
    readme = (NOTEBOOKS / "README.md").read_text(encoding="utf-8")
    for name in SUPPORTED:
        assert name in readme
    assert "Git history" in readme


def test_notebook_builders_target_supported_files():
    builders = sorted(REPO_ROOT.glob("scripts/_build_*_notebook.py"))
    assert {path.name for path in builders} == {
        "_build_multimodal_lens_notebook.py",
        "_build_workspace_replication_notebook.py",
    }
    for builder in builders:
        source = builder.read_text(encoding="utf-8")
        names = set(re.findall(r"[A-Za-z0-9_]+_colab\.ipynb", source))
        assert names & SUPPORTED, f"{builder.name} targets no supported notebook"
        assert names <= SUPPORTED, f"{builder.name} targets a removed notebook: {names}"


def _markdown_files():
    return [REPO_ROOT / "README.md", *sorted(REPO_ROOT.glob("docs/*.md"))]


def test_every_notebook_path_named_in_docs_resolves():
    pattern = re.compile(r"notebooks/[A-Za-z0-9_./]+\.ipynb")
    broken = []
    for document in _markdown_files():
        for reference in pattern.findall(document.read_text(encoding="utf-8")):
            if not (REPO_ROOT / reference).is_file():
                broken.append(f"{document.name}: {reference}")
    assert not broken, f"stale notebook links: {broken}"
