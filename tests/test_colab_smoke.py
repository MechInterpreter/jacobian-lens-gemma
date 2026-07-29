# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/colab_smoke.py's repo-root/branch bootstrap, in
particular the ``colab exec -f`` path where the source is exec'd with no
``__file__`` in its globals."""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_script():
    spec = importlib.util.spec_from_file_location(
        "colab_smoke", REPO_ROOT / "scripts" / "colab_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def colab_smoke():
    return _import_script()


def test_resolve_repo_root_with_file_dunder(colab_smoke):
    fake_script = os.path.join("fake", "checkout", "scripts", "colab_smoke.py")
    expected = os.path.dirname(os.path.dirname(os.path.abspath(fake_script)))
    assert colab_smoke.resolve_repo_root(fake_script) == expected


def test_resolve_repo_root_without_file_dunder_finds_content_checkout(
    colab_smoke, tmp_path
):
    """The ``colab exec -f`` case: __file__ is absent, so the pushed
    checkout under content_dir must be located instead."""
    checkout = tmp_path / "jacobian-lens-gemma"
    (checkout / "jlens").mkdir(parents=True)

    root = colab_smoke.resolve_repo_root(None, content_dir=str(tmp_path))

    assert root == str(checkout)


def test_resolve_repo_root_without_file_dunder_raises_when_missing(
    colab_smoke, tmp_path
):
    with pytest.raises(RuntimeError, match="clone it first"):
        colab_smoke.resolve_repo_root(None, content_dir=str(tmp_path))


def test_resolve_under_root_leaves_absolute_paths(colab_smoke):
    abs_path = os.path.abspath(os.path.join("some", "abs", "path"))
    assert colab_smoke.resolve_under_root(abs_path, "/other/root") == abs_path


def test_resolve_under_root_joins_relative_paths(colab_smoke):
    relative = os.path.join("configs", "x.json")
    assert colab_smoke.resolve_under_root(relative, "/root") == os.path.join(
        "/root", relative
    )


def test_main_works_from_outside_repo_directory(colab_smoke, tmp_path, monkeypatch):
    """Regression: main() previously opened the benchmark manifest relative
    to the process's current directory rather than REPO_ROOT, so running it
    from anywhere but the repo checkout crashed with FileNotFoundError."""
    monkeypatch.chdir(tmp_path)

    assert colab_smoke.main() == 0


def test_ensure_branch_is_a_no_op_when_already_current(colab_smoke, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-b", "feature"], check=True)
    (repo / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "init"],
        check=True,
        capture_output=True,
    )

    branch = colab_smoke.ensure_branch(str(repo), branch="feature")

    assert branch == "feature"
