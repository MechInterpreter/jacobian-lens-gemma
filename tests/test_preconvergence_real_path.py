# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Executable validation of the L27-L31 notebook's **real** branch, no weights.

The pattern this file exists to stop: MOCK passes, and then the real-only Colab
branch dies on a symbol that does not exist or a keyword that was renamed. MOCK
execution cannot catch it because the MOCK branch never runs those lines, and a
string-matching test cannot catch it because the strings look fine.

So the notebook's real model-loading cell is **executed here, verbatim**, with
only three things replaced:

* the ~16 GB model loader (:func:`jlens.mmpilot.real_backend._loader`);
* the processor factory (:func:`jlens.mmpilot.real_backend._processor_factory`);
* nothing else.

Everything downstream of those two hooks is the production path:
:func:`build_real_backend`'s own unpacking of ``(lens_model, load_info)``, the
parameter freezing, the architecture audit, the **real**
:func:`resolve_audio_interface` probe against a processor that reproduces
``Gemma4Processor``'s audio contract, the **real**
:class:`GemmaPilotBackend` constructor, the bundle attribute access the cell
performs (``lens_model`` / ``backend`` / ``model_revision`` /
``processor_revision`` / ``audio_interface`` / ``audio_blocked_reason``), and
the real :func:`assert_audio_protocol` validation.

What this is and is not:

* **Is**: executable proof that the installed code and the notebook's real
  branch agree on every symbol, signature, return shape and attribute the L4
  run will touch.
* **Is not**: a run of the real checkpoint. Nothing here loads weights or
  touches the Hub, and nothing here says anything about Gemma.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jlens.mmpilot import real_backend as R
from jlens.mmpilot.audio import resolve_audio_interface
from jlens.mmpilot.mock import (
    MockAudioGemmaLike,
    MockAudioProcessor,
    MockWorld,
    mock_audio_config,
)
from jlens.mmpilot.preconvergence import check_preconvergence_call_contracts
from jlens.mmpilot.tri_modal import AudioProtocolMismatch, assert_audio_protocol

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = (
    REPO_ROOT
    / "notebooks"
    / "research_grade_l27_l31_preconvergence_study_colab.ipynb"
)


def _code_cells() -> list[str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    ]


def _cell_containing(needle: str) -> str:
    matches = [source for source in _code_cells() if needle in source]
    assert len(matches) == 1, f"expected exactly one cell containing {needle!r}"
    return matches[0]


# --------------------------------------------------------------- fake hooks


class FakeLensModel:
    """What ``load_gemma4`` returns: a wrapper whose ``_hf_model`` is the loaded
    HF model. The fake HF model reproduces the audio contract, so it computes."""

    def __init__(self, hf_model):
        self._hf_model = hf_model
        text_config = hf_model.config.get_text_config()
        self.n_layers = int(text_config.num_hidden_layers)
        self.d_model = int(text_config.hidden_size)

    def parameters(self):
        return self._hf_model.parameters()


@pytest.fixture(scope="module")
def fake_world():
    return MockWorld()


@pytest.fixture
def fake_hooks(monkeypatch, fake_world):
    """Only the two network hooks. Everything else is production code."""
    calls: dict = {}

    def fake_load_gemma4(
        repo_id, *, revision, dtype, device_map, allow_model_load, token
    ):
        calls["load_gemma4"] = {
            "repo_id": repo_id,
            "revision": revision,
            "dtype": dtype,
            "device_map": device_map,
            "allow_model_load": allow_model_load,
            "token_given": token is not None,
        }
        if not allow_model_load:
            raise RuntimeError("refusing to load")
        hf_model = MockAudioGemmaLike()
        return FakeLensModel(hf_model), {
            "model_repo_id": repo_id,
            "model_revision": f"resolved-{revision}",
            "tokenizer_revision": f"resolved-{revision}",
        }

    def fake_verify_architecture(
        model, *, expect_n_layers, expect_d_model, expect_vocab_size
    ):
        calls["verify_architecture"] = {
            "expect_n_layers": expect_n_layers,
            "expect_d_model": expect_d_model,
            "expect_vocab_size": expect_vocab_size,
        }
        return SimpleNamespace(
            to_dict=lambda: {
                "model_class": "MockAudioGemmaLike",
                "n_layers": model.n_layers,
                "d_model": model.d_model,
                "params_frozen": True,
            }
        )

    def fake_processor_factory(repo_id, *, revision, token):
        calls["processor"] = {"repo_id": repo_id, "revision": revision}
        return MockAudioProcessor()

    monkeypatch.setattr(R, "_loader", lambda: (fake_load_gemma4, fake_verify_architecture))
    monkeypatch.setattr(R, "_processor_factory", lambda: fake_processor_factory)
    return calls


@pytest.fixture
def expected_fingerprint(fake_world):
    """The fingerprint the fake interface really produces.

    Pinning the constant to the fake world is what lets ``assert_audio_protocol``
    run for real here instead of being skipped; the *live-ness* of the check is
    proved separately by ``test_a_wrong_expected_fingerprint_is_refused``.
    """
    resolved = resolve_audio_interface(
        MockAudioProcessor(),
        mock_audio_config(),
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="resolved-fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        processor_revision="resolved-fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
    )
    return resolved.protocol_fingerprint


# ---------------------------------------------- executing the real branch


def _namespace(tmp_path, expected_fingerprint, monkeypatch):
    """The notebook state the model cell runs against, and nothing more."""
    monkeypatch.setenv("HF_TOKEN", "fake-token-not-a-real-credential")

    class _PrepShim:
        """Stands in for the Stage-0 cache being complete. The completeness
        contract itself is tested in the notebook test; here it must simply be
        satisfied so the real branch is reached."""

        PREPARATION_VERSION = "test"

        @staticmethod
        def preparation_is_complete(_directory):
            return {"complete": True}

    namespace = {
        "__name__": "__notebook__",
        "os": __import__("os"),
        "torch": torch,
        "Path": Path,
        "prep": _PrepShim,
        "PREP_DIR": tmp_path / "prep",
        "PREPROCESSING_ONLY": False,
        "RUN_REAL_PRECONVERGENCE_STUDY": True,
        "MODEL_STAGE_ENABLED": True,
        "PREDOWNLOAD": {"passed": True},
        "MODEL_REPO_ID": "google/gemma-4-E4B-it",
        "MODEL_REVISION": "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        "EXPECT_N_LAYERS": 42,
        "EXPECT_D_MODEL": 2560,
        "EXPECT_VOCAB": 262144,
        "AUDIO_PROTOCOL_FINGERPRINT_EXPECTED": expected_fingerprint,
        "AUDIO_PROTOCOL_VERSION_EXPECTED": "jlens.mmpilot.native_spoken_audio.v1",
        "MOCK_MULTIMODAL_LAYER": 1,
        "CALIBRATION_MODEL": None,
        "MOCK_WORLD": None,
        "refresh_gates": lambda: {"MODEL_STAGE_ENABLED": True},
    }
    return namespace


@pytest.fixture
def executed(tmp_path, fake_hooks, expected_fingerprint, monkeypatch):
    namespace = _namespace(tmp_path, expected_fingerprint, monkeypatch)
    exec(compile(_cell_containing("build_real_backend("), "<model cell>", "exec"), namespace)  # noqa: S102
    return namespace, fake_hooks


# ------------------------------------------------------------------ tests


def test_the_real_branch_executes_and_produces_a_bundle(executed):
    namespace, _ = executed
    assert namespace["BUNDLE"] is not None
    assert namespace["MODEL"] is namespace["BUNDLE"].lens_model
    assert namespace["BACKEND"] is namespace["BUNDLE"].backend


def test_the_model_loader_was_called_with_the_pinned_revision(executed):
    _, calls = executed
    assert calls["load_gemma4"]["repo_id"] == "google/gemma-4-E4B-it"
    assert calls["load_gemma4"]["revision"] == (
        "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
    )
    assert calls["load_gemma4"]["allow_model_load"] is True
    assert calls["load_gemma4"]["token_given"] is True


def test_the_architecture_audit_ran_with_the_expected_shape(executed):
    _, calls = executed
    assert calls["verify_architecture"] == {
        "expect_n_layers": 42,
        "expect_d_model": 2560,
        "expect_vocab_size": 262144,
    }


def test_the_processor_was_built_at_the_resolved_revision(executed):
    _, calls = executed
    assert calls["processor"]["revision"].startswith("resolved-")


def test_every_bundle_attribute_the_cell_reads_exists(executed):
    namespace, _ = executed
    bundle = namespace["BUNDLE"]
    for attribute in (
        "lens_model",
        "backend",
        "model_revision",
        "processor_revision",
        "load_info",
        "audio_interface",
        "audio_blocked_reason",
        "architecture",
        "processor",
        "interface",
        "device",
    ):
        assert hasattr(bundle, attribute), attribute


def test_the_revisions_the_cell_records_come_from_the_bundle(executed):
    namespace, _ = executed
    assert namespace["MODEL_REVISION_USED"] == namespace["BUNDLE"].model_revision
    assert namespace["PROCESSOR_REVISION_USED"] == (
        namespace["BUNDLE"].processor_revision
    )
    assert namespace["TOKENIZER_REVISION_USED"].startswith("resolved-")


def test_the_audio_interface_resolved_and_was_validated(executed):
    namespace, _ = executed
    assert namespace["BUNDLE"].audio_interface is not None
    assert namespace["BUNDLE"].audio_blocked_reason == ""
    protocol = namespace["AUDIO_PROTOCOL"]
    assert protocol["matches_expected_fingerprint"] is True
    assert protocol["protocol_version"] == "jlens.mmpilot.native_spoken_audio.v1"


def test_the_media_loaders_were_built_with_a_retry_journal(executed):
    namespace, _ = executed
    assert set(namespace["MEDIA"]) >= {"load_image", "load_audio"}
    assert namespace["MEDIA_RETRY_JOURNAL"] is not None


def test_the_calibration_model_is_the_same_object_as_the_lens_model(executed):
    """One ~16 GB load serves both halves of the study."""
    namespace, _ = executed
    assert namespace["CALIBRATION_MODEL"] is namespace["MODEL"]


def test_a_blocked_audio_path_refuses_instead_of_degrading_to_two_modalities(
    tmp_path, monkeypatch, expected_fingerprint, fake_world
):
    def fake_load_gemma4(repo_id, *, revision, dtype, device_map, allow_model_load, token):
        return FakeLensModel(MockAudioGemmaLike()), {
            "model_repo_id": repo_id,
            "model_revision": f"resolved-{revision}",
        }

    def fake_verify_architecture(model, **_kwargs):
        return SimpleNamespace(to_dict=lambda: {})

    class NoAudioProcessor(MockAudioProcessor):
        """A processor with no feature extractor: the resolver's own refusal."""

        def __init__(self):
            super().__init__()
            self.feature_extractor = None

    monkeypatch.setattr(
        R, "_loader", lambda: (fake_load_gemma4, fake_verify_architecture)
    )
    monkeypatch.setattr(R, "_processor_factory", lambda: lambda *a, **k: NoAudioProcessor())
    namespace = _namespace(tmp_path, expected_fingerprint, monkeypatch)
    with pytest.raises(RuntimeError, match="native spoken-audio path did not resolve"):
        exec(  # noqa: S102
            compile(_cell_containing("build_real_backend("), "<model cell>", "exec"),
            namespace,
        )


def test_a_wrong_expected_fingerprint_is_refused(fake_world):
    """The protocol check is live, not decorative."""
    resolved = resolve_audio_interface(
        MockAudioProcessor(),
        mock_audio_config(),
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="r",
        processor_revision="r",
    )
    with pytest.raises(AudioProtocolMismatch):
        assert_audio_protocol(resolved, expected_fingerprint="sha256:not-this-one")


# ------------------------------- every real-only symbol resolves and binds


def test_every_real_branch_call_site_binds_against_the_installed_signature():
    assert check_preconvergence_call_contracts() == []


def _imported_symbols(source: str) -> set[tuple[str, str]]:
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("jlens"):
            for alias in node.names:
                found.add((node.module, alias.name))
    return found


def test_every_jlens_symbol_the_notebook_imports_actually_exists():
    """Resolves every ``from jlens... import name`` in every cell, real or not.

    An ``ImportError`` on the real branch is exactly the failure mode that has
    been reaching the L4, and it is invisible to a MOCK run.
    """
    import importlib

    missing = []
    for source in _code_cells():
        for module_name, symbol in sorted(_imported_symbols(source)):
            module = importlib.import_module(module_name)
            if not hasattr(module, symbol):
                missing.append(f"{module_name}.{symbol}")
    assert missing == []


def test_the_notebook_never_imports_a_symbol_that_does_not_exist():
    """The two names the brief names explicitly. Neither is importable."""
    joined = "\n".join(_code_cells())
    assert "load_real_bundle" not in joined
    assert "import preflight" not in joined
    from jlens.mmpilot import preflight as preflight_module
    from jlens.mmpilot import real_backend as real_backend_module

    assert not hasattr(preflight_module, "preflight")
    assert not hasattr(real_backend_module, "load_real_bundle")


def test_the_notebook_uses_the_two_entry_points_the_brief_requires():
    joined = "\n".join(_code_cells())
    assert "from jlens.mmpilot.real_backend import build_real_backend" in joined
    assert "from jlens.mmpilot.tri_modal import assert_audio_protocol" in joined
