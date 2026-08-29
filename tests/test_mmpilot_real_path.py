# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Executable validation of the notebook's *real* branch, without weights.

Three L4 starts died on defects that only exist on the ``RUN_REAL_ROBUSTNESS``
branch — a renamed class, ``load_gemma4``'s return value unpacked as a
processor, a required ``interface`` argument never passed. MOCK execution
cannot catch those because the MOCK branch never runs that code, and
string-matching tests cannot catch them because the strings look fine.

These tests run the actual real-path code. :func:`build_real_backend` executes
with only its two network hooks replaced (the model loader and the processor
factory), so the unpacking, the freezing, the architecture audit call, the
interface resolution and the **real** :class:`GemmaPilotBackend` constructor
all run for real. The resulting backend is then pushed through one unit of
every downstream stage operation — prompt building, capability scoring,
activation capture, the invariance gate, pursuit, direction estimation, and an
intervention — over a deterministic fake model that actually computes.

What this is and is not:

* **Is**: executable proof that the installed code and the notebook's real
  branch agree on every signature, return shape, and field the run will touch.
* **Is not**: a run of the real checkpoint. Nothing here loads weights,
  touches the Hub, or proves anything about Gemma.
"""

import inspect
import json

import pytest
import torch

from jlens.mmpilot import real_backend as R
from jlens.mmpilot.backend import GemmaPilotBackend, run_invariance_gate
from jlens.mmpilot.capability import (
    build_prompt,
    build_question,
    candidate_token_ids,
    capability_record,
)
from jlens.mmpilot.causal import estimate_concept_direction, run_condition
from jlens.mmpilot.jspace import (
    capture_final_prompt_activations,
    code_map,
    jspace_code,
    validate_lens,
)
from jlens.mmpilot.mock import MockGemmaLike, MockWorld
from jlens.mmpilot.preflight import (
    PreflightError,
    check_call_contracts,
    real_path_preflight,
)

CONCEPTS = {
    "bus": ("bus", "buses"),
    "cat": ("cat", "cats"),
    "clock": ("clock", "clocks"),
    "dog": ("dog", "dogs"),
    "pizza": ("pizza", "pizzas"),
    "zebra": ("zebra", "zebras"),
}


class FakeTokenizer:
    """Hashes words to ids the way the mock world does, behind the interface
    the real backend actually uses (``__call__`` with ``add_special_tokens``)."""

    chat_template = None

    def __init__(self, backend_like):
        self._encode = backend_like

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": self._encode(text)}

    def decode(self, token_ids, skip_special_tokens=False):
        return f"token-{int(token_ids[0])}"


class FakeProcessor:
    """A processor with the real Gemma processor's *shape*: a ``__call__``
    taking text/images/return_tensors, a tokenizer, an image processor, no
    chat template — exactly what ``resolve_processor_interface`` inspects."""

    def __init__(self, world: MockWorld):
        from jlens.mmpilot.mock import MockPilotBackend

        self._mock = MockPilotBackend(world)
        self.tokenizer = FakeTokenizer(self._mock._tokenize)
        self.image_processor = object()
        self.chat_template = None

    def __call__(self, text=None, images=None, return_tensors=None):
        built = self._mock.build_inputs(
            prompt=text,
            modality="image" if images is not None else "text",
            image=images,
        )
        return dict(built.tensors)


class FakeLensModel:
    """What ``load_gemma4`` returns: a wrapper whose ``_hf_model`` is the
    loaded HF model. The fake HF model is the mock world's, so it computes."""

    def __init__(self, world: MockWorld):
        self._hf_model = MockGemmaLike(world)


@pytest.fixture(scope="module")
def world():
    return MockWorld(CONCEPTS)


@pytest.fixture(scope="module")
def fake_hooks(world):
    """The two network hooks, with call recording."""
    calls = {}

    def fake_load_gemma4(repo_id, *, revision, dtype, device_map, allow_model_load, token):
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
        return FakeLensModel(world), {
            "model_repo_id": repo_id,
            "model_revision": "resolved-" + revision,
        }

    def fake_verify_architecture(model, *, expect_n_layers, expect_d_model, expect_vocab_size):
        calls["verify_architecture"] = {
            "expect_n_layers": expect_n_layers,
            "expect_d_model": expect_d_model,
            "expect_vocab_size": expect_vocab_size,
        }
        text_config = model._hf_model.config.get_text_config()
        if text_config.hidden_size != expect_d_model:
            raise ValueError("d_model mismatch")

        class Report:
            def to_dict(self):
                return {
                    "n_layers": text_config.num_hidden_layers,
                    "d_model": text_config.hidden_size,
                    "vocab_size": text_config.vocab_size,
                    "layout_path": "model.language_model.layers",
                }

        return Report()

    def fake_processor_factory(repo_id, *, revision, token=None):
        calls["processor"] = {"repo_id": repo_id, "revision": revision}
        return FakeProcessor(world)

    return calls, fake_load_gemma4, fake_verify_architecture, fake_processor_factory


@pytest.fixture(scope="module")
def bundle(fake_hooks, monkeypatch_module):
    calls, load, verify, factory = fake_hooks
    monkeypatch_module.setattr(R, "_loader", lambda: (load, verify))
    monkeypatch_module.setattr(R, "_processor_factory", lambda: factory)
    result = R.build_real_backend(
        "google/gemma-4-E4B-it",
        revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        token="fake-token",
        device="cpu",
        allow_model_load=True,
        expect_n_layers=6,
        expect_d_model=world_d_model(),
        expect_vocab_size=0,
    )
    return calls, result


def world_d_model():
    from jlens.mmpilot.mock import MOCK_D_MODEL

    return MOCK_D_MODEL


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    yield patcher
    patcher.undo()


# ------------------------------------------------------- backend construction


def test_the_real_construction_sequence_executes_with_fakes(bundle):
    calls, result = bundle

    assert isinstance(result.backend, GemmaPilotBackend)
    assert calls["load_gemma4"]["allow_model_load"] is True
    assert calls["load_gemma4"]["token_given"] is True
    assert calls["load_gemma4"]["dtype"] == torch.bfloat16
    # The processor is loaded at the RESOLVED revision, not the requested one.
    assert calls["processor"]["revision"] == (
        "resolved-fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
    )
    assert result.model_revision == calls["processor"]["revision"]
    assert calls["verify_architecture"]["expect_d_model"] == world_d_model()


def test_the_loaded_model_is_frozen(bundle):
    _, result = bundle
    assert all(
        not parameter.requires_grad
        for parameter in result.backend.hf_model.parameters()
    )


def test_the_resolved_interface_is_by_inspection_not_assumption(bundle):
    _, result = bundle
    interface = result.interface

    assert interface["supports_image"] is True
    assert interface["supports_audio"] is False
    assert interface["has_chat_template"] is False
    assert "text" in interface["call_parameters"]
    assert result.backend.supports("image")
    assert not result.backend.supports("spoken_audio")


def test_refusing_the_model_load_propagates(fake_hooks, monkeypatch_module):
    _, load, verify, factory = fake_hooks
    monkeypatch_module.setattr(R, "_loader", lambda: (load, verify))
    monkeypatch_module.setattr(R, "_processor_factory", lambda: factory)

    with pytest.raises(RuntimeError, match="refusing to load"):
        R.build_real_backend(
            "google/gemma-4-E4B-it", revision="r", allow_model_load=False, device="cpu"
        )


def test_processor_only_backend_never_calls_the_model_loader(
    fake_hooks, monkeypatch_module
):
    calls, _load, _verify, factory = fake_hooks

    def forbidden_loader():
        raise AssertionError("the processor-only path reached the model loader")

    monkeypatch_module.setattr(R, "_loader", forbidden_loader)
    monkeypatch_module.setattr(R, "_processor_factory", lambda: factory)
    result = R.build_processor_backend(
        "google/gemma-4-E4B-it", revision="pinned-revision", token="fake-token"
    )
    assert result.processor_revision == "pinned-revision"
    assert result.backend.encode_candidate(" cat")
    assert calls["processor"]["revision"] == "pinned-revision"


# ----------------------------------------- one unit of every downstream stage


@pytest.fixture(scope="module")
def probe(bundle, world):
    _, result = bundle
    question = build_question(sorted(CONCEPTS))
    caption = "a photo of a cat sitting on a chair"
    inputs = result.backend.build_inputs(
        prompt=build_prompt(question, modality="text", caption=caption),
        modality="text",
    )
    return result.backend, inputs


def test_invariance_gate_runs_on_the_real_backend_class(probe):
    backend, inputs = probe
    gate = run_invariance_gate(backend, inputs, [2, 4])
    assert gate["passed"] is True


def test_one_capability_unit_scores_through_the_real_backend(probe):
    backend, inputs = probe
    candidates = candidate_token_ids(backend, sorted(CONCEPTS))
    record = capability_record(
        backend,
        inputs,
        sample_id="g1:text",
        concept="cat",
        group_id="g1",
        candidate_ids=candidates,
    )
    assert set(record["candidate_scores"]) == set(CONCEPTS)
    assert record["prompt_len"] == inputs.prompt_len


def test_one_image_unit_builds_through_the_processor_call_route(bundle, world):
    _, result = bundle
    image = world.evidence(
        concepts_present=["cat"], modality="image", nuisance_key="cat-img"
    )
    inputs = result.backend.build_inputs(
        prompt=build_prompt(build_question(sorted(CONCEPTS)), modality="image"),
        modality="image",
        image=image,
    )
    assert inputs.modality == "image"
    assert inputs.route["route"] == "processor_call"
    logits = result.backend.forward_logits(inputs.tensors)
    assert logits.shape[1] == inputs.prompt_len


@pytest.fixture(scope="module")
def one_of_every_stage(probe, bundle, world):
    """Activation -> code -> direction -> intervention, one unit each."""
    from jlens.mmpilot.mock import mock_lens
    from jlens.mmpilot.pipeline import build_dictionaries
    from jlens.pursuit import PursuitSettings

    backend, inputs = probe
    activations = capture_final_prompt_activations(backend, inputs, [4])
    lens = mock_lens(layers=(4,))
    dictionaries = build_dictionaries(
        lens, (4,), backend, device="cpu", dtype=torch.float32, build_chunk_rows=None
    )
    settings = PursuitSettings(k=8, normalize_atoms=True, refine_steps=1,
                               tol_relative_residual=0.0, correlation_chunk_size=None)
    code = jspace_code(
        activations[4],
        dictionaries[4],
        settings,
        activation_checksum="sha256:x",
        lens_checksum="sha256:mock-identity-lens",
    )
    return backend, inputs, activations, dictionaries, code


def test_one_activation_and_code_unit(one_of_every_stage):
    _, inputs, activations, _, code = one_of_every_stage
    assert activations[4].shape[-1] == world_d_model()
    assert code["n_active"] > 0
    assert code["convergence_status"]


def test_one_direction_and_intervention_unit(one_of_every_stage):
    backend, inputs, activations, dictionaries, code = one_of_every_stage
    from jlens.mmpilot.capability import candidate_token_ids, score_candidate_sequences

    positive_code = code_map(code)
    negative_atom = next(a for a in range(3) if a not in positive_code)
    direction = estimate_concept_direction(
        [positive_code],
        [{negative_atom: 1.0}],
        dictionaries[4],
        concept="cat",
        source_modality="text",
        layer=4,
        positive_ids=["g1:text"],
        negative_ids=["g2:text"],
        lens_checksum="sha256:mock-identity-lens",
        top_k=4,
    )
    candidates = candidate_token_ids(backend, sorted(CONCEPTS))
    clean = score_candidate_sequences(backend, inputs, candidates)
    record = run_condition(
        backend,
        inputs,
        layer=4,
        unit_direction=torch.tensor(direction["unit_direction"]),
        alpha=0.25,
        reference_norm=float(activations[4].norm()),
        sign=-1,
        candidate_ids=candidates,
        target_concept="cat",
        clean_scores=clean,
    )
    assert record["resolved_position"] == inputs.final_prompt_position
    assert "signed_target_effect" in record
    assert record["activation_norm_ratio"] > 0


def test_no_network_or_weight_load_happened(bundle):
    """The only loader and processor factory the module knows were the fakes."""
    calls, _ = bundle
    assert set(calls) == {"load_gemma4", "verify_architecture", "processor"}


# ------------------------------------------------------------- the lens path


@pytest.fixture()
def published_lens(tmp_path):
    from jlens.metadata import file_sha256
    from jlens.mmpilot.mock import mock_lens

    lens = mock_lens(layers=(38,), d_model=16)
    path = tmp_path / "artifacts" / "lens.validated.pt"
    path.parent.mkdir(parents=True)
    lens.save(str(path))
    checksum = file_sha256(str(path))
    manifest = {
        "status": "validated_text_only",
        "lens_path": str(path),
        "lens_checksum": checksum,
        "model_revision": "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        "native_readout_layers_passing": [38],
        "native_validation_path": "native_readout_validation.json",
    }
    (path.parent / "validated_lens_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return path, checksum, manifest


def test_a_published_lens_loads_and_validates_end_to_end(published_lens):
    path, checksum, _ = published_lens
    validated = R.load_validated_lens(
        path,
        expect_checksum=checksum,
        layers=(38,),
        model_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
    )
    assert 38 in validated.lens.jacobians
    # And the notebook's validate_lens call binds and passes on it.
    report = validate_lens(
        validated.lens,
        lens_path=validated.path,
        lens_checksum=validated.checksum,
        layers=(38,),
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        expect_model_repo_id="google/gemma-4-E4B-it",
        expect_model_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        expect_d_model=16,
        expect_checksum=checksum,
    )
    assert report["frozen"] is True


def test_a_checksum_mismatch_refuses(published_lens):
    path, _, _ = published_lens
    with pytest.raises(R.LensManifestError, match="pinned"):
        R.load_validated_lens(
            path, expect_checksum="sha256:not-it", layers=(38,), model_revision="r"
        )


def test_a_manifest_revision_mismatch_refuses(published_lens):
    path, checksum, _ = published_lens
    with pytest.raises(R.LensManifestError, match="model revision"):
        R.load_validated_lens(
            path, expect_checksum=checksum, layers=(38,), model_revision="other"
        )


def test_a_layer_never_validated_refuses(published_lens):
    path, checksum, _ = published_lens
    with pytest.raises(R.LensManifestError, match="natively"):
        R.load_validated_lens(
            path,
            expect_checksum=checksum,
            layers=(38, 21),
            model_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        )


def test_a_missing_manifest_refuses(tmp_path):
    from jlens.metadata import file_sha256
    from jlens.mmpilot.mock import mock_lens

    path = tmp_path / "lens.validated.pt"
    mock_lens(layers=(38,), d_model=16).save(str(path))
    with pytest.raises(R.LensManifestError, match="manifest"):
        R.load_validated_lens(
            path,
            expect_checksum=file_sha256(str(path)),
            layers=(38,),
            model_revision="r",
        )


# --------------------------------------------------------------- preflight


def _preflight_inputs(published_lens, **overrides):
    path, checksum, _ = published_lens
    selected = ["zebra", "cat", "toilet", "giraffe", "bird", "clock"]
    subset = {
        "splits": {
            split: [
                {"image_id": f"{split}-{i}", "group_id": f"g-{split}-{i}"}
                for i in range(4)
            ]
            for split in ("train", "test")
        }
    }
    payload = dict(
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        lens_path=path,
        lens_expect_checksum=checksum,
        layers=(38,),
        expect_d_model=16,
        selected_concepts=selected,
        focal_concepts=selected[:3],
        unrelated_controls={"zebra": "giraffe", "cat": "bird", "toilet": "clock"},
        subset=subset,
        split_provenance_checksum="sha256:abc",
        resolve_hub_revision=False,
    )
    payload.update(overrides)
    return payload


def test_the_preflight_passes_on_a_complete_real_path(published_lens):
    report = real_path_preflight(**_preflight_inputs(published_lens))
    assert report["passed"] is True
    names = {check["check"] for check in report["checks"]}
    assert {
        "lens_checksum_matches_pin",
        "requested_revision_matches_lens_manifest",
        "layers_natively_validated",
        "lens_contains_requested_layers",
        "lens_d_model",
        "call_signatures_bind",
        "six_concepts_selected",
        "three_focal_concepts",
        "unrelated_controls_external",
        "subset_train_image_unique",
        "subset_test_image_unique",
        "split_provenance_checksum_present",
    } <= names


def test_the_preflight_lists_every_failure_not_just_the_first(published_lens):
    inputs = _preflight_inputs(
        published_lens,
        lens_expect_checksum="sha256:wrong",
        selected_concepts=["a", "b"],
        split_provenance_checksum="",
    )
    with pytest.raises(PreflightError) as error:
        real_path_preflight(**inputs)
    message = str(error.value)
    assert "lens_checksum_matches_pin" in message
    assert "six_concepts_selected" in message
    assert "split_provenance_checksum_present" in message
    assert "must not start" in message


def test_the_preflight_refuses_a_revision_the_lens_was_not_validated_for(
    published_lens,
):
    inputs = _preflight_inputs(published_lens, model_revision="deadbeef")
    with pytest.raises(PreflightError, match="requested_revision_matches_lens_manifest"):
        real_path_preflight(**inputs)


def test_the_preflight_resolves_the_hub_revision_through_its_hook(
    published_lens, monkeypatch
):
    import jlens.mmpilot.preflight as P

    seen = {}

    def fake_info(repo_id, revision, token):
        seen["args"] = (repo_id, revision, token is not None)

        class Info:
            sha = "abc123"

        return Info()

    monkeypatch.setattr(P, "_hub_model_info", fake_info)
    report = real_path_preflight(
        **_preflight_inputs(published_lens, resolve_hub_revision=True, token="t")
    )
    assert report["passed"]
    assert seen["args"] == (
        "google/gemma-4-E4B-it",
        "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        True,
    )


def test_a_hub_resolution_failure_blocks_the_download(published_lens, monkeypatch):
    import jlens.mmpilot.preflight as P

    def failing_info(repo_id, revision, token):
        raise OSError("401 gated repo")

    monkeypatch.setattr(P, "_hub_model_info", failing_info)
    with pytest.raises(PreflightError, match="hub_revision_resolves"):
        real_path_preflight(
            **_preflight_inputs(published_lens, resolve_hub_revision=True)
        )


def test_a_subset_with_a_repeated_image_fails_preflight(published_lens):
    inputs = _preflight_inputs(published_lens)
    inputs["subset"]["splits"]["test"].append(
        {"image_id": "test-0", "group_id": "g-dup"}
    )
    with pytest.raises(PreflightError, match="subset_test_image_unique"):
        real_path_preflight(**inputs)


# ------------------------------------------------------- signature contracts


def test_every_notebook_call_binds_against_the_installed_signatures():
    assert check_call_contracts() == []


def test_the_backend_constructor_contract_is_pinned():
    """Fails when GemmaPilotBackend gains or loses a required argument."""
    parameters = inspect.signature(GemmaPilotBackend).parameters
    required = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    assert required == ["hf_model", "processor", "interface"]
    assert "device" in parameters


def test_the_loader_contract_is_pinned():
    from jlens.gemma4 import load_gemma4

    parameters = inspect.signature(load_gemma4).parameters
    for name in ("repo_id", "revision", "token", "allow_model_load", "dtype", "device_map"):
        assert name in parameters, name
    assert parameters["allow_model_load"].default is False, (
        "the 16 GB download must stay opt-in"
    )


def test_the_interface_resolver_returns_the_fields_the_backend_reads(bundle):
    _, result = bundle
    for field in (
        "supports_image",
        "supports_audio",
        "audio_kwarg",
        "call_parameters",
        "has_chat_template",
        "image_token_id",
        "audio_token_id",
        "processor_class",
        "components",
    ):
        assert field in result.interface, field


