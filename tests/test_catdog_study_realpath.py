# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Real-path execution of the frozen cat->dog animal-sound study (Stage 6A-6E).

Reuses ``_Harness`` from ``test_multimodal_followup_realpath.py`` -- the same
synthetic-backend infrastructure that runs the notebook's *actual* Stage
5A-5C cells -- and drives Stage 6's five sub-stages across five separate
sessions, exactly as a real Colab run would: each stage reads the previous
one's persisted, checksum-pinned artifact rather than sharing in-memory state.

``_Harness`` fakes any cell containing ``build_real_backend`` -- Stage 6C and
6E's model-load cells included -- exactly like every other stage's, so that a
chain test can run without a GPU. That means the *scientific* logic
downstream of the load (capability, development, freeze, confirmation) is
exercised for real here, but a *failing* preflight or the literal dtype
argument cannot be observed by driving that same substituted path. Those two
properties are instead checked directly against the generated notebook's own
source (below): that ``preflight_fp32_or_refuse`` is called strictly before
``build_real_backend`` in both load cells, and that both request
``dtype=torch.float32`` with no bf16 fallback anywhere in the cell. The
preflight's refusal arithmetic itself -- what makes it raise, and the exact
shortfall it names -- is exercised directly, with a synthetic ``free_bytes``,
in ``tests/test_fp32_preflight.py``.

**Nothing here is a scientific result.** The backend is synthetic and the
COCO annotation fixtures are invented.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# pytest's default (rootdir-based) import mode does not add this file's own
# directory to sys.path, so a bare `import test_multimodal_followup_realpath`
# fails even though that module sits right next to this one.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_multimodal_followup_realpath import _Harness  # noqa: E402


def _write_catdog_coco_fixture(root: Path, *, n_per_concept: int = 20) -> None:
    """A small ``annotations/`` tree with clean, single-species cat/dog photos.

    Every image passes :data:`DEFAULT_THRESHOLDS` by construction: exactly one
    animal, large and centered, named in every caption, no depiction words, no
    dominant person. This is deliberately generous -- the gate's *rejection*
    behaviour is already covered by ``tests/test_evidence_quality.py``; this
    fixture exists to give Stage 6's real cells a population to work with.
    """
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    categories = [
        {"id": 1, "name": "cat"}, {"id": 2, "name": "dog"},
        {"id": 3, "name": "bird"}, {"id": 4, "name": "cow"},
        {"id": 5, "name": "sheep"}, {"id": 6, "name": "horse"},
        {"id": 7, "name": "elephant"}, {"id": 8, "name": "bear"},
        {"id": 9, "name": "zebra"}, {"id": 10, "name": "giraffe"},
        {"id": 11, "name": "person"},
    ]
    name_to_id = {c["name"]: c["id"] for c in categories}
    images, annotations_list, captions = [], [], []
    for concept in ("cat", "dog"):
        for index in range(n_per_concept):
            image_id = (1 if concept == "cat" else 2) * 1_000_000 + index
            images.append({
                "id": image_id, "width": 640, "height": 480,
                "file_name": f"train2014_{image_id:012d}.jpg",
            })
            annotations_list.append({
                "image_id": image_id, "category_id": name_to_id[concept],
                "area": 200_000.0,
            })
            for phrase in (
                f"a {concept} sitting on the floor",
                f"a photograph of a {concept} indoors",
                f"a {concept} looking at the camera",
                f"a happy {concept} in a sunny room",
                f"a {concept} resting comfortably",
            ):
                captions.append({"image_id": image_id, "caption": phrase})
    (annotations / "instances_train2014.json").write_text(
        json.dumps({
            "images": images, "categories": categories,
            "annotations": annotations_list,
        }),
        encoding="utf-8",
    )
    (annotations / "captions_train2014.json").write_text(
        json.dumps({"annotations": captions}), encoding="utf-8",
    )
    (annotations / "instances_val2014.json").write_text(
        json.dumps({"images": [], "categories": categories, "annotations": []}),
        encoding="utf-8",
    )
    (annotations / "captions_val2014.json").write_text(
        json.dumps({"annotations": []}), encoding="utf-8",
    )


def _catdog_manifest_groups(n_per_concept: int = 20) -> list[dict]:
    """Manifest groups whose image ids match ``_write_catdog_coco_fixture``."""
    groups = []
    for concept in ("cat", "dog"):
        for index in range(n_per_concept):
            image_id = (1 if concept == "cat" else 2) * 1_000_000 + index
            groups.append({
                "group_id": f"g_catdog_{concept}_{index:04d}",
                "image_id": f"COCO_train2014_{image_id:012d}",
                "caption": f"a {concept} sitting on the floor",
                "image_path": f"/animals/{concept}/{index:04d}.jpg",
                "audio_path": f"/animals/{concept}/{index:04d}.wav",
            })
    return groups


def _fake_preflight(**_kwargs) -> dict:
    """A canned 'sufficient' fp32 preflight result -- no GPU touched."""
    return {
        "sufficient": True, "device_name": "fake-A100-80GB",
        "free_gib": 79.0, "required_gib": 49.0, "no_bf16_fallback": True,
    }


class _CatDogHarness(_Harness):
    """``_Harness`` with a COCO-shaped manifest and a faked fp32 preflight."""

    def __init__(self, tmp_path: Path, monkeypatch, *, n_per_concept: int = 20, **defects) -> None:
        super().__init__(tmp_path, monkeypatch, **defects)
        self.coco_root = tmp_path / "coco_fixture"
        _write_catdog_coco_fixture(self.coco_root, n_per_concept=n_per_concept)
        catdog_groups = _catdog_manifest_groups(n_per_concept)
        self.manifest.write_text(
            json.dumps({"groups": self.groups + catdog_groups}), encoding="utf-8"
        )
        # Patched once, here, rather than on every .run() call: a test that
        # wants a *different* preflight outcome (the two refusal tests below)
        # calls monkeypatch.setattr again after construction, and that later
        # call must be the one that wins for its .run().
        from jlens.mmpilot import fp32_preflight

        self.monkeypatch.setattr(
            fp32_preflight, "preflight_fp32_or_refuse", _fake_preflight
        )

    def run(self, **switches):
        switches.setdefault("IMAGE_MEDIA_ROOT", str(self.coco_root))
        return super().run(**switches)


# ---------------------------------------------------------------- the chain


def test_catdog_full_chain_6a_through_6e(tmp_path, monkeypatch) -> None:
    """The complete Stage 6A -> 6B -> 6C -> 6D -> 6E chain, five sessions."""
    harness = _CatDogHarness(tmp_path, monkeypatch)
    harness.script_model()

    common = {
        "CATDOG_N_DEV_CANDIDATES_PER_CONCEPT": 6,
        "CATDOG_N_CONFIRM_CANDIDATES_PER_CONCEPT": 10,
        "CATDOG_DEV_IMAGES_PER_DIRECTION": 3,
        "CATDOG_CONFIRM_IMAGES": 8,
    }

    # --- Stage 6A: evidence-quality index -----------------------------
    ns_a = harness.run(RUN_STAGE6A_EVIDENCE_QUALITY_INDEX=True, **common)
    index = ns_a["CATDOG_EVIDENCE_INDEX"]
    assert index is not None
    assert index["n_approved"]["cat"] >= 16
    assert index["n_approved"]["dog"] >= 16
    evidence_run_dir = str(ns_a["CATDOG_EVIDENCE_INDEX_RUN_DIR"])
    evidence_checksum = index["index_checksum"]

    # --- Stage 6B: disjoint population freeze --------------------------
    ns_b = harness.run(
        RUN_STAGE6B_POPULATION_FREEZE=True,
        CATDOG_EVIDENCE_INDEX_RUN_DIR=evidence_run_dir,
        EXPECTED_CATDOG_EVIDENCE_INDEX_CHECKSUM=evidence_checksum,
        **common,
    )
    freeze = ns_b["CATDOG_POPULATION_FREEZE"]
    assert freeze is not None
    assert freeze["disjoint"] is True
    assert freeze["frozen_before_model_load"] is True
    dev_ids = set(freeze["development_image_ids"])
    confirm_ids = set(freeze["confirmation_image_ids"])
    assert dev_ids.isdisjoint(confirm_ids)
    freeze_run_dir = str(ns_b["CATDOG_POPULATION_FREEZE_RUN_DIR"])
    freeze_digest = freeze["freeze_digest"]

    # --- Stage 6C: fp32 preflight, capability audit, development -------
    ns_c = harness.run(
        RUN_STAGE6C_CATDOG_DEVELOPMENT=True,
        CONFIRM_CATDOG_DEVELOPMENT_BUDGET=True,
        CATDOG_POPULATION_FREEZE_RUN_DIR=freeze_run_dir,
        EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST=freeze_digest,
        **common,
    )
    development = ns_c["CATDOG_DEVELOPMENT_REPORT"]
    assert development is not None
    assert ns_c["CATDOG_MODEL_LOADED_DTYPE"] == "float32"
    assert development["verdict"] == "NEW_PROPERTY_DEVELOPMENT_GO"
    assert "cat->dog" in development["passing_directions"]
    row = development["directions"][0]
    assert row["direction"] == "cat->dog"
    assert row["instrument_state"] == "EFFECT_GO"
    control = row["direct_answer_positive_control"]
    assert control["passed"] is True
    dev_run_dir = str(ns_c["CATDOG_DEVELOPMENT_RUN_DIR"])
    dev_checksum = development["report_checksum"]

    # --- Stage 6D: freeze the confirmation design -----------------------
    ns_d = harness.run(
        RUN_STAGE6D_CATDOG_FREEZE=True,
        CATDOG_DEVELOPMENT_RUN_DIR=dev_run_dir,
        EXPECTED_CATDOG_DEVELOPMENT_CHECKSUM=dev_checksum,
        **common,
    )
    design = ns_d["CATDOG_FROZEN_DESIGN"]
    assert design is not None
    assert design["direction"] == ["cat", "dog"]
    assert design["frozen_before_fresh_population_opened"] is True
    assert design["answer_aliases"]["dog"][0] == "bark"
    design_path = str(ns_d["CATDOG_FROZEN_DESIGN_PATH"])

    # --- Stage 6E: fresh confirmation -----------------------------------
    ns_e = harness.run(
        RUN_STAGE6E_CATDOG_CONFIRMATION=True,
        CONFIRM_CATDOG_CONFIRMATION_BUDGET=True,
        CATDOG_FROZEN_DESIGN_PATH=design_path,
        CATDOG_POPULATION_FREEZE_RUN_DIR=freeze_run_dir,
        EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST=freeze_digest,
        **common,
    )
    confirmation = ns_e["CATDOG_CONFIRMATION_REPORT"]
    assert confirmation is not None
    assert confirmation["verdict"] == "NEW_PROPERTY_CONFIRMATION_GO"
    assert confirmation["gate"]["population_fresh"] is True
    assert confirmation["gate"]["design_was_frozen_first"] is True
    for cell in confirmation["cells"]:
        assert cell["exact_success_rate"] >= 0.75
    # every recruited confirmation photograph came from Stage 6B's
    # confirmation pool -- disjoint from development by Stage 6B's own proof
    confirmed_image_ids = {
        str(row["image_id"])
        for rows in ns_e["_confirm_recruitment"]["groups"].values()
        for row in rows
    }
    assert confirmed_image_ids <= confirm_ids
    assert confirmed_image_ids.isdisjoint(dev_ids)


def test_catdog_stage6d_refuses_to_freeze_when_direct_answer_control_failed(
    tmp_path, monkeypatch
) -> None:
    """Development's frozen requirement is stronger than the generic verdict.

    ``new_property_development_verdict`` treats the direct-answer control as
    diagnostic everywhere else in the codebase -- it cannot gate a GO. This
    study's own design asks more: the control must have actually worked, not
    merely have been present, for Stage 6D to freeze a confirmation design at
    all. The scripted world here makes the exact exchange pass while the
    control itself does not.
    """
    harness = _CatDogHarness(tmp_path, monkeypatch, direct_answer_never_moves=True)
    harness.script_model()
    common = {
        "CATDOG_N_DEV_CANDIDATES_PER_CONCEPT": 6,
        "CATDOG_N_CONFIRM_CANDIDATES_PER_CONCEPT": 10,
        "CATDOG_DEV_IMAGES_PER_DIRECTION": 3,
        "CATDOG_CONFIRM_IMAGES": 8,
    }
    ns_a = harness.run(RUN_STAGE6A_EVIDENCE_QUALITY_INDEX=True, **common)
    index = ns_a["CATDOG_EVIDENCE_INDEX"]
    ns_b = harness.run(
        RUN_STAGE6B_POPULATION_FREEZE=True,
        CATDOG_EVIDENCE_INDEX_RUN_DIR=str(ns_a["CATDOG_EVIDENCE_INDEX_RUN_DIR"]),
        EXPECTED_CATDOG_EVIDENCE_INDEX_CHECKSUM=index["index_checksum"],
        **common,
    )
    freeze = ns_b["CATDOG_POPULATION_FREEZE"]
    ns_c = harness.run(
        RUN_STAGE6C_CATDOG_DEVELOPMENT=True,
        CONFIRM_CATDOG_DEVELOPMENT_BUDGET=True,
        CATDOG_POPULATION_FREEZE_RUN_DIR=str(ns_b["CATDOG_POPULATION_FREEZE_RUN_DIR"]),
        EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST=freeze["freeze_digest"],
        **common,
    )
    development = ns_c["CATDOG_DEVELOPMENT_REPORT"]
    row = development["directions"][0]
    assert row["passed"] is True  # the exchange itself still passes
    assert row["direct_answer_positive_control"]["passed"] is False

    with pytest.raises(Exception, match="direct-answer"):
        harness.run(
            RUN_STAGE6D_CATDOG_FREEZE=True,
            CATDOG_DEVELOPMENT_RUN_DIR=str(ns_c["CATDOG_DEVELOPMENT_RUN_DIR"]),
            EXPECTED_CATDOG_DEVELOPMENT_CHECKSUM=development["report_checksum"],
            **common,
        )


def test_catdog_stage6b_refuses_a_too_small_pool(tmp_path, monkeypatch) -> None:
    """Fewer clean-evidence photos than dev+confirm need: a clean refusal."""
    harness = _CatDogHarness(tmp_path, monkeypatch, n_per_concept=5)
    ns_a = harness.run(
        RUN_STAGE6A_EVIDENCE_QUALITY_INDEX=True,
        CATDOG_N_DEV_CANDIDATES_PER_CONCEPT=6,
        CATDOG_N_CONFIRM_CANDIDATES_PER_CONCEPT=6,
    )
    index = ns_a["CATDOG_EVIDENCE_INDEX"]
    with pytest.raises(Exception, match="clean-evidence"):
        harness.run(
            RUN_STAGE6B_POPULATION_FREEZE=True,
            CATDOG_EVIDENCE_INDEX_RUN_DIR=str(ns_a["CATDOG_EVIDENCE_INDEX_RUN_DIR"]),
            EXPECTED_CATDOG_EVIDENCE_INDEX_CHECKSUM=index["index_checksum"],
            CATDOG_N_DEV_CANDIDATES_PER_CONCEPT=6,
            CATDOG_N_CONFIRM_CANDIDATES_PER_CONCEPT=6,
        )


# --------------------------------------- fp32-before-load, at the source level
#
# The shared ``_Harness`` always fakes a cell containing "build_real_backend"
# (matching every other stage's model-load cell), so a test that wants to see
# a *failing* preflight or inspect the literal dtype argument cannot drive it
# through ``.run()`` -- that path is unconditionally short-circuited to
# "model load always succeeds" by design, the same design that lets every
# other real-path test in this suite run without a GPU. These two properties
# are instead verified against the generated notebook's actual source: that
# the preflight call appears strictly before the load call in both of Stage
# 6's model-load cells, and that both pass ``dtype=torch.float32`` rather than
# leaving the ``build_real_backend`` default (``torch.bfloat16``) in place.
# The refusal arithmetic itself -- what makes the preflight raise, and that it
# names the shortfall without ever falling back to bf16 -- is exercised
# directly, with a synthetic ``free_bytes``, in ``tests/test_fp32_preflight.py``.


def _catdog_model_load_cells() -> list[str]:
    from test_multimodal_followup_realpath import _code_cells

    return [
        source for source in _code_cells()
        if "build_real_backend" in source and "CATDOG" in source
    ]


def test_stage6_model_load_cells_exist_and_call_build_real_backend() -> None:
    cells = _catdog_model_load_cells()
    assert len(cells) == 2, "expected exactly one Stage 6C and one Stage 6E load cell"


@pytest.mark.parametrize("cell_index", [0, 1])
def test_stage6_preflight_runs_strictly_before_the_model_load(cell_index: int) -> None:
    source = _catdog_model_load_cells()[cell_index]
    preflight_at = source.index("preflight_fp32_or_refuse(")
    load_at = source.index("build_real_backend(")
    assert preflight_at < load_at, (
        "the fp32 preflight must be called before build_real_backend, so a "
        "refusal happens before any weight is loaded"
    )


@pytest.mark.parametrize("cell_index", [0, 1])
def test_stage6_requests_float32_and_never_names_bfloat16(cell_index: int) -> None:
    source = _catdog_model_load_cells()[cell_index]
    assert "dtype=torch.float32" in source
    assert "bfloat16" not in source
    assert "fallback" not in source.lower()
