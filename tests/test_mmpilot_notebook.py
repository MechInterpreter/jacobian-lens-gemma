# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and MOCK execution of the SpokenCOCO pilot notebook.

The execution test runs every code cell in one namespace with
``RUN_REAL_PILOT`` left at its committed default, so it checks two things at
once: that the notebook works end to end, and that opening it never starts the
real experiment.
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
    REPO_ROOT / "notebooks" / "multimodal_jspace_spokencoco_pilot_colab.ipynb"
)

RUNNER = Path(__file__).resolve().parent / "_mmpilot_notebook_runner.py"

#: The CPU-only audit (4-6) precedes authentication and model loading, so the
#: user can rank concepts on a free runtime and stop before any GPU is spent.
REQUIRED_SECTIONS = [
    "colab bootstrap",
    "configuration",
    "mount google drive",
    "install dependencies",
    "audit spokencoco",
    "audit synchronized evidence",
    "build the pilot subset",
    "authenticate",
    "audit the model architecture",
    "run the capability gate",
    "load and validate the frozen j-lens",
    "extract activations",
    "compute j-space codes",
    "run representational tests",
    "estimate concept directions",
    "run causal-transfer interventions",
    "generate the go/no-go report",
    "show resume status",
]


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells(payload):
    return [cell for cell in payload["cells"] if cell["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def _code_source(payload):
    return "\n".join("".join(cell["source"]) for cell in _code_cells(payload))


def test_notebook_is_valid_and_output_free(notebook):
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(_code_cells(notebook)):
        assert cell["outputs"] == [], f"code cell {index} has stored outputs"
        assert cell["execution_count"] is None, f"code cell {index} has an exec count"


def test_all_code_cells_parse_and_use_no_shell_magics(notebook):
    for index, cell in enumerate(_code_cells(notebook)):
        try:
            ast.parse("".join(cell["source"]))
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_sections_appear_in_the_required_order(notebook):
    """Bootstrap is section 0; the seventeen commissioned stages follow it."""
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(0, 18)), headings
    for (_, title), expected in zip(headings, REQUIRED_SECTIONS, strict=True):
        assert expected in title.lower(), (expected, title)


def test_bootstrap_cells_come_before_any_repository_import(notebook):
    """The regression this guards: section 1 imported ``jlens`` while the
    package had not been cloned or installed, so a fresh Colab runtime died
    with ModuleNotFoundError on the first executed cell."""
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    install = next(i for i, cell in enumerate(cells) if "pip" in cell and "install" in cell
                   and "-e" in cell)
    verify = next(i for i, cell in enumerate(cells) if "jlens.__file__" in cell)
    first_repo_import = next(
        i for i, cell in enumerate(cells) if re.search(r"^\s*from jlens", cell, re.MULTILINE)
    )
    assert install < first_repo_import, "the editable install must precede any jlens import"
    assert verify < first_repo_import, "`import jlens` must be verified first"
    # Cells before the verification may only use the standard library.
    for index in range(verify + 1):
        assert not re.search(r"^\s*(from|import) jlens\.", cells[index], re.MULTILINE) or (
            index == verify
        ), f"cell {index} imports from jlens before the bootstrap verified it"


def test_bootstrap_defines_only_primitives_before_cloning(notebook):
    """The first cell may assign string constants and print. Nothing else —
    an import here is what broke a fresh runtime."""
    constants = "".join(_code_cells(notebook)[0]["source"])
    tree = ast.parse(constants)
    assigned = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), (
                f"{node.targets[0].id} is not a primitive string constant"
            )
            assigned[node.targets[0].id] = node.value.value
        else:
            assert isinstance(node, ast.Expr) and isinstance(node.value, ast.Call), (
                f"unexpected statement in the constants cell: {ast.dump(node)[:80]}"
            )
    assert not [node for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert assigned["BRANCH"] == "experiment/spokencoco-jspace-pilot"
    assert assigned["REPO_DIR"] == "/content/jacobian-lens-gemma"
    assert assigned["REPO_URL"].endswith("jacobian-lens-gemma.git")


def test_clone_cell_is_idempotent_and_verifies_the_branch(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    clone_cell = next(cell for cell in cells if "clone" in cell)
    for expected in ("fetch", "checkout", "reset", "--hard", "rev-parse"):
        assert expected in clone_cell, expected
    assert '(REPO_PATH / ".git").is_dir()' in clone_cell
    assert "refusing to continue against the wrong code" in clone_cell


def test_bootstrap_does_not_need_drive(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    verify = next(i for i, cell in enumerate(cells) if "jlens.__file__" in cell)
    mount = next(i for i, cell in enumerate(cells) if "drive.mount" in cell)
    assert verify < mount
    for index in range(verify + 1):
        assert "drive" not in cells[index].lower()


def test_install_failure_is_reported_not_swallowed(notebook):
    source = _code_source(notebook)
    assert "pip install -e failed" in source
    assert "still not importable after installation" in source
    assert "is shadowing this checkout" in source


def test_the_real_pilot_is_off_by_default(notebook):
    source = _code_source(notebook)
    assert re.search(r"^RUN_REAL_PILOT = False$", source, re.MULTILINE)
    assert re.search(r"^RUN_MODEL_STAGES = False$", source, re.MULTILINE)
    # The only "= True" allowed is inside the printed instructions telling the
    # user what to set by hand.
    for switch in ("RUN_REAL_PILOT", "RUN_MODEL_STAGES"):
        assert not re.search(rf"^{switch} = True$", source, re.MULTILINE), switch


def test_every_expensive_action_is_gated_on_the_flag(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    for needle in ("drive.mount", "load_gemma4", "AutoProcessor.from_pretrained",
                   "getpass.getpass"):
        owners = [cell for cell in cells if needle in cell]
        assert owners, needle
        for cell in owners:
            assert "RUN_REAL_PILOT" in cell, needle


def test_model_loading_and_every_gpu_stage_is_gated_on_run_model_stages(notebook):
    """The repair's central guarantee: opening the notebook and running every
    cell must never load Gemma or score anything with it."""
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    for needle in (
        "load_gemma4",
        "AutoProcessor.from_pretrained",
        "MockPilotBackend()",
        "stage_capability",
        "stage_activations",
        "stage_codes",
        "stage_representational",
        "stage_directions",
        "stage_causal",
        "stage_report",
        "run_invariance_gate(",
        "getpass.getpass",
    ):
        owners = [cell for cell in cells if needle in cell]
        assert owners, needle
        for cell in owners:
            assert "RUN_MODEL_STAGES" in cell, needle


def test_the_cpu_audit_precedes_authentication_and_model_loading(notebook):
    """The user must be able to bootstrap, mount Drive, audit, rank and stop."""
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    index = {
        "expansion": next(i for i, c in enumerate(cells) if "build_expanded_manifest" in c),
        "audit": next(i for i, c in enumerate(cells) if "audit_groups(" in c),
        "discovery": next(
            i for i, c in enumerate(cells) if "discover_category_universe" in c
        ),
        "ranking": next(
            i
            for i, c in enumerate(cells)
            if "RANKED CONCEPT COVERAGE (complete, including rejections)" in c
        ),
        "subset": next(i for i, c in enumerate(cells) if "select_concepts" in c),
        "auth": next(i for i, c in enumerate(cells) if "getpass.getpass" in c),
        "model": next(i for i, c in enumerate(cells) if "load_gemma4" in c),
        "capability": next(i for i, c in enumerate(cells) if "stage_capability" in c),
    }
    order = [
        "discovery",
        "expansion",
        "audit",
        "ranking",
        "subset",
        "auth",
        "model",
        "capability",
    ]
    positions = [index[name] for name in order]
    assert positions == sorted(positions), index
    # Nothing before the model cell may import a backend or a model loader.
    for cell in cells[: index["model"]]:
        assert "load_gemma4" not in cell
        assert "AutoProcessor" not in cell


def test_the_audit_states_the_evidence_rule_and_never_transcribes(notebook):
    source = _source(notebook).lower()
    assert "coco object annotation" in source
    assert "whole word" in source or "whole-word" in source
    assert "cattle" in source, "the substring counter-example is worth stating"
    assert "no audio is transcribed" in source
    for forbidden in ("whisper", "speech_recognition", "transcribe(", "asr"):
        assert forbidden not in _code_source(notebook).lower(), forbidden


def test_the_top_two_feasible_concepts_are_selected_and_two_are_required(notebook):
    """Every discovered category is screened; the top two feasible are taken."""
    source = _code_source(notebook)
    assert re.search(r"^N_CONCEPTS_TO_SELECT = 4$", source, re.MULTILINE)
    assert re.search(r"^MIN_CONCEPTS_REQUIRED = 4$", source, re.MULTILINE)
    assert "max_concepts=2 if TINY_SMOKE else N_CONCEPTS_TO_SELECT" in source
    assert "n_concepts=2 if TINY_SMOKE else MIN_CONCEPTS_REQUIRED" in source
    assert re.search(r"^GROUPS_PER_CONCEPT = 6$", source, re.MULTILINE)


def test_the_six_concept_seed_list_is_gone(notebook):
    """The candidate universe is discovered, not the old handpicked six."""
    source = _code_source(notebook)
    assert "CONCEPT_LEXICON" not in source, "the seed lexicon must not be imported"
    assert "discover_category_universe" in source
    assert "config_from_specs" in source
    assert "UNIVERSE.lexicon()" in source
    # No literal candidate mapping survives anywhere in the notebook.
    for seed in ("horse", "pizza"):
        assert f'"{seed}":' not in source, seed
    prose = " ".join(_source(notebook).split()).lower()
    assert "instances_train*.json" in prose or "instances_*.json" in prose
    assert "discovered, not assumed" in prose or "discovered here, not listed" in prose


def test_the_scientific_minimums_are_stated_and_not_lowered(notebook):
    prose = " ".join(_source(notebook).split()).lower()
    assert "never lowered automatically" in prose
    assert "dataset no-go" in prose
    for minimum in ("6 distinct images", "4 source-training positives", "2 held-out positives"):
        assert minimum in prose, minimum


def test_the_derived_cache_is_fingerprinted_and_published_safely(notebook):
    source = _code_source(notebook)
    for expected in (
        "DERIVED_CACHE_ROOT",
        "cstf_spokencoco_derived/jlens_mmpilot_v1",
        "build_fingerprint",
        "DerivedCache(",
        "DERIVED_CACHE.describe()",
        "DERIVED_CACHE.state()",
        "DERIVED_CACHE.load()",
        "DERIVED_CACHE.publish(",
        "probe_media",
        "subset_to_relative",
        "subset_to_absolute",
        "STATE_INCOMPATIBLE",
    ):
        assert expected in source, expected
    for artifact in (
        "metadata.json",
        "concept_coverage.json",
        "concept_evidence_index.jsonl.gz",
        "rejected_evidence_counts.json",
        "selected_concepts.json",
        "pilot_subset.json",
        "split_provenance.json",
    ):
        assert artifact in source, artifact
    # An incompatible fingerprint stops the run rather than being reused.
    assert "Refusing to reuse them" in source
    prose = " ".join(_source(notebook).split())
    assert "_SUCCESS.json` is written last" in prose or "_SUCCESS.json is written last" in prose


def test_derived_evidence_artifacts_are_written_to_the_run_directory(notebook):
    source = _code_source(notebook)
    for artifact in (
        "synchronized_evidence_audit.json",
        "synchronized_evidence_manifest.json",
        "concept_ranking.json",
        "split_provenance.json",
    ):
        assert artifact in source, artifact
    for helper in (
        "persist_evidence_audit",
        "persist_synchronized_manifest",
        "persist_concept_ranking",
        "source_checksums",
    ):
        assert helper in source, helper


def test_representative_examples_are_printed_before_model_scoring(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    examples = next(i for i, c in enumerate(cells) if "format_examples" in c)
    capability = next(i for i, c in enumerate(cells) if "stage_capability" in c)
    assert examples < capability
    assert "n_positive=3" in cells[examples] and "n_rejected=3" in cells[examples]


def test_configured_drive_paths_match_the_existing_dataset(notebook):
    """Images live under coco/ and audio under SpokenCOCO/, so the two roots
    are configured separately — a single dataset root resolves neither."""
    config = "".join(_code_cells(notebook)[3]["source"])
    assignments = {
        node.targets[0].id: node.value.value
        for node in ast.parse(config).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert assignments["SPOKENCOCO_BASE_ROOT"] == (
        "/content/drive/MyDrive/datasets/cstf_spokencoco"
    )
    assert assignments["IMAGE_MEDIA_ROOT"] == (
        "/content/drive/MyDrive/datasets/cstf_spokencoco/coco"
    )
    assert assignments["AUDIO_MEDIA_ROOT"] == (
        "/content/drive/MyDrive/datasets/cstf_spokencoco/SpokenCOCO"
    )
    assert assignments["DOWNLOAD_CACHE"] == (
        "/content/drive/MyDrive/datasets/cstf_spokencoco_download_cache"
    )
    assert assignments["MANIFEST_PATH"] == (
        "/content/drive/MyDrive/datasets/spokencoco_manifest.json"
    )


def test_all_four_roots_are_offered_in_the_required_order(notebook):
    cell = next(
        "".join(c["source"]) for c in _code_cells(notebook) if "CANDIDATE_ROOTS" in "".join(c["source"])
    )
    order = re.search(r"CANDIDATE_ROOTS = \[(.*?)\]", cell, re.DOTALL).group(1)
    names = [name.strip().rstrip(",") for name in order.strip().splitlines() if name.strip()]
    assert names == [
        "IMAGE_MEDIA_ROOT",
        "AUDIO_MEDIA_ROOT",
        "DOWNLOAD_CACHE",
        "SPOKENCOCO_BASE_ROOT",
    ]


def test_media_root_audit_runs_before_normalization(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    audit_cell = next(i for i, cell in enumerate(cells) if "audit_media_roots" in cell)
    normalize_cell = next(i for i, cell in enumerate(cells) if "normalize_manifest" in cell)
    assert audit_cell <= normalize_cell
    cell = cells[audit_cell]
    assert cell.index("audit_media_roots") < cell.index("normalize_manifest")
    # Normalization must receive the per-role roots, not one shared list.
    assert "image_roots=IMAGE_ROOTS" in cell
    assert "audio_roots=AUDIO_ROOTS" in cell
    assert "expected_roots=EXPECTED_ROOTS" in cell


def test_notebook_never_downloads_or_rewrites_the_dataset(notebook):
    source = _code_source(notebook)
    for forbidden in ("shutil.rmtree", "os.remove", "download_dataset", "wget", "curl "):
        assert forbidden not in source, forbidden
    assert "never re-downloaded" in _source(notebook) or "not re-downloaded" in _source(notebook)


def test_lens_is_checksum_pinned_and_never_refitted(notebook):
    source = _code_source(notebook)
    assert "LENS_EXPECT_SHA256" in source
    assert "file_sha256" in source
    assert "validate_lens" in source
    assert "fit(" not in source
    assert "This pilot does not fit a lens." in source


def test_hf_token_is_read_with_getpass_and_not_printed(notebook):
    source = _code_source(notebook)
    assert "getpass.getpass" in source
    assert "print(_token)" not in source
    assert 'os.environ["HF_TOKEN"] = _token' in source


def test_invariance_gate_precedes_the_intervention_cell(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    gate = next(i for i, cell in enumerate(cells) if "run_invariance_gate" in cell)
    causal = next(i for i, cell in enumerate(cells) if "stage_causal" in cell)
    assert gate < causal


def test_interpretation_boundaries_are_stated_in_prose(notebook):
    prose = " ".join(_source(notebook).split()).lower()
    assert "not erasure" in prose
    assert "projection ablation" in prose
    assert "environmental audio" in prose
    assert "spoken_audio" in prose
    assert "never a scientific result" in prose or "not evidence about gemma" in prose


def test_no_forbidden_method_is_implemented(notebook):
    source = _code_source(notebook).lower()
    for forbidden in (
        "contrastive",
        "phrase_reconstructor",
        "reconstructor",
        "autoencoder",
        "adapter",
        "logisticregression",
    ):
        assert forbidden not in source, forbidden


def _clean_environment(tmp_path):
    """An environment where ``jlens`` is not importable: no repo on the path,
    ``PYTHONPATH`` cleared, and a working directory outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MMPILOT_REPO_DIR"] = str(REPO_ROOT)
    env["MMPILOT_SCRATCH"] = str(tmp_path / "scratch")
    env["MMPILOT_RUN_DIR"] = str(tmp_path / "run")
    return env


def test_the_test_environment_really_is_clean(tmp_path):
    """Guards the guard: if ``jlens`` were importable here anyway, the
    execution test below would pass even with the bootstrap broken."""
    probe = subprocess.run(
        [sys.executable, "-c", "import jlens"],
        cwd=tmp_path,
        env=_clean_environment(tmp_path),
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0
    assert "No module named 'jlens'" in probe.stderr


def _run_notebook(tmp_path, *overrides, env=None, run_dir=None, expect_ok=True):
    workdir = tmp_path / "elsewhere"
    workdir.mkdir(exist_ok=True)
    environment = {**_clean_environment(tmp_path), **(env or {})}
    if run_dir is not None:
        environment["MMPILOT_RUN_DIR"] = str(run_dir)
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(NOTEBOOK_PATH), *overrides],
        cwd=workdir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if expect_ok:
        assert result.returncode == 0, (
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-3000:]}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1]), result.stdout


def test_cpu_audit_runs_at_the_committed_defaults_without_loading_a_model(tmp_path):
    """The CPU-first path: bootstrap, audit, rank, split — and stop.

    Run at the committed defaults, so this is exactly what a user gets by
    opening the notebook and running every cell on a free runtime.
    """
    report, stdout = _run_notebook(tmp_path)

    assert report["ok"]
    assert Path(report["jlens_file"]).resolve().parent.parent == REPO_ROOT
    assert Path(report["cwd"]).resolve() == REPO_ROOT, "cell 0c must chdir into the repo"
    assert report["run_real_pilot"] is False
    assert report["run_model_stages"] is False
    assert report["model_is_none"], "no model may load at the committed defaults"
    assert report["overrides"] == {}

    # The audit produced a real verdict on real groups.
    assert report["n_valid_positives"] > 0
    assert len(report["selected_concepts"]) >= 2
    assert report["leakage_ok"]
    assert report["lexicon_hash"].startswith("sha256:")
    # The planted visual-only groups were rejected for the right reason.
    counts = report["rejection_counts"]
    assert counts["no_caption_lexical_evidence"] > 0
    assert counts["valid_synchronized_positive"] == report["n_valid_positives"]

    # Nothing downstream of the gate ran.
    assert report["recommendation"] is None
    assert report["resume_status"] is None
    assert report["invariance_passed"] is None
    assert "CPU AUDIT COMPLETE" in stdout
    assert "RUN_MODEL_STAGES  = True" in stdout, "the stop banner must say what to set"

    run_dir = Path(tmp_path / "run")
    for artifact in (
        "derived_manifest.json",
        "expanded_manifest.json",
        "synchronized_evidence_audit.json",
        "synchronized_evidence_manifest.json",
        "concept_ranking.json",
        "split_provenance.json",
    ):
        assert (run_dir / artifact).is_file(), artifact
    assert not (run_dir / "report.md").exists(), "no report without a model run"
    assert not (run_dir / "units").exists(), "no model units without a model run"

    audit = json.loads((run_dir / "synchronized_evidence_audit.json").read_text(encoding="utf-8"))
    assert audit["audio_transcribed"] is False
    assert audit["original_manifest_mutated"] is False
    assert audit["lexicon_hash"] == report["lexicon_hash"]
    assert audit["source_metadata_checksums"]
    rejected = [r for r in audit["records"] if not r["is_valid_synchronized_positive"]]
    assert rejected
    for record in rejected[:5]:
        assert record["rejection_reason"]
        assert record["normalized_caption"]
        assert record["split_provenance"]


def test_full_mock_execution_from_a_clean_environment(tmp_path):
    """The whole pipeline, with the model switch flipped the way a user flips it.

    This is the end-to-end check on the bootstrap too: cell 0 has to make the
    package importable, and everything after it has to run to a decision.
    """
    report, _ = _run_notebook(tmp_path, "RUN_MODEL_STAGES=True")

    assert report["ok"]
    assert Path(report["repo_path"]).resolve() == REPO_ROOT
    assert report["commit"]
    assert report["run_real_pilot"] is False, "the real pilot must stay off"
    assert report["run_model_stages"] is True
    assert report["model_is_none"], "the real Gemma must not load in MOCK"
    assert report["scientific_evidence"] is False
    assert report["recommendation"] in ("GO", "WEAK GO", "NO-GO")
    assert report["not_evaluated"] == [], "a complete run leaves nothing unevaluated"
    assert report["leakage_ok"]
    assert report["invariance_passed"]
    assert report["resume_status"] == "starting"
    assert report["n_interventions"] > 0

    run_dir = Path(tmp_path / "run")
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "derived_manifest.json").is_file()
    assert (run_dir / "expanded_manifest.json").is_file()
    assert (run_dir / "synchronized_evidence_audit.json").is_file()


def test_runtime_expansion_precedes_selection_and_model_scoring(notebook):
    source = _code_source(notebook)
    discovery = source.index("discover_metadata_sources")
    ranking = source.index("RANKED CONCEPT COVERAGE (complete, including rejections)")
    capability = source.index("stage_capability")
    assert discovery < ranking < capability
    for expected in (
        "max_files=40",
        "max_depth=3",
        "coco_object_annotation",
        "persist_expanded_manifest",
        "GROUPS_PER_CONCEPT = 6",
        "N_CONCEPTS_TO_SELECT = 4",
        "TINY_SMOKE = False",
    ):
        assert expected in source, expected


def test_the_cpu_path_discovers_its_concepts_rather_than_assuming_six(tmp_path):
    """The candidate universe comes off the local instances file. The mock
    declares more categories than its world has concepts, exactly as COCO does,
    so a run that merely rediscovered the seeds would show up here."""
    report, stdout = _run_notebook(tmp_path)

    discovered = set(report["discovered_categories"])
    assert {"hot dog", "orange", "person", "remote", "train"} <= discovered
    assert len(discovered) > len(report["selected_concepts"])
    # The excluded categories were considered and rejected, not never seen.
    assert set(report["excluded_categories"]) == {"orange", "person"}
    assert set(report["eligible_categories"]) == discovered - {"orange", "person"}
    # Every discovered eligible category was ranked, not just the survivors.
    assert set(report["ranked_concepts"]) == set(report["eligible_categories"])
    assert report["n_feasible"] >= 2
    assert report["universe_hash"].startswith("sha256:")
    assert report["lexical_hash"].startswith("sha256:")

    assert "CANDIDATE CONCEPT UNIVERSE (discovered, not assumed)" in stdout
    assert "RANKED CONCEPT COVERAGE (complete, including rejections)" in stdout
    # Rejections are printed with their reasons.
    assert "instances_train2014.json" in stdout
    assert "excluded categories and why" in stdout


def test_the_cpu_path_misses_then_hits_the_derived_cache(tmp_path):
    """The whole point of the cache: the second session does not rescan."""
    cache_env = {
        "MMPILOT_CACHE_ROOT": str(tmp_path / "drive_cache"),
        "MMPILOT_CACHE_STAGING": str(tmp_path / "staging"),
    }
    first, first_out = _run_notebook(
        tmp_path, env=cache_env, run_dir=tmp_path / "run_a"
    )
    assert first["cache_state"] == "miss"
    assert first["cache_hit"] is False
    assert first["cache_published"] is True
    assert "cache MISS" in first_out

    published = Path(first["cache_dir"])
    assert (published / "_SUCCESS.json").is_file()
    for artifact in (
        "metadata.json",
        "concept_coverage.json",
        "concept_evidence_index.jsonl.gz",
        "rejected_evidence_counts.json",
        "selected_concepts.json",
        "pilot_subset.json",
        "split_provenance.json",
    ):
        assert (published / artifact).is_file(), artifact

    second, second_out = _run_notebook(
        tmp_path, env=cache_env, run_dir=tmp_path / "run_b"
    )
    assert second["cache_state"] == "hit"
    assert second["cache_hit"] is True
    assert second["cache_fingerprint"] == first["cache_fingerprint"]
    assert second["cache_dir"] == first["cache_dir"]
    assert "cache HIT" in second_out
    assert "media probes" in second_out

    # The hit produced the same scientific answer without re-deriving it.
    assert second["selected_concepts"] == first["selected_concepts"]
    assert second["n_valid_positives"] == first["n_valid_positives"]
    assert second["rejection_counts"] == first["rejection_counts"]
    assert second["ranked_concepts"] == first["ranked_concepts"]
    # And it skipped the expensive stages: no re-derived manifest this time.
    assert not (tmp_path / "run_b" / "expanded_manifest.json").exists()
    assert (tmp_path / "run_a" / "expanded_manifest.json").is_file()


def test_a_published_cache_stores_no_absolute_paths_and_keeps_sources_intact(tmp_path):
    cache_env = {
        "MMPILOT_CACHE_ROOT": str(tmp_path / "drive_cache"),
        "MMPILOT_CACHE_STAGING": str(tmp_path / "staging"),
    }
    instances = (
        tmp_path / "scratch" / "data" / "coco" / "annotations" / "instances_train2014.json"
    )
    report, _ = _run_notebook(tmp_path, env=cache_env, run_dir=tmp_path / "run_a")
    before = instances.read_bytes()
    published = Path(report["cache_dir"])

    subset = json.loads((published / "pilot_subset.json").read_text(encoding="utf-8"))
    assert subset["paths_are_relative"] is True
    blob = (published / "pilot_subset.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in blob
    assert "MyDrive" not in blob
    rows = [row for rows in subset["splits"].values() for row in rows]
    assert rows and all("image_path" not in row for row in rows)
    assert all(row["image_relpath"] for row in rows)

    success = json.loads((published / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert success["source_metadata_mutated"] is False
    assert success["credentials_stored"] is False
    assert str(tmp_path) not in json.dumps(success["fingerprint"])

    # The source annotation file the universe was read from is byte-identical.
    _run_notebook(tmp_path, env=cache_env, run_dir=tmp_path / "run_b")
    assert instances.read_bytes() == before


def test_an_incompatible_published_cache_stops_the_run(tmp_path):
    """Compatibility is never inferred from a directory name."""
    cache_env = {
        "MMPILOT_CACHE_ROOT": str(tmp_path / "drive_cache"),
        "MMPILOT_CACHE_STAGING": str(tmp_path / "staging"),
    }
    report, _ = _run_notebook(tmp_path, env=cache_env, run_dir=tmp_path / "run_a")
    success_path = Path(report["cache_dir"]) / "_SUCCESS.json"
    stored = json.loads(success_path.read_text(encoding="utf-8"))
    stored["fingerprint_digest"] = "sha256:not-the-one-this-session-computes"
    success_path.write_text(json.dumps(stored), encoding="utf-8")

    failure, _ = _run_notebook(
        tmp_path, env=cache_env, run_dir=tmp_path / "run_c", expect_ok=False
    )
    assert failure["ok"] is False
    assert "Refusing to reuse them" in failure["traceback"]


def test_an_incomplete_published_cache_is_rebuilt_not_half_read(tmp_path):
    cache_env = {
        "MMPILOT_CACHE_ROOT": str(tmp_path / "drive_cache"),
        "MMPILOT_CACHE_STAGING": str(tmp_path / "staging"),
    }
    report, _ = _run_notebook(tmp_path, env=cache_env, run_dir=tmp_path / "run_a")
    (Path(report["cache_dir"]) / "_SUCCESS.json").unlink()

    rebuilt, stdout = _run_notebook(tmp_path, env=cache_env, run_dir=tmp_path / "run_d")
    assert rebuilt["cache_state"] == "incomplete"
    assert rebuilt["cache_hit"] is False
    assert "no valid success marker" in stdout
    assert rebuilt["selected_concepts"] == report["selected_concepts"]
    assert (Path(report["cache_dir"]) / "_SUCCESS.json").is_file()


def test_notebook_does_not_download_or_mutate_original_manifest(notebook):
    source = _code_source(notebook)
    assert "media_redownloaded\": False" in source
    assert "MANIFEST_PATH = \"/content/drive/MyDrive/datasets/spokencoco_manifest.json\"" in source
    assert "write_text" not in "\n".join(line for line in source.splitlines() if "MANIFEST_PATH" in line)
