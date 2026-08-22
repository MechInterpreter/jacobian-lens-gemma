from jlens.mmpilot.multimodal_workspace_repair import (
    causal_swap_report,
    freeze_repair_confirmation_design,
    multimodal_capability_report,
    select_loading_tomography,
)

MODALITIES = ("text", "image", "spoken_audio")


def _capability_rows(*, failures=()):
    rows = []
    failures = set(failures)
    for concept in ("bird", "giraffe"):
        for modality in MODALITIES:
            for kind in ("identity", "property"):
                for index in range(4):
                    rows.append(
                        {
                            "concept": concept,
                            "modality": modality,
                            "prompt_kind": kind,
                            "clean_correct": (
                                concept,
                                modality,
                                kind,
                                index,
                            )
                            not in failures,
                        }
                    )
    return rows


def test_capability_requires_property_endpoint_in_every_modality() -> None:
    report = multimodal_capability_report(
        _capability_rows(
            failures={("bird", "image", "property", index) for index in range(2)}
        ),
        concepts=("bird", "giraffe"),
    )
    assert report["eligible_concepts"] == ["giraffe"]
    assert report["teacher_forcing_used"] is False
    assert report["candidate_list_supplied"] is False


def _loading_rows(
    instrument: str, advantage: float, *, source_cosine: float | None = None
):
    del instrument
    rows = []
    for source, target in (("bird", "giraffe"), ("giraffe", "bird")):
        for modality in MODALITIES:
            for layer in (20, 21):
                for sample in range(4):
                    rows.append(
                        {
                            "sample_id": f"{source}:{modality}:{sample}",
                            "source": source,
                            "target": target,
                            "modality": modality,
                            "layer": layer,
                            "position": 3,
                            "position_class": "final_prompt_token",
                            "source_cosine": (
                                advantage + 0.2
                                if source_cosine is None
                                else source_cosine
                            ),
                            "target_cosine": 0.1,
                            "unrelated_cosines": {"toilet": 0.0},
                            "source_advantage": advantage,
                            "source_coordinate": 1.0,
                            "target_coordinate": 0.0,
                            "prompt_len": 4,
                            "evidence_span": None,
                            "causal_result_consulted": False,
                        }
                    )
    return rows


def test_tomography_chooses_bidirectionally_loaded_instrument() -> None:
    capability = multimodal_capability_report(
        _capability_rows(), concepts=("bird", "giraffe")
    )
    report = select_loading_tomography(
        {
            "weak": _loading_rows("weak", -0.01),
            "good": _loading_rows("good", 0.2),
        },
        instrument_layers={"weak": (20, 21), "good": (20, 21)},
        candidate_pairs=(("bird", "giraffe"),),
        capability=capability,
    )
    assert report["verdict"] == "MULTIMODAL_LOADING_TOMOGRAPHY_GO"
    assert report["selected_instrument"] == "good"
    assert report["selected_pair"] == ["bird", "giraffe"]
    assert report["selected_band"] == [20, 21]
    assert report["selection_depended_on_causal_outcome"] is False


def test_tomography_refuses_a_one_direction_only_loading_result() -> None:
    capability = multimodal_capability_report(
        _capability_rows(), concepts=("bird", "giraffe")
    )
    one_direction = [
        row
        for row in _loading_rows("one-way", 0.2)
        if row["source"] == "bird"
    ]
    report = select_loading_tomography(
        {"one-way": one_direction},
        instrument_layers={"one-way": (20, 21)},
        candidate_pairs=(("bird", "giraffe"),),
        capability=capability,
    )
    assert report["verdict"] == "MULTIMODAL_LOADING_TOMOGRAPHY_NO_GO"
    assert report["selected_instrument"] is None


def test_tomography_rejects_a_barely_loaded_layer_despite_positive_advantage() -> None:
    capability = multimodal_capability_report(
        _capability_rows(), concepts=("bird", "giraffe")
    )
    report = select_loading_tomography(
        {"coin-flip": _loading_rows("coin-flip", 0.2, source_cosine=0.002)},
        instrument_layers={"coin-flip": (20, 21)},
        candidate_pairs=(("bird", "giraffe"),),
        capability=capability,
        min_source_cosine=0.01,
    )
    assert report["verdict"] == "MULTIMODAL_LOADING_TOMOGRAPHY_NO_GO"
    assert report["ranking"][0]["source_cosine_eligible_layers"] == []


def test_tomography_uses_final_token_loading_but_keeps_paper_swap_rule() -> None:
    capability = multimodal_capability_report(
        _capability_rows(), concepts=("bird", "giraffe")
    )
    rows = _loading_rows("localized", 0.2)
    for row in list(rows):
        rows.extend(
            {
                **row,
                "sample_id": f"{row['sample_id']}:evidence:{index}",
                "position": index,
                "position_class": "evidence",
                "source_cosine": 0.03,
                "target_cosine": 0.04,
                "source_advantage": -0.01,
            }
            for index in range(10)
        )
    report = select_loading_tomography(
        {"localized": rows},
        instrument_layers={"localized": (20, 21)},
        candidate_pairs=(("bird", "giraffe"),),
        capability=capability,
        min_source_cosine=0.01,
    )
    assert report["verdict"] == "MULTIMODAL_LOADING_TOMOGRAPHY_GO"
    assert report["selected_band"] == [20, 21]
    assert report["primary_loading_position_class"] == "final_prompt_token"
    assert report["intervention_position_rule"] == "all_prompt_positions"
    assert set(report["position_rule_by_modality"].values()) == {
        "all_prompt_positions"
    }


def test_tomography_returns_no_go_without_capability_or_model_rows() -> None:
    capability = multimodal_capability_report(
        _capability_rows(
            failures={
                (concept, modality, "property", index)
                for concept in ("bird", "giraffe")
                for modality in MODALITIES
                for index in range(4)
            }
        ),
        concepts=("bird", "giraffe"),
    )
    report = select_loading_tomography(
        {},
        instrument_layers={},
        candidate_pairs=(("bird", "giraffe"),),
        capability=capability,
    )
    assert report["verdict"] == "MULTIMODAL_LOADING_TOMOGRAPHY_NO_GO"
    assert report["ranking"] == []
    assert report["causal_outcomes_opened"] is False


def _causal_rows(primary, zero, random, unrelated):
    rows = []
    for direction in ("bird->giraffe", "giraffe->bird"):
        for modality in MODALITIES:
            for index in range(len(primary)):
                rows.append(
                    {
                        "direction": direction,
                        "modality": modality,
                        "clean_correct": True,
                        "conditions": {
                            "exact_alpha1": {
                                "success": primary[index],
                                "integrity_passed": True,
                            },
                            "zero": {"success": zero[index]},
                            "random_alpha1": {"success": random[index]},
                            "unrelated_alpha1": {
                                "success": unrelated[index]
                            },
                        },
                    }
                )
    return rows


def test_causal_report_requires_exact_swap_to_beat_all_controls() -> None:
    positive = [True] * 16
    negative = [False] * 16
    report = causal_swap_report(
        _causal_rows(positive, negative, negative, negative),
        stage="confirmation",
        min_primary_successes=4,
    )
    assert report["verdict"] == "MULTIMODAL_SWAP_CONFIRMATION_GO"
    assert all(cell["passed"] for cell in report["cells"])

    null = causal_swap_report(
        _causal_rows(positive, positive, negative, negative),
        stage="confirmation",
        min_primary_successes=4,
    )
    assert null["verdict"] == "MULTIMODAL_SWAP_CONFIRMATION_NO_GO"
    assert all(not cell["controls_pass"] for cell in null["cells"])


def test_confirmation_design_freezes_the_loading_selected_alpha1_design() -> None:
    capability = multimodal_capability_report(
        _capability_rows(), concepts=("bird", "giraffe")
    )
    tomography = select_loading_tomography(
        {"pooled-r": _loading_rows("pooled-r", 0.2)},
        instrument_layers={"pooled-r": (20, 21)},
        candidate_pairs=(("bird", "giraffe"),),
        capability=capability,
    )
    development = causal_swap_report(
        _causal_rows([True] * 8, [False] * 8, [False] * 8, [False] * 8),
        stage="development",
        min_primary_successes=4,
    )
    development.update(
        {
            "instrument": tomography["selected_instrument"],
            "pair": tomography["selected_pair"],
            "layer_band": tomography["selected_band"],
            "position_rule_by_modality": tomography[
                "position_rule_by_modality"
            ],
        }
    )
    design = freeze_repair_confirmation_design(
        capability=capability,
        tomography=tomography,
        development=development,
        development_population_digest="sha256:development",
        confirmation_population_digest="sha256:confirmation",
        forbidden_development_image_ids=("image-a",),
        forbidden_prior_image_ids=("image-b",),
    )
    assert design["primary_alpha"] == 1.0
    assert design["instrument"] == "pooled-r"
    assert design["layer_band"] == [20, 21]
    assert design["teacher_forcing_used"] is False
    assert design["candidate_list_supplied"] is False
