from jlens.mmpilot.bird_cat_property_generalization import (
    CONDITIONS,
    MODALITIES,
    PROPERTY_PRIORITY,
    answer_matches,
    capability_report,
    confirmation_report,
    development_report,
    frozen_design,
    property_prompt,
)


def _capability_rows(n=6, failures=()):
    return [
        {
            "group_id": f"g{index}",
            "property": property_name,
            "modality": modality,
            "pass": (property_name, modality, index) not in failures,
        }
        for property_name in PROPERTY_PRIORITY
        for modality in MODALITIES
        for index in range(n)
    ]


def _trial_rows(property_name, n, *, exact_successes=None, control_successes=0):
    exact_successes = n if exact_successes is None else exact_successes
    rows = []
    for modality in MODALITIES:
        for condition in CONDITIONS:
            successes = exact_successes if condition == "exact" else control_successes
            for index in range(n):
                rows.append(
                    {
                        "group_id": f"g{index}",
                        "property": property_name,
                        "modality": modality,
                        "condition": condition,
                        "success": index < successes,
                        "integrity_pass": True,
                    }
                )
    return rows


def test_design_is_no_refit_exact_confirmed_method():
    design = frozen_design()
    assert design["source"] == "bird"
    assert design["target"] == "cat"
    assert design["layers"] == list(range(16, 41))
    assert design["alpha"] == 1.0
    assert design["lens_refitted"] is False
    assert "intervention outcome" in design["selection_uses"]


def test_prompts_are_modality_specific_and_answers_are_unrestricted_aliases():
    assert "Evidence: a bird" in property_prompt("taxonomic_class", "text", "a bird")
    assert "attached image" in property_prompt("taxonomic_class", "image")
    assert "spoken recording" in property_prompt("taxonomic_class", "spoken_audio")
    assert answer_matches(" Mammalia.", "taxonomic_class", "target")
    assert answer_matches("A mammalian.", "taxonomic_class", "target")
    assert answer_matches("an avian", "taxonomic_class", "source")
    assert answer_matches(" kitten<turn|>", "young_name", "target")
    assert answer_matches("A fledgling.", "young_name", "source")
    assert not answer_matches("chick", "young_name", "target")


def test_capability_selection_reads_only_clean_rows_and_uses_frozen_tiebreak():
    report = capability_report(_capability_rows())
    assert report["verdict"] == "BIRD_CAT_PROPERTY_CAPABILITY_GO"
    assert report["selected_property"] == "taxonomic_class"
    assert report["selection_read_intervention_outcomes"] is False


def test_capability_can_select_second_property_without_lowering_threshold():
    failures = {
        ("taxonomic_class", modality, index)
        for modality in MODALITIES
        for index in (0, 1)
    }
    report = capability_report(_capability_rows(failures=failures))
    assert report["selected_property"] == "young_name"


def test_development_requires_every_modality_and_every_control():
    capability = capability_report(_capability_rows())
    report = development_report(
        capability, _trial_rows("taxonomic_class", 6, exact_successes=4)
    )
    assert report["verdict"] == "BIRD_CAT_PROPERTY_DEVELOPMENT_GO"

    rows = _trial_rows("taxonomic_class", 6, exact_successes=4)
    for row in rows:
        if row["modality"] == "image" and row["condition"] == "random_1":
            row["success"] = True
    report = development_report(capability, rows)
    assert report["verdict"] == "BIRD_CAT_PROPERTY_DEVELOPMENT_NO_GO"


def test_fresh_confirmation_holm_corrects_all_fifteen_controls():
    report = confirmation_report(
        _trial_rows("taxonomic_class", 12, exact_successes=10),
        property_name="taxonomic_class",
    )
    assert report["verdict"] == "FRESH_MULTIMODAL_BIRD_CAT_PROPERTY_GENERALIZATION_GO"
    assert len(report["paired_comparisons"]) == 15
    assert all(row["holm_adjusted_p"] <= 0.05 for row in report["paired_comparisons"])
