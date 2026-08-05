# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The baseline artifact manifest: prior evidence, recorded and kept out.

This module **documents** the completed runs this project already owns. It does
not read them, and the calibration workflow must not read them either: a lens
fitted or validated against multimodal run results would be a lens whose
text-only claim is false.

Exactly one baseline artifact is readable by the new workflow — the frozen v2
text-only lens — and only as a *comparison* baseline, never as an input to
fitting or to a gate decision. Every other entry is `readable_by_calibration:
False` and exists so that a report can say what was already established without
the code being able to touch it.

Paths are Drive paths from a Colab runtime and are **not reachable from a
Windows checkout**. They are configurable, they are not asserted to exist, and
no MOCK test requires them. Checksums that this repository has actually recorded
are written down; every other checksum is marked
``REQUIRES_VERIFICATION_IN_COLAB`` rather than invented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from jlens.mmpilot.store import payload_checksum

__all__ = [
    "BASELINE_ARTIFACTS",
    "DRIVE_ROOT",
    "GEMMA4_REVISION",
    "BaselineArtifact",
    "baseline_manifest",
    "format_baseline_manifest",
    "readable_baselines",
]

#: Configurable Drive root. Never required to exist; nothing here is opened.
DRIVE_ROOT = "/content/drive/MyDrive/jacobian-lens-gemma"

GEMMA4_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"

UNVERIFIED = "REQUIRES_VERIFICATION_IN_COLAB"


@dataclass(frozen=True)
class BaselineArtifact:
    """One completed, immutable piece of prior evidence.

    Attributes:
        readable_by_calibration: Whether the new calibration may open it. Only
            the frozen v2 lens is True, and only as a comparison baseline.
        checksum: A checksum this repository has actually recorded, or
            :data:`UNVERIFIED`. Never a guess.
    """

    key: str
    role: str
    expected_path: str
    model_revision: str
    checksum: str
    scientific_status: str
    immutable: bool = True
    readable_by_calibration: bool = False
    may_be_used_to_fit: bool = False
    may_be_used_to_validate: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE_ARTIFACTS: tuple[BaselineArtifact, ...] = (
    BaselineArtifact(
        key="text_jlens_v2_lens",
        role=(
            "The frozen text-only J-lens the completed multimodal runs rest on. "
            "Fitted on 32 prompts; validated at layer 38 only. This study's "
            "pilot baseline, not a research-grade calibration."
        ),
        expected_path=(
            f"{DRIVE_ROOT}/runs/text_jlens_early_layer_recalibration_v2/"
            "artifacts/lens.validated.pt"
        ),
        model_revision=GEMMA4_REVISION,
        # Recorded in this repository's own memory of the v2 publication.
        checksum="sha256:4b17bf60...3641c2b (PREFIX ONLY — " + UNVERIFIED + ")",
        scientific_status=(
            "VALIDATED_TEXT_ONLY at physical layer 38. Layers 20, 26 and 32 were "
            "fitted and FAILED the v2 conjunction. Layer 32's failure was later "
            "diagnosed as a tie-block artifact rather than a clean negative."
        ),
        readable_by_calibration=True,
        may_be_used_to_fit=False,
        may_be_used_to_validate=False,
    ),
    BaselineArtifact(
        key="text_jlens_pilot_100",
        role=(
            "The 100-prompt WikiText pilot fit. Source of the single runtime "
            "measurement the entire compute budget is extrapolated from."
        ),
        expected_path=f"{DRIVE_ROOT}/runs/text_jlens_pilot",
        model_revision=GEMMA4_REVISION,
        checksum=UNVERIFIED,
        scientific_status=(
            "Completed. 100 prompts x 128 tokens x 7 layers in 9,665.4 s on an "
            "NVIDIA L4 (docs/pilot_report.md). Not a validated lens."
        ),
        readable_by_calibration=False,
    ),
    BaselineArtifact(
        key="mmpilot_four_concept",
        role="The confirmed four-concept SpokenCOCO text-image transfer pilot.",
        expected_path=f"{DRIVE_ROOT}/runs/mmpilot_pilot_20260803T160711",
        model_revision=GEMMA4_REVISION,
        checksum=UNVERIFIED,
        scientific_status=(
            "GO_CONFIRMED_AFTER_IMAGE_DEDUP at physical layer 38 "
            "(cat, giraffe, toilet, zebra)."
        ),
        readable_by_calibration=False,
    ),
    BaselineArtifact(
        key="mmpilot_robustness",
        role="The six-concept robustness study that established layer 38.",
        expected_path=f"{DRIVE_ROOT}/runs/mmrobust_robustness_20260804T154417",
        model_revision=GEMMA4_REVISION,
        checksum="sha256:61d0f0e7... (PREFIX ONLY — " + UNVERIFIED + ")",
        scientific_status=(
            "ROBUSTNESS_GO at physical layer 38. Bidirectional causal transfer, "
            "independent photographs as the statistical unit."
        ),
        readable_by_calibration=False,
    ),
    BaselineArtifact(
        key="mmlocalize_localization",
        role="The depth-localization run over physical layers 20, 26, 32, 38.",
        expected_path=(
            f"{DRIVE_ROOT}/runs/mmlocalize_localization_20260804T215140"
        ),
        model_revision=GEMMA4_REVISION,
        checksum=UNVERIFIED,
        scientific_status=(
            "Completed. Layer 38 reproduced; 20, 26 and 32 failed the tie-aware "
            "lens-validity gate. Layer 32 nevertheless showed striking cross-modal "
            "representational structure. Earlier causal transfer is UNRESOLVED, "
            "not disproven — an ineligible layer was never causally tested."
        ),
        readable_by_calibration=False,
    ),
    BaselineArtifact(
        key="audio_audit",
        role="The native spoken-audio feasibility audit.",
        expected_path=f"{DRIVE_ROOT}/runs/audioaudit_real_20260805T012451_c2d58028",
        model_revision=GEMMA4_REVISION,
        checksum=UNVERIFIED,
        scientific_status=(
            "AUDIO_READY. Native waveform reaches Gemma4AudioModel; placeholders "
            "and features agree; waveform changes logits; activation capture and "
            "intervention invariance pass."
        ),
        readable_by_calibration=False,
    ),
)


def readable_baselines() -> tuple[BaselineArtifact, ...]:
    """The baselines the calibration workflow is permitted to open."""
    return tuple(item for item in BASELINE_ARTIFACTS if item.readable_by_calibration)


def baseline_manifest(
    artifacts: Sequence[BaselineArtifact] = BASELINE_ARTIFACTS,
    *,
    drive_root: str = DRIVE_ROOT,
) -> dict:
    """The serializable manifest recorded beside the calibration run."""
    payload = {
        "schema": "jlens.calibration.baseline_manifest.v1",
        "drive_root": drive_root,
        "paths_are_configurable": True,
        "paths_verified_to_exist": False,
        "note": (
            "Drive paths are not reachable from a Windows checkout and are never "
            "opened by MOCK tests. Unknown checksums are marked "
            f"{UNVERIFIED} rather than invented."
        ),
        "calibration_may_read_multimodal_results": False,
        "calibration_may_fit_on_any_baseline": False,
        "calibration_may_validate_against_any_baseline": False,
        "artifacts": [item.to_dict() for item in artifacts],
        "readable_by_calibration": [
            item.key for item in artifacts if item.readable_by_calibration
        ],
    }
    payload["manifest_checksum"] = payload_checksum(payload)
    return payload


def format_baseline_manifest(manifest: Mapping | None = None) -> str:
    """The block printed in the notebook's baseline section."""
    manifest = manifest or baseline_manifest()
    lines = [
        "BASELINE ARTIFACT MANIFEST — prior evidence, documented and not consumed",
        f"  drive root   {manifest['drive_root']} (configurable; never opened here)",
        f"  checksum     {manifest['manifest_checksum']}",
        "",
    ]
    for item in manifest["artifacts"]:
        flag = "READABLE" if item["readable_by_calibration"] else "not readable"
        lines += [
            f"  [{flag}] {item['key']}",
            f"      role     {item['role']}",
            f"      path     {item['expected_path']}",
            f"      status   {item['scientific_status']}",
            f"      checksum {item['checksum']}",
            f"      immutable={item['immutable']}  may_fit={item['may_be_used_to_fit']}"
            f"  may_validate={item['may_be_used_to_validate']}",
            "",
        ]
    lines += [
        "  The new calibration reads no multimodal run result to fit or validate a",
        "  lens. The v2 lens is readable only as a comparison baseline.",
    ]
    return "\n".join(lines)
