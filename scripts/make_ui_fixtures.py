# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate the synthetic UI fixtures for the explorer.

    python scripts/make_ui_fixtures.py

Writes, deterministically (byte-identical on re-run):

- explorer/public/data/fixtures/causal_fixture.json
- explorer/public/data/fixtures/multimodal_fixture.json
- explorer/public/data/fixtures/assets/fixture_image.png
- explorer/public/data/fixtures/assets/fixture_audio.wav

These exist ONLY to exercise UI states before the real causal smoke run and
multimodal capture run have been executed. Every record carries
``synthetic_fixture`` status, the bundles declare it at provenance level, and
the frontend badges it loudly. The frontend automatically prefers measured
bundles (``explorer/public/data/measured/``) when they exist — dropping the
real ``explorer_causal_bundle.json`` / ``multimodal_explorer_bundle.json``
there makes these fixtures invisible. Numbers below are hand-written and
carry no experimental meaning.

The image and audio assets are generated here (a flat-color test card and a
0.4 s sine sweep) — original, tiny, and free of third-party rights.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens.explorer_export import SCHEMA_VERSION, write_bundle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(REPO_ROOT, "explorer", "public", "data", "fixtures")
SCHEMA_PATH = os.path.join(REPO_ROOT, "schemas", "explorer_bundle.schema.json")

MODEL_REPO = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
FIXTURE_NOTE = (
    "SYNTHETIC UI FIXTURE — hand-written values for interface development "
    "only. No forward pass produced these numbers. Replace by running the "
    "real notebook and merging its bundle (see docs/causal_smoke_run.md / "
    "docs/multimodal_capture.md)."
)

# The causal fixture attaches to this real text-demo example so the panel is
# reachable in the UI without fabricating a fake text example.
CANBERRA_EXAMPLE = "text:factual-canberra:26682621dffc6240"


def _hash16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _provenance(run_id: str, modalities: list[str]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "exporter_version": "fixtures-1.0.0",
        "source_run_ids": [run_id],
        "source_artifact_fingerprints": {},
        "model_repo_id": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "implementation_commit": None,
        "created_utc": "1970-01-01T00:00:00+00:00",
        "data_status": "synthetic_fixture",
        "modalities_present": modalities,
        "merged_bundles": [],
        "notes": FIXTURE_NOTE,
    }


def _score(token_id: int, token: str, logit: float, prob: float) -> dict:
    return {"token_id": token_id, "token": token, "logit": logit, "prob": prob}


def causal_fixture() -> dict:
    """Steering records at layers 35/38 for the Canberra example: output
    atom, top non-output atom, full cone, plus one norm-matched random
    control per targeted condition — multipliers -1/0/+1. All synthetic."""
    records = []
    target = {"token_id": 72710, "token": " Canberra"}

    def record(
        condition: str,
        *,
        layer: int,
        kind: str,
        multiplier: float,
        logit_delta: float,
        rank_after: int,
        prob_after: float,
        top1_after: str,
        control_family: str | None = None,
        matched: str | None = None,
        atom: dict | None = None,
    ) -> dict:
        base_logit = 21.4
        base_prob = 0.62
        return {
            "condition_id": condition,
            "example_id": CANBERRA_EXAMPLE,
            "layer": layer,
            "position": -1,
            "target_kind": kind,
            "atom_token_id": (atom or {}).get("token_id"),
            "atom_label": (atom or {}).get("token"),
            "multiplier": multiplier,
            "status": "synthetic_fixture",
            "norm_preserving": False,
            "delta_norm": 42.0,
            "activation_norm": 240.0,
            "delta_to_activation_ratio": 0.175,
            "target_token_id": target["token_id"],
            "target_token": target["token"],
            "target_logit_before": base_logit,
            "target_logit_after": base_logit + logit_delta,
            "target_logit_delta": logit_delta,
            "target_rank_before": 0,
            "target_rank_after": rank_after,
            "target_prob_before": base_prob,
            "target_prob_after": prob_after,
            "top1_before": _score(72710, " Canberra", base_logit, base_prob),
            "top1_after": _score(
                72710 if rank_after == 0 else 506,
                top1_after,
                base_logit + logit_delta if rank_after == 0 else 19.0,
                prob_after if rank_after == 0 else 0.31,
            ),
            "top10_before": [_score(72710, " Canberra", base_logit, base_prob)],
            "top10_after": [
                _score(
                    72710 if rank_after == 0 else 506,
                    top1_after,
                    base_logit + logit_delta if rank_after == 0 else 19.0,
                    prob_after if rank_after == 0 else 0.31,
                )
            ],
            "top10_overlap": 0.9 if abs(multiplier) < 1 else 0.7,
            "kl_divergence_after_vs_before": abs(logit_delta) * 0.05,
            "completion_before": " Canberra.",
            "completion_after": top1_after + ("." if rank_after == 0 else " capital."),
            "control_family": control_family,
            "matched_target_condition_id": matched,
            "provenance": {"fixture": True, "note": FIXTURE_NOTE},
        }

    kinds = [
        ("output_atom_contribution", {"token_id": 72710, "token": " Canberra"}),
        ("top_non_output_atom_contribution", {"token_id": 34618, "token": " capital"}),
        ("full_cone_reconstruction", None),
    ]
    for layer in (35, 38):
        for kind, atom in kinds:
            for multiplier, (delta, rank, prob, top1) in {
                -1.0: (-6.5, 3, 0.08, " the"),
                0.0: (0.0, 0, 0.62, " Canberra"),
                1.0: (4.2, 0, 0.87, " Canberra"),
            }.items():
                condition = f"fix_{kind[:12]}_{layer}_{multiplier:+.0f}"
                records.append(
                    record(
                        condition,
                        layer=layer,
                        kind=kind,
                        multiplier=multiplier,
                        logit_delta=delta,
                        rank_after=rank,
                        prob_after=prob,
                        top1_after=top1,
                        atom=atom,
                    )
                )
                if multiplier != 0.0:
                    records.append(
                        record(
                            f"fix_ctrl_{kind[:12]}_{layer}_{multiplier:+.0f}",
                            layer=layer,
                            kind="isotropic_random_direction",
                            multiplier=multiplier,
                            logit_delta=-0.1 * multiplier,
                            rank_after=0,
                            prob_after=0.61,
                            top1_after=" Canberra",
                            control_family="isotropic_random_direction",
                            matched=condition,
                        )
                    )

    return {
        "schema": "jlens.explorer.bundle.v1",
        "provenance": _provenance("fixture_causal_ui", ["text"]),
        "examples": [],
        "layer_records": [],
        "cones": [],
        "pursuit_traces": [],
        "trajectories": [],
        "causal_records": sorted(records, key=lambda r: r["condition_id"]),
        "notes": FIXTURE_NOTE,
    }


def multimodal_fixture() -> dict:
    """One image-conditioned and one audio-conditioned example at layer 38
    with a k=10 cone and pursuit trace, mirroring what the capture notebook
    will produce — every value synthetic."""
    image_prompt = "Describe the colors in this test card in one word. The dominant color is"
    audio_prompt = "Describe this sound in one word. The sound is a"
    image_hash = _hash16("fixture:" + image_prompt)
    audio_hash = _hash16("fixture:" + audio_prompt)
    image_id = f"image_text:fixture-test-card:{image_hash}"
    audio_id = f"audio_text:fixture-sine-sweep:{audio_hash}"

    def example(eid: str, slug: str, prompt: str, modality: str, top1: dict, seq_len: int, asset: dict) -> dict:
        return {
            "example_id": eid,
            "prompt_slug": slug,
            "prompt_hash": eid.rsplit(":", 1)[1],
            "category": "multimodal-fixture",
            "format": "plain",
            "modality": modality,
            "display_title": prompt,
            "prompt_text": prompt,
            "data_status": "synthetic_fixture",
            "seq_len": seq_len,
            "selected_positions": [-1],
            "model_output": {
                "-1": {
                    "input_token_id": 563,
                    "input_token": " is" if modality == "image_text" else " a",
                    "model_top1_id": top1["token_id"],
                    "model_top1_token": top1["token"],
                    "model_topk": [top1],
                }
            },
            "strength": None,
            "selection_reason": "synthetic fixture for UI development",
            "input": {
                "text": {
                    "token_ids": None,
                    "token_labels": None,
                    "positions_available": [-1],
                    "special_token_flags": None,
                    "prompt_text_is_pre_template": False,
                    "tokenization_available": False,
                },
                "image": asset if modality == "image_text" else None,
                "audio": asset if modality == "audio_text" else None,
            },
        }

    def cone(eid: str, atoms: list[tuple[int, str, float, bool]]) -> dict:
        total = sum(c for _, _, c, _ in atoms)
        return {
            "example_id": eid,
            "layer": 38,
            "position": -1,
            "requested_k": 10,
            "n_selected": len(atoms),
            "data_status": "synthetic_fixture",
            "selected_atoms": [
                {
                    "token_id": token_id,
                    "label": label,
                    "coefficient": coefficient,
                    "is_output_token": is_output,
                    "is_effective": coefficient > 0,
                    "coefficient_share": coefficient / total,
                    "nuisance": None,
                }
                for token_id, label, coefficient, is_output in atoms
            ],
            "coefficient_sum": total,
            "top_coefficient": max(c for _, _, c, _ in atoms),
            "concentration": {
                "herfindahl": 0.28,
                "top1_share": max(c for _, _, c, _ in atoms) / total,
                "n_nonzero": len(atoms),
            },
            "reconstruction": {
                "target_norm": 210.0,
                "residual_norm": 165.0,
                "relative_residual": 165.0 / 210.0,
                "explained_fraction": 1.0 - (165.0 / 210.0) ** 2,
            },
            "cone_signature_digest": None,
            "source_provenance": {"fixture": True, "note": FIXTURE_NOTE},
        }

    def trace(eid: str, atoms: list[tuple[int, str, float, bool]]) -> dict:
        norms = [210.0, 190.0, 180.0, 174.0, 171.0, 168.5, 167.0, 166.2, 165.7, 165.3, 165.0]
        steps = []
        for index, (token_id, label, _, _) in enumerate(atoms, start=1):
            relative = norms[index] / norms[0]
            steps.append(
                {
                    "step": index,
                    "added_token_id": token_id,
                    "added_label": label,
                    "support_after": [a[0] for a in atoms[:index]],
                    "residual_norm": norms[index],
                    "relative_residual": relative,
                    "explained_fraction": 1.0 - relative * relative,
                    "coefficients_after": None,
                    "final_coefficient_zero": False,
                }
            )
        return {
            "example_id": eid,
            "layer": 38,
            "position": -1,
            "requested_k": 10,
            "n_iterations": len(atoms),
            "stop_reason": "max_atoms",
            "data_status": "synthetic_fixture",
            "per_step_coefficients_available": False,
            "initial_residual_norm": norms[0],
            "target_norm": norms[0],
            "steps": steps,
        }

    def layer_record(eid: str, jlens_top: list[dict]) -> dict:
        return {
            "example_id": eid,
            "layer": 38,
            "position": -1,
            "source_site": "block_output",
            "data_status": "synthetic_fixture",
            "input_token_id": 563,
            "input_token": " is",
            "model_topk": None,
            "jlens_topk": jlens_top,
            "rank_of_model_top1": 2,
            "topk_overlap_with_model": 0.3,
            "eval_metadata": {"note": FIXTURE_NOTE},
            "target_activation_norm": 210.0,
            "residual_norm": 165.0,
            "relative_residual": 165.0 / 210.0,
            "explained_fraction": 1.0 - (165.0 / 210.0) ** 2,
        }

    image_atoms = [
        (3730, " blue", 30.0, True),
        (2260, " color", 14.0, False),
        (7033, " green", 11.0, False),
        (1063, " red", 9.0, False),
        (17120, " bright", 7.5, False),
        (30186, " stripe", 6.0, False),
        (496, " a", 5.0, False),
        (5112, " light", 4.0, False),
        (11689, " square", 3.0, False),
        (563, " is", 2.0, False),
    ]
    audio_atoms = [
        (10238, " tone", 26.0, True),
        (24923, " whistle", 13.0, False),
        (16050, " sound", 11.5, False),
        (30759, " pitch", 9.0, False),
        (2453, " high", 8.0, False),
        (43907, " sine", 6.5, False),
        (14066, " signal", 5.0, False),
        (496, " a", 4.0, False),
        (36752, " beep", 3.0, False),
        (563, " is", 2.5, False),
    ]

    image_example = example(
        image_id,
        "fixture-test-card",
        image_prompt,
        "image_text",
        _score(3730, " blue", 18.9, 0.44),
        295,
        {
            "asset_url": "data/fixtures/assets/fixture_image.png",
            "width": 96,
            "height": 64,
            "prompt_text": image_prompt,
            "modality_token_range": [1, 257],
            "processor_metadata": {"fixture": True},
        },
    )
    audio_example = example(
        audio_id,
        "fixture-sine-sweep",
        audio_prompt,
        "audio_text",
        _score(10238, " tone", 17.2, 0.38),
        140,
        {
            "asset_url": "data/fixtures/assets/fixture_audio.wav",
            "duration_seconds": 0.4,
            "sample_rate": 16000,
            "prompt_text": audio_prompt,
            "modality_token_range": [1, 121],
            "processor_metadata": {"fixture": True},
        },
    )

    return {
        "schema": "jlens.explorer.bundle.v1",
        "provenance": _provenance("fixture_multimodal_ui", ["audio_text", "image_text"]),
        "examples": [audio_example, image_example],
        "layer_records": [
            layer_record(audio_id, [_score(16050, " sound", 15.0, 0.2), _score(10238, " tone", 14.0, 0.12)]),
            layer_record(image_id, [_score(2260, " color", 16.0, 0.2), _score(3730, " blue", 15.0, 0.15)]),
        ],
        "cones": [cone(audio_id, audio_atoms), cone(image_id, image_atoms)],
        "pursuit_traces": [trace(audio_id, audio_atoms), trace(image_id, image_atoms)],
        "trajectories": [],
        "causal_records": [],
        "notes": FIXTURE_NOTE,
    }


# ------------------------------------------------------- asset generation


def make_png(path: str) -> None:
    """96x64 three-stripe test card, written as a minimal valid PNG with
    fixed zlib settings (deterministic bytes)."""
    width, height = 96, 64
    stripes = [(70, 110, 190), (235, 235, 235), (200, 120, 60)]
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type 0
        color = stripes[min(2, (3 * y) // height)]
        rows.extend(bytes(color) * width)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(png)


def make_wav(path: str) -> None:
    """0.4 s, 16 kHz mono sine sweep (440 -> 880 Hz), 16-bit PCM."""
    sample_rate = 16000
    n_samples = int(0.4 * sample_rate)
    samples = bytearray()
    for index in range(n_samples):
        t = index / sample_rate
        frequency = 440.0 + (880.0 - 440.0) * (index / n_samples)
        value = int(12000 * math.sin(2 * math.pi * frequency * t))
        samples += struct.pack("<h", value)
    data = bytes(samples)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
    )
    with open(path, "wb") as handle:
        handle.write(header + data)


def main() -> int:
    assets = os.path.join(FIXTURE_DIR, "assets")
    os.makedirs(assets, exist_ok=True)
    make_png(os.path.join(assets, "fixture_image.png"))
    make_wav(os.path.join(assets, "fixture_audio.wav"))
    for name, bundle in (
        ("causal_fixture.json", causal_fixture()),
        ("multimodal_fixture.json", multimodal_fixture()),
    ):
        out = os.path.join(FIXTURE_DIR, name)
        fingerprint = write_bundle(bundle, out, schema_path=SCHEMA_PATH)
        print(f"wrote {os.path.relpath(out, REPO_ROOT)} {fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
