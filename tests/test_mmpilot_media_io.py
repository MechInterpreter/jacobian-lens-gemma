# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The retrying Drive media loaders.

Every failure is injected at :func:`jlens.mmpilot.media_io._read_once`, which
exists for exactly this: a test that waited for a real Drive mount to hiccup
would never run, and one that mocked the whole loader would test the mock.
"""

import errno
import io
import json

import pytest

from jlens.mmpilot import media_io
from jlens.mmpilot.manifest import MediaIOError
from jlens.mmpilot.media_io import (
    DEFAULT_READ_ATTEMPTS,
    MEDIA_IO_VERSION,
    MediaDecodeError,
    RetryJournal,
    drive_media_loaders,
    media_checksum,
    read_media_bytes,
)


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Assert the backoff without waiting for it."""
    slept: list[float] = []
    monkeypatch.setattr(media_io, "_sleep", slept.append)
    return slept


def _flaky(failures, *, code=errno.EIO, payload=b"bytes"):
    """A reader that raises ``failures`` transient errors, then succeeds."""
    state = {"n": 0}

    def read(path):
        if state["n"] < failures:
            state["n"] += 1
            raise OSError(code, "Input/output error")
        return payload

    read.state = state
    return read


# ------------------------------------------------------------------ retries


def test_a_clean_read_never_sleeps(monkeypatch, tmp_path, no_sleeping):
    target = tmp_path / "x.bin"
    target.write_bytes(b"hello")
    assert read_media_bytes(target) == b"hello"
    assert no_sleeping == []


def test_a_transient_errno_is_retried_and_then_succeeds(monkeypatch, no_sleeping):
    monkeypatch.setattr(media_io, "_read_once", _flaky(3))
    journal = RetryJournal()
    assert read_media_bytes("anywhere", journal=journal) == b"bytes"
    assert journal.n_retries == 3
    assert len(no_sleeping) == 3


def test_the_backoff_doubles_and_is_capped(monkeypatch, no_sleeping):
    monkeypatch.setattr(media_io, "_read_once", _flaky(5))
    read_media_bytes("anywhere", attempts=8, initial_delay=0.1)
    assert no_sleeping == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.6])


def test_a_transient_errno_that_never_clears_refuses_with_advice(monkeypatch):
    monkeypatch.setattr(media_io, "_read_once", _flaky(99))
    with pytest.raises(MediaIOError, match="force_remount"):
        read_media_bytes("anywhere", attempts=3)


def test_the_attempt_budget_is_respected(monkeypatch, no_sleeping):
    reader = _flaky(99)
    monkeypatch.setattr(media_io, "_read_once", reader)
    with pytest.raises(MediaIOError):
        read_media_bytes("anywhere", attempts=4)
    assert reader.state["n"] == 4


def test_the_default_attempt_budget_is_more_generous_than_the_probes():
    from jlens.mmpilot.manifest import DEFAULT_PROBE_ATTEMPTS

    assert DEFAULT_READ_ATTEMPTS > DEFAULT_PROBE_ATTEMPTS


@pytest.mark.parametrize(
    "code", [errno.EIO, errno.ESTALE, errno.EAGAIN, errno.ETIMEDOUT]
)
def test_every_transient_errno_is_retried(monkeypatch, code, no_sleeping):
    monkeypatch.setattr(media_io, "_read_once", _flaky(1, code=code))
    assert read_media_bytes("anywhere") == b"bytes"


# ------------------------------------------------------------- non-retries


def test_a_missing_file_is_not_retried(monkeypatch, no_sleeping):
    def read(path):
        raise FileNotFoundError(errno.ENOENT, "nope")

    monkeypatch.setattr(media_io, "_read_once", read)
    with pytest.raises(MediaIOError, match="not present"):
        read_media_bytes("anywhere")
    assert no_sleeping == []


def test_a_permission_error_is_not_retried(monkeypatch, no_sleeping):
    def read(path):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(media_io, "_read_once", read)
    with pytest.raises(MediaIOError, match="permission denied"):
        read_media_bytes("anywhere")
    assert no_sleeping == []


def test_an_unrecognised_errno_is_not_retried(monkeypatch, no_sleeping):
    def read(path):
        raise OSError(errno.ENOSPC, "no space")

    monkeypatch.setattr(media_io, "_read_once", read)
    with pytest.raises(MediaIOError, match="not a known transient"):
        read_media_bytes("anywhere")
    assert no_sleeping == []


# ------------------------------------------------------------------ decode


def test_an_image_is_decoded_from_the_buffer(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    target = tmp_path / "x.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(target)
    loaded = media_io.load_image_bytes(target)
    assert loaded.size == (4, 4)
    assert loaded.mode == "RGB"


def test_a_corrupt_image_is_a_decode_error_not_a_transport_retry(
    monkeypatch, no_sleeping
):
    pytest.importorskip("PIL")
    monkeypatch.setattr(media_io, "_read_once", lambda path: b"not an image")
    with pytest.raises(MediaDecodeError, match="Retrying cannot repair"):
        media_io.load_image_bytes("anywhere")
    assert no_sleeping == []


def test_audio_is_decoded_and_downmixed(tmp_path):
    sf = pytest.importorskip("soundfile")
    numpy = pytest.importorskip("numpy")

    target = tmp_path / "x.wav"
    stereo = numpy.stack(
        [numpy.zeros(64, dtype="float32"), numpy.ones(64, dtype="float32")], axis=1
    )
    sf.write(target, stereo, 16000)
    waveform, rate = media_io.load_audio_bytes(target)
    assert rate == 16000
    assert waveform.ndim == 1
    # 16-bit PCM, so exact equality is the wrong assertion: the point is that
    # the two channels were averaged, not that WAV is lossless.
    assert waveform == pytest.approx(numpy.full(64, 0.5), abs=1e-4)


def test_a_corrupt_recording_is_a_decode_error(monkeypatch, no_sleeping):
    pytest.importorskip("soundfile")
    monkeypatch.setattr(media_io, "_read_once", lambda path: b"not a wav")
    with pytest.raises(MediaDecodeError, match="Retrying cannot repair"):
        media_io.load_audio_bytes("anywhere")


def test_the_image_read_itself_is_retried(monkeypatch, tmp_path, no_sleeping):
    pytest.importorskip("PIL")
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    monkeypatch.setattr(media_io, "_read_once", _flaky(2, payload=buffer.getvalue()))
    journal = RetryJournal()
    assert media_io.load_image_bytes("anywhere", journal=journal).size == (2, 2)
    assert journal.n_retries == 2


# ------------------------------------------------------------------ journal


def test_the_journal_summarises_by_errno_and_is_json_safe(monkeypatch, no_sleeping):
    monkeypatch.setattr(media_io, "_read_once", _flaky(2))
    journal = RetryJournal()
    read_media_bytes("a", journal=journal)
    monkeypatch.setattr(media_io, "_read_once", _flaky(1, code=errno.ESTALE))
    read_media_bytes("b", journal=journal)
    record = journal.to_dict()
    assert record["n_retries"] == 3
    assert record["n_paths_affected"] == 2
    # errno names, as this platform spells them: Windows reports ESTALE as
    # WSAESTALE, and hard-coding either spelling would fail on the other.
    assert set(record["retries_by_errno"]) == {
        errno.errorcode[errno.EIO],
        errno.errorcode[errno.ESTALE],
    }
    assert record["media_io_version"] == MEDIA_IO_VERSION
    json.dumps(record)


def test_an_empty_journal_reports_zero_rather_than_nothing():
    record = RetryJournal().to_dict()
    assert record["n_retries"] == 0
    assert record["entries"] == []


# ------------------------------------------------------------------ wiring


def test_both_loaders_come_from_one_call_site():
    loaders = drive_media_loaders()
    assert set(loaders) == {"load_image", "load_audio"}


def test_the_journal_is_shared_by_both_loaders(monkeypatch, tmp_path, no_sleeping):
    pytest.importorskip("PIL")
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    journal = RetryJournal()
    loaders = drive_media_loaders(journal=journal)
    monkeypatch.setattr(media_io, "_read_once", _flaky(1, payload=buffer.getvalue()))
    loaders["load_image"]("anywhere")
    assert journal.n_retries == 1


def test_the_checksum_is_over_the_bytes_that_were_decoded():
    assert media_checksum(b"abc").startswith("sha256:")
    assert media_checksum(b"abc") != media_checksum(b"abd")
