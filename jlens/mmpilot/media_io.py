# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Reading media off a Colab Drive mount without losing a multi-hour run.

:mod:`jlens.mmpilot.manifest` already retries the *probe* — ``os.stat`` on a
path that exists and answers a moment later. It does not retry the **read**, and
the read is where a long run dies: ``Image.open(path)`` and ``soundfile.read``
both hold the file handle open across many small requests to the FUSE layer, so
a single ``OSError: [Errno 5] Input/output error`` two hours into a study raises
straight out of the loader and takes the session with it. The units already
written survive, but the runtime does not, and on a shared L4 that is an hour of
queueing to get back.

Two things fix it, and they are the same two things:

**Read the bytes first, decode second.** The whole file is pulled into a
:class:`io.BytesIO` with bounded retries, and the decoder is handed a buffer.
The flaky part is then one short operation that can be attempted again as a
whole, rather than a decode interleaved with hundreds of resumable-in-principle
reads that no library exposes a retry hook for.

**Retry only what "retry" can fix.** The errno classification is imported from
:mod:`jlens.mmpilot.manifest` rather than restated, so a transient errno means
the same thing to a loader as it does to a probe. A missing file is a missing
file on the first attempt; a permission error is never waited on.

Nothing here changes what is measured. The bytes that reach the processor are
the bytes on disk, and :func:`load_image_bytes` records their sha256 so a caller
that already checksums its media keeps agreeing with itself.
"""

from __future__ import annotations

import errno as errno_module
import hashlib
import io
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from jlens.mmpilot.manifest import (
    ABSENT_ERRNOS,
    MAX_PROBE_DELAY,
    TRANSIENT_ERRNOS,
    MediaIOError,
    _sleep,
)

#: Bound into the run fingerprint. A change in the retry policy does not change
#: any measurement, but it does change what a run's exclusion log means, so the
#: version is recorded rather than left implicit.
MEDIA_IO_VERSION = "jlens.mmpilot.media_io.v1"

#: Total attempts including the first. Six rather than the probe's four: a byte
#: read of a whole JPEG or WAV is a much longer window for the mount to hiccup
#: in than a ``stat``, and the extra two attempts cost nothing on a healthy run.
DEFAULT_READ_ATTEMPTS = 6

#: First backoff, doubled each attempt and capped at
#: :data:`~jlens.mmpilot.manifest.MAX_PROBE_DELAY`. Worst case is under 4 s.
DEFAULT_READ_DELAY = 0.1


class MediaDecodeError(MediaIOError):
    """The bytes were read but are not a decodable image/recording.

    Separate from a transport failure on purpose: retrying cannot repair a
    truncated JPEG, and treating it as a mount hiccup would spend four more
    attempts to arrive at the same answer more slowly.
    """


@dataclass
class RetryJournal:
    """Every retried failure this run survived, and what it cost.

    Passed into the loaders so a study can report mount flakiness as a fact
    about the environment instead of discovering it in a traceback.
    """

    entries: list[dict] = field(default_factory=list)

    def record(self, entry: Mapping) -> None:
        self.entries.append(dict(entry))

    @property
    def n_retries(self) -> int:
        return len(self.entries)

    @property
    def n_paths(self) -> int:
        return len({str(entry.get("path")) for entry in self.entries})

    def to_dict(self) -> dict:
        by_errno: dict[str, int] = {}
        for entry in self.entries:
            name = str(entry.get("errno_name") or entry.get("errno"))
            by_errno[name] = by_errno.get(name, 0) + 1
        return {
            "schema": "jlens.mmpilot.media_retry_journal.v1",
            "media_io_version": MEDIA_IO_VERSION,
            "n_retries": self.n_retries,
            "n_paths_affected": self.n_paths,
            "retries_by_errno": dict(sorted(by_errno.items())),
            "entries": [dict(entry) for entry in self.entries[:50]],
            "entries_truncated": max(0, self.n_retries - 50),
        }


def _read_once(path: Path) -> bytes:
    """Indirection so tests can inject filesystem failures deterministically."""
    with open(path, "rb") as handle:
        return handle.read()


def read_media_bytes(
    path: str | os.PathLike[str],
    *,
    attempts: int = DEFAULT_READ_ATTEMPTS,
    initial_delay: float = DEFAULT_READ_DELAY,
    journal: RetryJournal | None = None,
) -> bytes:
    """The whole file, with bounded retries on transient mount failures.

    Args:
        attempts: Total tries including the first.
        journal: Appended to once per *retried* failure.

    Raises:
        MediaIOError: On an absent path, a permission error (never retried), an
            errno not known to be transient (never retried — guessing that an
            unrecognised failure means "try again" is how a real corruption
            becomes four identical stack traces), or transient errnos that do
            not clear within ``attempts``.
    """
    target = Path(path)
    observed: list[dict] = []
    delay = float(initial_delay)
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            return _read_once(target)
        except FileNotFoundError as exc:
            raise MediaIOError(
                f"{target} is not present; refusing to treat a missing media "
                "file as a transient mount failure"
            ) from exc
        except PermissionError as exc:
            raise MediaIOError(
                f"{target} could not be read: permission denied. This is not a "
                "mount hiccup and waiting cannot fix it."
            ) from exc
        except OSError as exc:
            if exc.errno in ABSENT_ERRNOS:
                raise MediaIOError(
                    f"{target} is not present ("
                    f"{errno_module.errorcode.get(exc.errno, exc.errno)})"
                ) from exc
            entry = {
                "attempt": attempt,
                "path": str(target),
                "errno": exc.errno,
                "errno_name": errno_module.errorcode.get(exc.errno, "unknown"),
                "error": str(exc),
            }
            observed.append(entry)
            if exc.errno not in TRANSIENT_ERRNOS:
                raise MediaIOError(
                    f"{target} failed to read with "
                    f"{entry['errno_name']}, which is not a known transient "
                    "mount error. Refusing to retry an error whose meaning is "
                    "unknown."
                ) from exc
            if attempt >= int(attempts):
                raise MediaIOError(
                    f"{target} still failed with {entry['errno_name']} after "
                    f"{attempt} attempt(s).\n"
                    "Remount Google Drive and re-run: drive.flush_and_unmount() "
                    "then drive.mount('/content/drive', force_remount=True). "
                    "The run directory resumes, so at most the in-flight unit "
                    "is lost."
                ) from exc
            if journal is not None:
                journal.record(entry)
            _sleep(min(delay, MAX_PROBE_DELAY))
            delay = min(delay * 2, MAX_PROBE_DELAY)
    raise MediaIOError(  # pragma: no cover - loop either returns or raises
        f"{target} exhausted its retries without a recorded reason"
    )


def media_checksum(payload: bytes) -> str:
    """``sha256:`` of the bytes that were actually decoded."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_image_bytes(
    path: str | os.PathLike[str],
    *,
    attempts: int = DEFAULT_READ_ATTEMPTS,
    journal: RetryJournal | None = None,
):
    """A retry-loaded RGB :class:`PIL.Image.Image`.

    ``Image.open`` is given a :class:`io.BytesIO`, and ``load()`` is forced
    before the buffer goes out of scope, so no lazy read can reach back to the
    mount after this function returns.
    """
    from PIL import Image, UnidentifiedImageError

    payload = read_media_bytes(path, attempts=attempts, journal=journal)
    try:
        with Image.open(io.BytesIO(payload)) as handle:
            image = handle.convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MediaDecodeError(
            f"{path} was read ({len(payload)} bytes) but is not a decodable "
            f"image: {exc}. Retrying cannot repair a corrupt file."
        ) from exc
    return image


def load_audio_bytes(
    path: str | os.PathLike[str],
    *,
    attempts: int = DEFAULT_READ_ATTEMPTS,
    journal: RetryJournal | None = None,
) -> tuple:
    """``(waveform, sample_rate)`` from a retry-loaded byte buffer.

    Multi-channel audio is averaged to mono, which is what the audio tower is
    fed everywhere else in this repository. Nothing is resampled here: the
    sample rate is returned as recorded, and the audio protocol decides what to
    do with it.
    """
    import soundfile as sf

    payload = read_media_bytes(path, attempts=attempts, journal=journal)
    try:
        waveform, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")
    except (RuntimeError, ValueError) as exc:
        raise MediaDecodeError(
            f"{path} was read ({len(payload)} bytes) but is not a decodable "
            f"recording: {exc}. Retrying cannot repair a corrupt file."
        ) from exc
    if getattr(waveform, "ndim", 1) > 1:
        waveform = waveform.mean(axis=1)
    return waveform, int(sample_rate)


def drive_media_loaders(
    *,
    attempts: int = DEFAULT_READ_ATTEMPTS,
    journal: RetryJournal | None = None,
) -> dict[str, Callable]:
    """The ``{"load_image", "load_audio"}`` mapping the pipeline expects.

    One call site, so a notebook cannot retry images and forget recordings —
    which is the failure mode, since the audio arm is both the slowest and the
    one whose files live deepest in the SpokenCOCO tree.
    """
    return {
        "load_image": lambda path: load_image_bytes(
            path, attempts=attempts, journal=journal
        ),
        "load_audio": lambda path: load_audio_bytes(
            path, attempts=attempts, journal=journal
        ),
    }


__all__ = [
    "DEFAULT_READ_ATTEMPTS",
    "DEFAULT_READ_DELAY",
    "MEDIA_IO_VERSION",
    "MediaDecodeError",
    "MediaIOError",
    "RetryJournal",
    "drive_media_loaders",
    "load_audio_bytes",
    "load_image_bytes",
    "media_checksum",
    "read_media_bytes",
]
