"""Pack cropped handwriting images into a single LMDB file.

The problem this solves is the one the brief calls trap number one. CVL's word
images are 99,904 separate files. Reading many small files from Google Drive is
slow enough that the GPU sits idle waiting for them -- Drive is a network
filesystem with per-file overhead, and the overhead, not the bytes, is the cost.

**Words or lines.** The record shape is the same for both -- an image, a writer, a
transcription -- so one pack format serves both units, and which unit a given
file holds is recorded in its header's ``source``. The distinction matters
downstream rather than here: the generator is trained on lines and evaluated on
lines, while the style embedding was trained on words.

So: pack once into a single file, keep that file on Drive, and copy it to the
Colab VM's local disk at the start of each session. One large sequential copy
instead of a hundred thousand small random reads.

**What is stored.** Each record is a PNG-encoded grayscale image, already
normalised to the configured height, plus its writer, transcription and split.
Baking the height in makes loading cheap; the height is recorded in the database
header so a config that disagrees with the pack is caught rather than silently
producing wrongly-sized batches.

**Why PNG rather than raw arrays.** Raw would avoid a decode per sample, but the
file would be several times larger, and the copy from Drive is the thing being
optimised. PNG decode of a 64-pixel-tall crop is not the bottleneck.
"""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import lmdb
import numpy as np

FORMAT_VERSION = 1
HEADER_KEY = b"__header__"
KEYS_KEY = b"__keys__"

# Address space reserved for the database. On Linux this costs nothing until
# written; on Windows the file is created at full size, so a wildly generous
# value is not free. 100k word crops at 64px are a few hundred megabytes, so 8 GB
# is roughly twenty times what is needed -- ample, without creating a 40 GB file.
DEFAULT_MAP_SIZE = 8 * 1024**3

# Records per write transaction. One transaction per record is the classic LMDB
# performance mistake: each one fsyncs. Batching is worth an order of magnitude.
WRITE_BATCH = 2000


class PackError(RuntimeError):
    pass


@dataclass(frozen=True)
class PackedSample:
    """One cropped image and its labels, as stored. A word or a whole line."""

    key: str
    writer_id: str
    text: str
    split: str
    image: np.ndarray


@dataclass
class PackHeader:
    """What the pack is, so a mismatch is caught rather than guessed at."""

    format_version: int = FORMAT_VERSION
    height: int = 64
    charset: str = "english"
    source: str = "cvl"
    count: int = 0
    writers: int = 0
    config: dict = field(default_factory=dict)


class PackWriter:
    """Write a pack. Use as a context manager so the header is always finalised."""

    def __init__(self, path: Path | str, header: PackHeader, map_size: int = DEFAULT_MAP_SIZE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = header
        self._env = lmdb.open(str(self.path), map_size=map_size, subdir=False, lock=True)
        self._keys: list[str] = []
        self._writers: set[str] = set()
        self._pending: list[tuple[bytes, bytes]] = []

    def add(self, sample: PackedSample) -> None:
        if sample.image.ndim != 2:
            raise PackError(
                f"{sample.key}: expected a grayscale image, got shape {sample.image.shape}"
            )
        if sample.image.shape[0] != self.header.height:
            raise PackError(
                f"{sample.key}: height {sample.image.shape[0]} does not match the pack's "
                f"declared height {self.header.height}"
            )

        ok, encoded = cv2.imencode(".png", sample.image)
        if not ok:
            raise PackError(f"{sample.key}: PNG encoding failed")

        payload = {
            "writer_id": sample.writer_id,
            "text": sample.text,
            "split": sample.split,
            "png": encoded.tobytes(),
        }
        self._pending.append((sample.key.encode(), pickle.dumps(payload, protocol=5)))
        self._keys.append(sample.key)
        self._writers.add(sample.writer_id)
        if len(self._pending) >= WRITE_BATCH:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        with self._env.begin(write=True) as txn:
            for key, value in self._pending:
                txn.put(key, value)
        self._pending.clear()

    def close(self) -> None:
        """Write the key list and header last.

        Order matters: a pack whose header is missing is detectably incomplete,
        whereas one with a header written first would look finished after an
        interrupted run.
        """
        self._flush()
        self.header.count = len(self._keys)
        self.header.writers = len(self._writers)
        with self._env.begin(write=True) as txn:
            txn.put(KEYS_KEY, pickle.dumps(self._keys, protocol=5))
            txn.put(HEADER_KEY, json.dumps(self.header.__dict__).encode())
        self._env.sync()
        self._env.close()

    def __enter__(self) -> PackWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class PackReader:
    """Read a pack. Cheap to construct; opens the environment lazily per worker.

    The lazy open matters: an LMDB environment cannot be shared across processes
    created by fork, so a DataLoader with several workers must open its own.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no pack at {self.path}")
        self._env = None

        # Every failure below closes the environment before raising. LMDB refuses
        # to open the same file twice in one process, so a leaked handle from a
        # failed open turns one honest error into every later open failing too --
        # which is what happened the first time this was run against a pack that
        # was still being built.
        try:
            env = self._open()
            with env.begin() as txn:
                raw_header = txn.get(HEADER_KEY)
                raw_keys = txn.get(KEYS_KEY)
            if raw_header is None or raw_keys is None:
                raise PackError(
                    f"{self.path} has no header. The pack was not closed properly -- "
                    "it is either still being written, or the truncated result of an "
                    "interrupted run. Rebuild it, or wait for the build to finish."
                )
            self.header = PackHeader(**json.loads(raw_header.decode()))
            if self.header.format_version != FORMAT_VERSION:
                raise PackError(
                    f"{self.path} is pack format {self.header.format_version}, "
                    f"this code reads {FORMAT_VERSION}"
                )
            self.keys: list[str] = pickle.loads(raw_keys)
        except Exception:
            self.close()
            raise

    def _open(self):
        if self._env is None:
            self._env = lmdb.open(
                str(self.path), subdir=False, readonly=True, lock=False, readahead=False
            )
        return self._env

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int | str) -> PackedSample:
        key = self.keys[index] if isinstance(index, int) else index
        with self._open().begin() as txn:
            raw = txn.get(key.encode())
        if raw is None:
            raise KeyError(f"{key} is not in {self.path}")
        payload = pickle.loads(raw)
        image = cv2.imdecode(np.frombuffer(payload["png"], np.uint8), cv2.IMREAD_GRAYSCALE)
        return PackedSample(
            key=key,
            writer_id=payload["writer_id"],
            text=payload["text"],
            split=payload["split"],
            image=image,
        )

    def __iter__(self) -> Iterator[PackedSample]:
        for index in range(len(self)):
            yield self[index]

    def writers(self) -> dict[str, list[str]]:
        """Writer id -> its keys. Read from the keys alone, without decoding images."""
        grouped: dict[str, list[str]] = {}
        with self._open().begin() as txn:
            for key in self.keys:
                payload = pickle.loads(txn.get(key.encode()))
                grouped.setdefault(payload["writer_id"], []).append(key)
        return grouped

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self) -> PackReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def data_size_bytes(self) -> int:
        """Bytes actually used, not the address space reserved.

        stat() on the file reports the map size, which on Windows is the full
        reservation -- it read as 40 GB for a 500-record test pack, which is the
        sort of number that makes you go looking for a bug that is not there.
        """
        env = self._open()
        return int(env.info()["last_pgno"] + 1) * int(env.stat()["psize"])

    def summary(self) -> str:
        size_mb = self.data_size_bytes() / 1024**2
        return "\n".join(
            [
                f"pack        {self.path.name}",
                f"records     {len(self)}",
                f"writers     {self.header.writers}",
                f"height      {self.header.height}px",
                f"charset     {self.header.charset}",
                f"size        {size_mb:.0f} MB",
            ]
        )


def compact(path: Path | str, destination: Path | str | None = None) -> Path:
    """Rewrite a pack at its true size, replacing the original by default.

    LMDB reserves its whole map size up front. On this machine that produced a
    file reporting 8 GB while occupying 40 MB of disk -- harmless locally, because
    the file is sparse, and **not** harmless the moment it is copied anywhere.
    Uploading to Drive, or `cp` without --sparse, transfers the apparent 8 GB.
    That would defeat the entire purpose of packing, so the shipped artefact is
    always a compacted copy.
    """
    path = Path(path)

    # A unique temporary name, not a fixed ".compact" suffix. An earlier run of
    # this died overnight and left its half-written file locked by a process that
    # never exited; every retry then failed on PermissionError trying to delete
    # it. A leftover from a dead run must not be able to block a live one.
    if destination is not None:
        target = Path(destination)
        target.unlink(missing_ok=True)
    else:
        target = path.with_suffix(f"{path.suffix}.compact.{os.getpid()}")

    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False)
    try:
        env.copy(str(target), compact=True)
    finally:
        env.close()

    if destination is None:
        path.unlink()
        Path(str(path) + "-lock").unlink(missing_ok=True)
        target.rename(path)
        return path
    return target


def is_complete(path: Path | str) -> bool:
    """Whether a pack exists and was finished.

    A pack that is mid-build exists on disk but has no header. Callers that want
    to skip rather than fail -- tests, mostly -- need to tell those apart without
    treating an incomplete file as a hard error.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        PackReader(path).close()
    except (PackError, lmdb.Error):
        return False
    return True
