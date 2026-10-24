"""Verified single-file result caches with bounded, read-only batch access."""

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from uuid import uuid4
from zipfile import ZipFile, ZIP_STORED

from reaxkit.core.runtime.result_store import ResultStore, SCHEMA_VERSION, _digest, _sync_directory


@dataclass(frozen=True)
class ArchivePath:
    archive: ZipFile
    member: PurePosixPath = PurePosixPath()

    def __truediv__(self, name):
        return ArchivePath(self.archive, self.member / name)

    @property
    def parent(self):
        return ArchivePath(self.archive, self.member.parent)

    @property
    def name(self):
        return self.member.name

    @property
    def suffix(self):
        return self.member.suffix

    def open(self, mode="rb"):
        if mode != "rb":
            raise ValueError("Packed results are read-only.")
        return self.archive.open(str(self.member), "r")

    def read_text(self, encoding="utf-8"):
        with self.open() as stream:
            return stream.read().decode(encoding)

    def exists(self):
        try:
            self.archive.getinfo(str(self.member))
            return True
        except KeyError:
            return False

    def __str__(self):
        return f"{self.archive.filename}!/{self.member}"


class PackedResultStore(ResultStore):
    """Replay completed analyses directly from an archive without extraction."""

    def __init__(self, path, identity):
        self._archive = ZipFile(path, "r")
        self._locked = False
        self.directory = ArchivePath(self._archive)
        try:
            self.manifest = json.loads((self.directory / "manifest.json").read_text())
            normalized = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
            if (self.manifest["schema_version"] != SCHEMA_VERSION
                    or self.manifest["state"] != "complete"
                    or self.manifest["identity"] != normalized):
                raise ValueError("Incompatible or incomplete packed result cache.")
            self.validate()
            if self.manifest.get("result_descriptor"):
                if _digest(self.directory / "result.json") != self.manifest["result_descriptor"]:
                    raise ValueError("Damaged packed result descriptor.")
        except BaseException:
            self.close()
            raise

    def transition(self, state, **kwargs):
        if state not in {"analysis_complete", "complete"}:
            raise ValueError("Packed results are read-only.")

    def close(self):
        self._archive.close()

    def __exit__(self, kind, error, traceback):
        self.close()

    def __del__(self):
        if hasattr(self, "_archive"):
            self._archive.close()


def publish_packed_results(store, destination):
    """Atomically publish and read back all committed scientific data."""
    store._owned()
    if store.manifest["state"] != "analysis_complete":
        raise ValueError("Only completed analyses can be packed.")
    destination = Path(destination).resolve()
    if destination.is_relative_to(store.directory):
        raise ValueError("Result cache must be outside the recovery checkpoint.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}-{uuid4().hex}.tmp")
    try:
        with ZipFile(temporary, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            manifest = {**store.manifest, "state": "complete", "error": None, "failed_source_frame": None}
            archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, allow_nan=False))
            if manifest.get("result_descriptor"):
                archive.write(store.directory / "result.json", "result.json")
            for batch, metadata in store.batches():
                paths = [batch / "batch.json"]
                paths.extend(batch / entry["file"] for entry in metadata["tables"].values())
                if "state_sha256" in metadata:
                    paths.append(batch / "state.json")
                for path in paths:
                    if not path.resolve().is_relative_to(store.directory) or path.is_symlink():
                        raise ValueError(f"Checkpoint member escapes its directory: {path}")
                    archive.write(path, path.relative_to(store.directory).as_posix())
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        with PackedResultStore(temporary, store.manifest["identity"]):
            pass
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
        return PackedResultStore(destination, store.manifest["identity"])
    finally:
        temporary.unlink(missing_ok=True)


def remove_packed_checkpoint(store, packed):
    """Release recovery files while keeping existing lazy result tables usable."""
    store._owned()
    directory = store.directory
    if store.manifest["state"] != "complete" or directory != store.manifest_path.parent:
        raise ValueError("Only the owned, completed checkpoint can be removed.")
    allowed = {"manifest.json", "result.json", "writer.lock", "batches"}
    if any(path.name not in allowed and not path.name.startswith((".s-", "orphan-"))
           for path in directory.iterdir()):
        raise ValueError("Checkpoint contains unrelated files; retaining it.")
    if directory.is_symlink() or directory.resolve() != directory:
        raise ValueError("Checkpoint directory changed; retaining it.")
    removing = directory.with_name(f".{directory.name}-cleanup-{uuid4().hex}")
    directory.rename(removing)
    store.directory = removing
    store.manifest_path = removing / "manifest.json"
    store.lock_path = removing / "writer.lock"
    store.close()
    store._packed = packed
    store.directory = packed.directory
    shutil.rmtree(removing)
    if directory.name == "checkpoint" and directory.parent.name.startswith("recovery-"):
        try:
            directory.parent.rmdir()
        except OSError:
            pass
