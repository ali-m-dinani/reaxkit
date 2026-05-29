"""
Run-scoped storage layout utilities.

This module implements the project layout:
  inputs/<run_id>/
  data/raw/<run_id>/
  data/parsed/<parsed_id>/
  data/run_index/<run_id>.json
  analysis/<workflow>/<analysis_id>/

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from reaxkit.core.storage.parsed_store import load_parsed_hdf5, update_parsed_meta, write_parsed_hdf5

_DEFAULT_SNAPSHOT_FILES: tuple[str, ...] = (
    "xmolout",
    "summary.txt",
    "fort.7",
    "fort.13",
    "fort.57",
    "fort.73",
    "fort.74",
    "fort.76",
    "fort.78",
    "fort.79",
    "fort.99",
    "trainset.in",
    "params",
    "control",
    "eregime.in",
    "vels",
    "molfra.out",
    "ffield",
    "dump.lammpstrj",
    "lammpstrj",
    "log.lammps",
)

def _utc_now_iso() -> str:
    """
    Utc now iso.
    """
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default):
    """
    Read json.
    """
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    """
    Write json.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_run_id(run_id: str) -> str:
    """
    Safe run id.
    """
    value = str(run_id or "").strip()
    if not value:
        raise ValueError("run_id cannot be empty.")
    return value


def default_project_root() -> Path:
    """
    Default project root.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.storage.storage_layout import default_project_root
    # Configure required arguments for your case.
    result = default_project_root(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return Path("reaxkit_workspace")


def generate_run_id() -> str:
    """
    Generate run id.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.storage.storage_layout import generate_run_id
    # Configure required arguments for your case.
    result = generate_run_id(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = secrets.token_hex(3)
    return f"run_{stamp}_{token}"


def file_sha256(path: Path) -> str:
    """
    File sha256.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    path : Path
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.storage.storage_layout import file_sha256
    # Configure required arguments for your case.
    result = file_sha256(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(root: Path) -> str:
    """
    Directory fingerprint.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    root : Path
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.storage.storage_layout import directory_fingerprint
    # Configure required arguments for your case.
    result = directory_fingerprint(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if not root.exists():
        return "missing"
    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        return "empty"
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: str(p.relative_to(root)).replace("\\", "/")):
        rel = str(path.relative_to(root)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def parsed_id_from_raw_and_handler(*, raw_hash: str, handler_version: str) -> str:
    """
    Parsed id from raw and handler.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    raw_hash : str
        Input parameter used by this function.
    handler_version : str
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.storage.storage_layout import parsed_id_from_raw_and_handler
    # Configure required arguments for your case.
    result = parsed_id_from_raw_and_handler(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    seed = f"{str(raw_hash)}|{str(handler_version)}"
    return f"parsed_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"


@dataclass(frozen=True)
class ReaxkitStorageLayout:
    """
    Reaxkit Storage Layout.
    
    This dataclass defines a structured container used by ReaxKit core workflows.
    
    Fields
    -----
    project_root : Path, optional
        Field value used by this structured record.
    """
    project_root: Path = Path("..")

    @property
    def inputs_root(self) -> Path:
        """
        Inputs root.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.inputs_root(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.project_root / "inputs"

    @property
    def data_root(self) -> Path:
        """
        Data root.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.data_root(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.project_root / "data"

    @property
    def raw_root(self) -> Path:
        """
        Raw root.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.raw_root(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.data_root / "raw"

    @property
    def parsed_root(self) -> Path:
        """
        Parsed root.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.parsed_root(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.data_root / "parsed"

    @property
    def run_index_root(self) -> Path:
        """
        Run index root.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.run_index_root(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.data_root / "run_index"

    @property
    def parsed_index_path(self) -> Path:
        """
        Parsed index path.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.parsed_index_path(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.parsed_root / "index.json"

    @property
    def analysis_root(self) -> Path:
        """
        Analysis root.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.analysis_root(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.project_root / "analysis"

    @property
    def figures_root(self) -> Path:
        """
        Figures root.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.figures_root(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.project_root / "figures"

    @property
    def reports_root(self) -> Path:
        """
        Reports root.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.reports_root(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.project_root / "reports"

    @property
    def cache_root(self) -> Path:
        """
        Cache root.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.cache_root(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.project_root / "cache"

    @property
    def logs_root(self) -> Path:
        """
        Logs root.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        None
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.logs_root(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.project_root / "logs"

    def ensure_base_layout(self, *, include_inputs: bool = False) -> None:
        """
        Ensure base layout.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        include_inputs : bool, optional
            Input parameter used by this function.
        
        Returns
        -----
        None
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.ensure_base_layout(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        roots = [
            self.raw_root,
            self.parsed_root,
            self.run_index_root,
            self.analysis_root,
            self.reports_root,
            self.cache_root,
            self.logs_root,
        ]
        if include_inputs:
            roots.insert(0, self.inputs_root)
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)

    def input_run_dir(self, run_id: str) -> Path:
        """
        Input run dir.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        run_id : str
            Input parameter used by this function.
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.input_run_dir(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.inputs_root / _safe_run_id(run_id)

    def raw_run_dir(self, run_id: str) -> Path:
        """
        Raw run dir.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        run_id : str
            Input parameter used by this function.
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.raw_run_dir(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.raw_root / _safe_run_id(run_id)

    def run_index_path(self, run_id: str) -> Path:
        """
        Run index path.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        run_id : str
            Input parameter used by this function.
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.run_index_path(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.run_index_root / f"{_safe_run_id(run_id)}.json"

    def parsed_dir(self, parsed_id: str) -> Path:
        """
        Parsed dir.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        parsed_id : str
            Input parameter used by this function.
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.parsed_dir(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        return self.parsed_root / str(parsed_id)

    def ensure_run_layout(self, run_id: str, *, include_inputs: bool = False) -> None:
        """
        Ensure run layout.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        run_id : str
            Input parameter used by this function.
        include_inputs : bool, optional
            Input parameter used by this function.
        
        Returns
        -----
        None
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.ensure_run_layout(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        self.ensure_base_layout(include_inputs=include_inputs)
        if include_inputs:
            self.input_run_dir(run_id).mkdir(parents=True, exist_ok=True)
        self.raw_run_dir(run_id).mkdir(parents=True, exist_ok=True)

    def ensure_analysis_run_layout(self, run_id: str) -> None:
        """
        Ensure analysis run layout.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        run_id : str
            Input parameter used by this function.
        
        Returns
        -----
        None
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.ensure_analysis_run_layout(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        self.ensure_run_layout(run_id, include_inputs=False)

    def ensure_input_run_layout(self, run_id: str) -> None:
        """
        Ensure input run layout.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        run_id : str
            Input parameter used by this function.
        
        Returns
        -----
        None
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.ensure_input_run_layout(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        self.ensure_run_layout(run_id, include_inputs=True)

    def register_parsed_dataset(
        self,
        *,
        run_id: str,
        handler_version: str,
        engine: str,
    ) -> str:
        """
        Register parsed dataset.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        run_id : str
            Input parameter used by this function.
        handler_version : str
            Input parameter used by this function.
        engine : str
            Input parameter used by this function.
        
        Returns
        -----
        str
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.register_parsed_dataset(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        run_id = _safe_run_id(run_id)
        self.ensure_analysis_run_layout(run_id)
        raw_dir = self.raw_run_dir(run_id)
        raw_hash = directory_fingerprint(raw_dir)
        parsed_id = parsed_id_from_raw_and_handler(raw_hash=raw_hash, handler_version=handler_version)
        parsed_dir = self.parsed_dir(parsed_id)
        parsed_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            parsed_dir / "meta.json",
            {
                "parsed_id": parsed_id,
                "run_id": run_id,
                "raw_hash": raw_hash,
                "run_hash": raw_hash,
                "engine": engine,
                "handler_version": handler_version,
                "updated_at": _utc_now_iso(),
            },
        )
        _write_json(
            self.run_index_path(run_id),
            {
                "run_id": run_id,
                "raw_dir": str(raw_dir),
                "raw_hash": raw_hash,
                "run_hash": raw_hash,
                "parsed_id": parsed_id,
                "engine": engine,
                "handler_version": handler_version,
                "updated_at": _utc_now_iso(),
            },
        )
        return parsed_id

    def persist_parsed_artifact(
        self,
        *,
        parsed_id: str,
        artifact_name: str,
        data: Any,
    ) -> Path:
        """
        Persist parsed artifact.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        parsed_id : str
            Input parameter used by this function.
        artifact_name : str
            Input parameter used by this function.
        data : Any
            Input parameter used by this function.
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.persist_parsed_artifact(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        parsed_dir = self.parsed_dir(parsed_id)
        parsed_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in str(artifact_name)).strip("_")
        if not safe_name:
            safe_name = "parsed_data"
        file_name = f"{safe_name}.h5"
        out_path = write_parsed_hdf5(parsed_dir / file_name, data)
        update_parsed_meta(parsed_dir, parsed_id=str(parsed_id), artifact_name=safe_name, file_name=file_name)
        return out_path

    def load_parsed_artifact(
        self,
        *,
        parsed_id: str,
        artifact_name: str,
    ) -> Any | None:
        """
        Load parsed artifact.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        parsed_id : str
            Input parameter used by this function.
        artifact_name : str
            Input parameter used by this function.
        
        Returns
        -----
        Any | None
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.load_parsed_artifact(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        parsed_dir = self.parsed_dir(parsed_id)
        meta_path = parsed_dir / "meta.json"
        safe_name = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in str(artifact_name)).strip("_")
        if not safe_name:
            safe_name = "parsed_data"
        default_file = f"{safe_name}.h5"
        target = parsed_dir / default_file
        if meta_path.exists():
            payload = _read_json(meta_path, default={})
            file_name = str(((payload.get("artifacts") or {}).get(safe_name) or {}).get("file") or default_file)
            target = parsed_dir / file_name
        if not target.exists():
            return None
        try:
            return load_parsed_hdf5(target)
        except Exception:
            return None

    def record_run_analysis(
        self,
        *,
        run_id: str,
        parsed_id: str | None,
        analysis_id: str,
        task_name: str,
        task_version: str = "1",
    ) -> Path:
        """
        Record run analysis.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        run_id : str
            Input parameter used by this function.
        parsed_id : str | None
            Input parameter used by this function.
        analysis_id : str
            Input parameter used by this function.
        task_name : str
            Input parameter used by this function.
        task_version : str, optional
            Input parameter used by this function.
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.record_run_analysis(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        run_id = _safe_run_id(run_id)
        path = self.run_index_path(run_id)
        payload = _read_json(path, default={})
        if not payload:
            payload = {"run_id": run_id}

        analyses = list(payload.get("analyses") or [])
        existing_idx = None
        for i, entry in enumerate(analyses):
            if str(entry.get("analysis_id", "")) == str(analysis_id):
                existing_idx = i
                break

        record = {
            "analysis_id": str(analysis_id),
            "task": str(task_name),
            "task_version": str(task_version),
            "parsed_id": str(parsed_id) if parsed_id is not None else None,
            "updated_at": _utc_now_iso(),
        }
        if existing_idx is None:
            analyses.append(record)
        else:
            analyses[existing_idx] = record

        payload["analyses"] = analyses
        payload["updated_at"] = _utc_now_iso()
        _write_json(path, payload)
        return path

    def record_run_generator(
        self,
        *,
        run_id: str,
        command: str,
        output_path: str | Path,
        settings_path: str | Path | None = None,
    ) -> Path:
        """
        Record run generator.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        run_id : str
            Input parameter used by this function.
        command : str
            Input parameter used by this function.
        output_path : str | Path
            Input parameter used by this function.
        settings_path : str | Path | None, optional
            Input parameter used by this function.
        
        Returns
        -----
        Path
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout
        instance = ReaxkitStorageLayout(...)
        # Configure required arguments for your case.
        result = instance.record_run_generator(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        run_id = _safe_run_id(run_id)
        path = self.run_index_path(run_id)
        payload = _read_json(path, default={})
        if not payload:
            payload = {"run_id": run_id}

        entries = list(payload.get("generators") or [])
        entry_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
        entries.append(
            {
                "generator_id": entry_id,
                "command": str(command),
                "output_path": str(output_path),
                "settings_path": str(settings_path) if settings_path is not None else None,
                "updated_at": _utc_now_iso(),
            }
        )
        payload["generators"] = entries
        payload["updated_at"] = _utc_now_iso()
        _write_json(path, payload)
        return path


def add_storage_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add storage cli arguments.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    parser : argparse.ArgumentParser
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
    # Configure required arguments for your case.
    result = add_storage_cli_arguments(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    parser.add_argument("--run-id", default=None, help="Run identifier for run-scoped layout (e.g., run_91ac0e).")
    parser.add_argument(
        "--project-root",
        default=str(default_project_root()),
        help="Project root that contains inputs/, data/, analysis/, etc.",
    )
    parser.add_argument("--analysis-id", default=None, help="Optional analysis artifact id; defaults to run id.")


def _copy_if_exists(src: Path, dst: Path) -> None:
    """
    Copy if exists.
    """
    if not src.exists() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)


def _detect_snapshot_source(out: dict) -> Path:
    """
    Detect snapshot source.
    """
    detection_keys = ("input", "run_dir", "xmolout", "fort7", "summary")
    source = out.get("_snapshot_source_dir")
    if source:
        p = Path(str(source))
        if p.exists():
            return p if p.is_dir() else p.parent
    chosen = None
    for key in detection_keys:
        value = out.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.exists():
            chosen = path if path.is_dir() else path.parent
            break
    if chosen is None:
        chosen = Path("..")
    return chosen


def _snapshot_raw_from_source(source: Path, raw_dir: Path, *, names: Sequence[str] | None = None) -> None:
    """
    Snapshot raw from source.
    """
    if source.resolve() == raw_dir.resolve():
        return

    snapshot_names = tuple(names) if names is not None else _DEFAULT_SNAPSHOT_FILES
    for name in snapshot_names:
        _copy_if_exists(source / name, raw_dir / name)


def snapshot_storage_inputs(args: dict, *, names: Sequence[str] | None = None) -> None:
    """
    Copy source raw inputs into run-scoped ``data/raw/<run_id>``.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    args : dict
        Input parameter used by this function.
    names : Sequence[str] | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.storage.storage_layout import snapshot_storage_inputs
    # Configure required arguments for your case.
    result = snapshot_storage_inputs(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    run_id = args.get("run_id")
    if not run_id:
        run_id = generate_run_id()
        args["run_id"] = run_id
    project_root = Path(args.get("project_root") or default_project_root())
    args["project_root"] = str(project_root)
    layout = ReaxkitStorageLayout(project_root=project_root)
    layout.ensure_analysis_run_layout(str(run_id))
    raw_dir = layout.raw_run_dir(str(run_id))
    source = _detect_snapshot_source(args)
    args["_snapshot_source_dir"] = str(source)
    _snapshot_raw_from_source(source, raw_dir, names=names)


def normalize_storage_args(
    args: dict,
    *,
    snapshot: bool = True,
    snapshot_inputs: Sequence[str] | None = None,
) -> dict:
    """
    Normalize storage args.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    args : dict
        Input parameter used by this function.
    snapshot : bool, optional
        Input parameter used by this function.
    snapshot_inputs : Sequence[str] | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    dict
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.storage.storage_layout import normalize_storage_args
    # Configure required arguments for your case.
    result = normalize_storage_args(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    out = dict(args)
    run_id = out.get("run_id")
    if not run_id:
        run_id = generate_run_id()
        out["run_id"] = run_id

    project_root = Path(out.get("project_root") or default_project_root())
    out["project_root"] = str(project_root)
    layout = ReaxkitStorageLayout(project_root=project_root)
    layout.ensure_analysis_run_layout(str(run_id))
    out["_snapshot_source_dir"] = str(_detect_snapshot_source(out))
    raw_dir = layout.raw_run_dir(str(run_id))
    if snapshot:
        snapshot_storage_inputs(out, names=snapshot_inputs)

    if not out.get("input") or str(out.get("input")) == ".":
        out["input"] = str(raw_dir)
    if not out.get("run_dir") or str(out.get("run_dir")) == ".":
        out["run_dir"] = str(raw_dir)
    if not out.get("cache_dir"):
        out["cache_dir"] = str(layout.cache_root)

    # Rewrite default bare filenames to the run-scoped raw directory.
    default_files = {
        "xmolout": "xmolout",
        "summary": "summary.txt",
        "fort7": "fort.7",
        "fort13": "fort.13",
        "fort57": "fort.57",
        "fort73": "fort.73",
        "fort74": "fort.74",
        "fort76": "fort.76",
        "fort78": "fort.78",
        "fort79": "fort.79",
        "fort99": "fort.99",
        "trainset": "trainset.in",
        "params": "params",
        "control": "control",
        "eregime": "eregime.in",
        "vels": "vels",
        "molfra": "molfra.out",
        "ffield": "ffield",
    }
    for key, default_name in default_files.items():
        raw = out.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_absolute():
            continue
        if path.parent != Path(".."):
            continue
        if path.name != default_name:
            continue
        out[key] = str(raw_dir / default_name)
    return out
