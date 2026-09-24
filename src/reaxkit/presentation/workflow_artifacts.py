"""Atomic publication for workflows with established public table filenames."""

from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar

from reaxkit.core.runtime.artifacts import ArtifactSpec, ArtifactWriter, TableChunks

_CURRENT_ARGS = ContextVar("reaxkit_artifact_args", default=None)


@contextmanager
def workflow_csv_rows(path, columns, *, args=None):
    """Bounded adapter for legacy csv.writer/DictWriter row producers."""
    import pandas as pd
    path = Path(path)
    args = args or _CURRENT_ARGS.get()
    spec = ArtifactSpec(path.name, path.name, "core", True, "csv", True)
    with ArtifactWriter(path.parent, [spec], overwrite=True,
                        manifest_name=path.name + ".artifacts.json",
                        profile=getattr(args, "output_profile", "standard"),
                        metadata={"command": getattr(args, "command", None)}) as artifacts:
        class Rows:
            def __init__(self):
                self.buffer = []

            def writeheader(self):
                pass  # The shared sink writes the declared header exactly once.

            def writerow(self, row):
                values = [row.get(name) for name in columns] if isinstance(row, dict) else list(row)
                if len(values) != len(columns):
                    raise ValueError("CSV row width differs from its declared schema.")
                self.buffer.append(["" if value is None else str(value) for value in values])
                if len(self.buffer) >= 2048:
                    self.flush()

            def flush(self):
                artifacts.append(path.name, pd.DataFrame(self.buffer, columns=columns))
                self.buffer.clear()

        rows = Rows()
        yield rows
        rows.flush()


@contextmanager
def workflow_artifact_policy(args):
    """Make the invocation policy available to nested artifact helpers."""
    token = _CURRENT_ARGS.set(args)
    try:
        yield
    finally:
        _CURRENT_ARGS.reset(token)


def write_workflow_csv(table, path, *, index=False, args=None, tier="core"):
    """Compatibility adapter preserving explicit CSV names and index headers."""
    args = args or _CURRENT_ARGS.get()
    path = Path(path)
    if index:
        names = [name if name is not None else "" for name in table.index.names]
        table = table.reset_index()
        table.columns = names + list(table.columns[len(names):])
    units = {str(column): str(column).rsplit("(", 1)[1][:-1] for column in table.columns
             if "(" in str(column) and str(column).endswith(")")}
    spec = ArtifactSpec(path.name, path.name, tier, tier in {"core", "summary"}, "csv", units=units)
    with ArtifactWriter(path.parent, [spec], profile=getattr(args, "output_profile", "standard"),
                        detail_format=getattr(args, "detail_format", None), overwrite=True,
                        manifest_name=f"{path.name}.artifacts.json",
                        metadata={"command": getattr(args, "command", None)}) as writer:
        writer.write_table(path.name, table)
        return path if writer.enabled(path.name) else None


def write_workflow_tables(tables, *, args=None, summary=(), details=(), enabled_details=()):
    """Publish ``{Path: DataFrame}`` as a single declared artifact transaction."""
    if not tables:
        return []
    args = args or _CURRENT_ARGS.get()
    paths = [Path(path) for path in tables]
    directory = paths[0].parent
    if any(path.parent != directory for path in paths):
        raise ValueError("A workflow artifact group must share one output directory.")
    profile = str(getattr(args, "output_profile", "standard"))
    detail_format = getattr(args, "detail_format", None)
    specs = []
    for path, table in zip(paths, tables.values()):
        tier = "detail" if path.name in details else "summary" if path.name in summary else "core"
        columns = getattr(table, "columns", ())
        units = {str(column): str(column).rsplit("(", 1)[1][:-1] for column in columns
                 if "(" in str(column) and str(column).endswith(")")}
        format_name = (detail_format or ("csv" if profile == "legacy" else "parquet")) if tier == "detail" else "csv"
        filename = path.with_suffix("." + format_name).name
        specs.append(ArtifactSpec(path.name, filename, tier, tier != "detail" or path.name in enabled_details, format_name, units=units))
    with ArtifactWriter(directory, specs, profile=profile, detail_format=detail_format,
                        overwrite=True, metadata={"command": getattr(args, "command", None)}) as writer:
        for path, table in zip(paths, tables.values()):
            if isinstance(table, TableChunks):
                writer.write_chunks(path.name, table)
            else:
                writer.write_table(path.name, table)
        return [directory / writer.specs[path.name].filename for path in paths if writer.enabled(path.name)]
