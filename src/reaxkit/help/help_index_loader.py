"""
ReaxKit help index search utilities.

This module provides a lightweight search engine used by the
``reaxkit help`` command to map natural-language queries
(e.g. "electric field", "bond order", "restraint")
to relevant ReaxFF input and output files.

The search operates on curated YAML indices shipped with ReaxKit
and ranks matches based on keyword overlap and fuzzy similarity.

Typical use cases include:

- discovering which ReaxFF file controls a given concept
- exploring available variables in input/output files
- guiding users toward the correct handler or workflow
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from functools import lru_cache
import re

try:
    import yaml  # pyyaml
except Exception as e:  # pragma: no cover
    raise ImportError("PyYAML is required to use ReaxKit help index. Install with: pip install pyyaml") from e


# ----------------------------
# Data access (package files)
# ----------------------------

@lru_cache(maxsize=1)
def load_input_index() -> Dict[str, Any]:
    """
    Load the ReaxFF input-file help index.

    Returns
    -------
    dict
    Parsed contents of ``reaxff_input_files_contents.yaml``.
    """
    return _read_yaml_from_pkg_data("reaxff_input_files_contents.yaml")


@lru_cache(maxsize=1)
def load_output_index() -> Dict[str, Any]:
    """
    Load the ReaxFF output-file help index.

    Returns
    -------
    dict
        Parsed contents of ``reaxff_output_files_contents.yaml``.
    """
    return _read_yaml_from_pkg_data("reaxff_output_files_contents.yaml")

@dataclass(frozen=True)
class PreparedEntry:
    """
    Preprocessed help entry used for search.

    This class stores precomputed text fields and tokenized
    representations for a single ReaxFF file entry.

    Attributes
    ----------
    file : str
        Canonical ReaxFF file name.
    entry : dict
        Raw YAML entry.
    blobs : dict
        Concatenated searchable text fields.
    tokens : dict
        Tokenized versions of searchable fields.
    """
    file: str
    entry: Dict[str, Any]
    blobs: Dict[str, str]
    tokens: Dict[str, set[str]]

def _prepare_index(idx: Dict[str, Any]) -> Dict[str, PreparedEntry]:
    """
    Precompute searchable blobs and token sets for an index mapping.

    Works on
    --------
    ReaxKit help-index dictionaries with a ``files`` section

    Parameters
    ----------
    idx : dict[str, Any]
        Raw parsed YAML index.

    Returns
    -------
    dict[str, PreparedEntry]
        Mapping from file key to prepared entry used by ranking/search.

    Examples
    --------
    >>>
    """
    files = idx.get("files", {}) or {}
    prepared = {}

    for file_key, entry in files.items():
        if not isinstance(entry, dict):
            continue

        blobs = _entry_search_blobs(file_key, entry)
        tokens = {k: set(_tokens(v)) for k, v in blobs.items()}

        prepared[file_key] = PreparedEntry(
            file=file_key,
            entry=entry,
            blobs=blobs,
            tokens=tokens,
        )

    return prepared


@lru_cache(maxsize=1)
def load_prepared_input_index() -> Dict[str, PreparedEntry]:
    """
    Load and cache the prepared input help index.

    Works on
    --------
    Packaged ``reaxff_input_files_contents.yaml`` data

    Returns
    -------
    dict[str, PreparedEntry]
        Preprocessed input index for search.

    Examples
    --------
    >>>
    """
    return _prepare_index(_load_input_index())


@lru_cache(maxsize=1)
def load_prepared_output_index() -> Dict[str, PreparedEntry]:
    """
    Load and cache the prepared output help index.

    Works on
    --------
    Packaged ``reaxff_output_files_contents.yaml`` data

    Returns
    -------
    dict[str, PreparedEntry]
        Preprocessed output index for search.

    Examples
    --------
    >>>
    """
    return _prepare_index(_load_output_index())


def _read_yaml_from_pkg_data(filename: str) -> Dict[str, Any]:
    """
    Load a YAML file bundled inside the ``reaxkit.engine.reaxff.data`` package.

    Parameters
    ----------
    filename : str
        Name of the YAML file to load.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    try:
        from importlib import resources
        data_pkg = resources.files("reaxkit.engine.reaxff.data")
        path = data_pkg / filename
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        raise FileNotFoundError(
            f"Could not read '{filename}' from package 'reaxkit.engine.reaxff.data'. "
            f"Make sure it exists under src/reaxkit/engine/reaxff/data/ and is included in package data."
        ) from e

    obj = yaml.safe_load(text) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/dict in '{filename}'.")
    return obj


def _load_input_index() -> Dict[str, Any]:
    """
    Load raw input index YAML data.

    Works on
    --------
    Packaged ReaxKit help index files

    Returns
    -------
    dict[str, Any]
        Parsed input-index mapping.

    Examples
    --------
    >>>
    """
    return _read_yaml_from_pkg_data("reaxff_input_files_contents.yaml")


def _load_output_index() -> Dict[str, Any]:
    """
    Load raw output index YAML data.

    Works on
    --------
    Packaged ReaxKit help index files

    Returns
    -------
    dict[str, Any]
        Parsed output-index mapping.

    Examples
    --------
    >>>
    """
    return _read_yaml_from_pkg_data("reaxff_output_files_contents.yaml")


# ----------------------------
# Search / ranking
# ----------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    """
    Normalize a string for case-insensitive search matching.

    Works on
    --------
    Free-text query and index strings

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Lowercased, whitespace-normalized text.

    Examples
    --------
    >>>
    """
    s = s.lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> List[str]:
    """
    Tokenize a normalized string into alphanumeric search terms.

    Works on
    --------
    Free-text query and index strings

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    list[str]
        Token sequence for matching.

    Examples
    --------
    >>>
    """
    return _WORD_RE.findall(_norm(s))


def _fuzzy_ratio(a: str, b: str) -> float:
    """
    Compute fuzzy similarity between two strings.

    Returns
    -------
    float
        Similarity score in the range 0–100.
    """
    a = _norm(a)
    b = _norm(b)
    if not a or not b:
        return 0.0

    try:
        from rapidfuzz.fuzz import ratio
        return float(ratio(a, b))
    except Exception:
        import difflib
        return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()


def _as_list(v: Any) -> List[str]:
    """
    Normalize a YAML value into a list of strings.

    Works on
    --------
    YAML scalar/list fields in help index entries

    Parameters
    ----------
    v : Any
        Input field value.

    Returns
    -------
    list[str]
        Normalized list representation.

    Examples
    --------
    >>>
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    return [str(v)]


def _entry_search_blobs(file_key: str, entry: Dict[str, Any]) -> Dict[str, str]:
    """
    Build concatenated searchable text fields for a help entry.

    Parameters
    ----------
    file_key : str
        Canonical ReaxFF file name.
    entry : dict
        YAML entry describing the file.

    Returns
    -------
    dict
        Mapping of field name to searchable text.
    """
    aliases = _as_list(entry.get("aliases"))
    desc = str(entry.get("desc") or "")
    tags = _as_list(entry.get("tags"))
    core_vars = _as_list(entry.get("core_vars"))
    optional_vars = _as_list(entry.get("optional_vars"))
    derived_vars = _as_list(entry.get("derived_vars"))
    best_for = _as_list(entry.get("best_for"))
    # some YAMLs might use related_run or related_runs
    related = _as_list(entry.get("related_runs") or entry.get("related_run"))
    notes = _as_list(entry.get("notes"))
    examples = _as_list(entry.get("file_templates"))

    return {
        "names": " ".join([file_key] + aliases),
        "desc": desc,
        "tags": " ".join(tags),
        "core": " ".join(core_vars),
        "optional": " ".join(optional_vars),
        "derived": " ".join(derived_vars),
        "best_for": " ".join(best_for),
        "related": " ".join(related),
        "notes": " ".join(notes),
        "file_templates": " ".join(examples),
    }


@dataclass(frozen=True)
class HelpHit:
    """
    Ranked result returned by the help index search.

    Attributes
    ----------
    kind : str
        Either ``"input"`` or ``"output"``.
    file : str
        ReaxFF file name.
    score : float
        Relevance score.
    why : list of str
        Short explanations for why the file matched.
    entry : dict
        Raw YAML entry.
    """
    kind: str                 # "input" or "output"
    file: str                 # key in YAML
    score: float
    why: List[str]            # short reasons
    entry: Dict[str, Any]     # raw entry


def search_help_indices(
    query: str,
    *,
    top_k: int = 8,
    min_score: float = 35.0,
) -> List[HelpHit]:
    """
    Search ReaxKit help indices for relevant ReaxFF files.

    Parameters
    ----------
    query : str
        Natural-language search query.
    top_k : int, optional
        Maximum number of results to return.
    min_score : float, optional
        Minimum relevance score for a result to be included.

    Returns
    -------
    list of HelpHit
        Ranked search results across input and output files.

    Examples
    --------
    >>> hits = search_help_indices("electric field")
    >>> hits[0].file
    'eregime.in'
    """
    q = _norm(query)
    q_toks = set(_tokens(query))

    in_idx = load_prepared_input_index()
    out_idx = load_prepared_output_index()

    hits: List[HelpHit] = []
    hits.extend(_search_one_index("input", in_idx, q, q_toks))
    hits.extend(_search_one_index("output", out_idx, q, q_toks))

    # overall top_k across both
    hits.sort(key=lambda h: h.score, reverse=True)
    hits = [h for h in hits if h.score >= min_score]
    return hits[:top_k]


def _search_one_index(
    kind: str,
    idx: Dict[str, PreparedEntry],
    q: str,
    q_toks: set[str],
) -> List[HelpHit]:
    """
    Search a single help index (input or output).

    Parameters
    ----------
    kind : str
        Either ``"input"`` or ``"output"``.
    idx : dict
        Parsed YAML index.
    q : str
        Normalized query string.
    q_toks : set of str
        Tokenized query terms.

    Returns
    -------
    list of HelpHit
        Ranked matches from the given index.
    """
    files = idx.get("files", {}) or {}
    if not isinstance(files, dict):
        return []

    res: List[HelpHit] = []

    for file_key, prep in idx.items():
        fast_score = 0.0

        entry = prep.entry
        blobs = prep.blobs
        tokens = prep.tokens

        score = 0.0
        why: List[str] = []

        # 1) deterministic boosts
        names_norm = _norm(blobs["names"])
        if q and q in names_norm.split():
            score += 120.0
            why.append("exact file/alias match")

        # token overlaps (fast and robust)
        def _overlap(field_name: str, weight: float) -> None:
            nonlocal fast_score
            ov = q_toks & tokens[field_name]
            if ov:
                fast_score += weight + 4.0 * len(ov)

        _overlap("tags", 30.0)
        _overlap("best_for", 22.0)
        _overlap("core", 18.0)
        _overlap("optional", 8.0)   # smaller boost; optional-only matches should rank lower
        _overlap("derived", 18.0)
        _overlap("related", 14.0)
        _overlap("desc", 10.0)

        if fast_score < 10.0:
            continue

        # 2) fuzzy matching over key fields (weighted)
        score = fast_score

        score += 0.35 * _fuzzy_ratio(q, blobs["tags"])
        score += 0.30 * _fuzzy_ratio(q, blobs["names"])
        score += 0.22 * _fuzzy_ratio(q, blobs["core"])
        score += 0.10 * _fuzzy_ratio(q, blobs["desc"])
        score += 0.06 * _fuzzy_ratio(q, blobs["optional"])
        score += 0.12 * _fuzzy_ratio(q, blobs["derived"])
        score += 0.04 * _fuzzy_ratio(q, blobs["notes"])

        # small preference: if it matches best_for strongly, nudge up
        if _fuzzy_ratio(q, blobs["best_for"]) >= 80:
            score += 10.0
            why.append("strong best_for match")

        # add hit if not totally irrelevant (threshold handled later)
        res.append(HelpHit(kind=kind, file=str(file_key), score=score, why=why, entry=entry))

    return res


def _group_hits(hits: Iterable[HelpHit]) -> Tuple[List[HelpHit], List[HelpHit]]:
    """
    Split search results into input and output file groups.

    Parameters
    ----------
    hits : iterable of HelpHit
        Search results.

    Returns
    -------
    tuple of list of HelpHit
        ``(input_hits, output_hits)`` sorted by score.
    """

    ins = [h for h in hits if h.kind == "input"]
    outs = [h for h in hits if h.kind == "output"]
    ins.sort(key=lambda h: h.score, reverse=True)
    outs.sort(key=lambda h: h.score, reverse=True)
    return ins, outs


def _format_hits(
    hits: List[HelpHit],
    *,
    show_why: bool = True,
    show_examples: bool = False,
    show_tags: bool = False,
    show_core_vars: bool = False,
    show_optional_vars: bool = False,
    show_derived_vars: bool = False,
    show_notes: bool = False,
    engine: Optional[str] = None,
) -> str:
    """
    Format help search results for CLI display.

    Parameters
    ----------
    hits : list of HelpHit
        Search results.
    show_why, show_examples, show_tags, show_core_vars, show_optional_vars, show_derived_vars, show_notes : bool
        Flags controlling which metadata fields are displayed.

    Returns
    -------
    str
        Human-readable formatted output.
    """
    in_hits, out_hits = _group_hits(hits)
    engine_map = _resolve_engine_loader_map(engine) if engine else {}

    def _fmt_one(h: HelpHit) -> str:
        e = h.entry
        kind_flag = f" --{h.kind}"
        related = e.get("related_runs") or e.get("related_run") or []
        related_list = related if isinstance(related, list) else [related]
        related_str = f"  related_run: [{', '.join(related_list)}]" if related_list else ""
        lines = [f"• {h.file}{kind_flag}   (score={h.score:.1f}){related_str}"]

        desc = e.get("desc")
        if desc:
            lines.append(f"  {desc}")
        if show_why and h.why:
            lines.append(f"  why: {', '.join(h.why[:3])}")
        if show_examples:
            ex = e.get("file_templates") or []
            if ex:
                lines.append(f"  ex:  {ex[0]}")
        if show_tags:
            tags = e.get("tags") or []
            if tags:
                lines.append(f"  tags: {tags}")
        if engine:
            rows = _engine_targets_for_file(engine_map, h.file)
            if rows:
                lines.append(f"  dataclasses ({engine}):")
                for row in rows:
                    lines.append(f"    - {row['dataclass']} via {row['loader']} ({row['role']})")

        if show_core_vars:
            xs = e.get("core_vars") or []
            if xs:
                lines.append("  core_vars:")
                lines.extend(f"    - {v}" for v in xs)

        if show_optional_vars:
            xs = e.get("optional_vars") or []
            if xs:
                lines.append("  optional_vars:")
                lines.extend(f"    - {v}" for v in xs)

        if show_derived_vars:
            xs = e.get("derived_vars") or []
            if xs:
                lines.append("  derived_vars:")
                lines.extend(f"    - {v}" for v in xs)

        if show_notes:
            xs = e.get("notes") or []
            if xs:
                lines.append("  notes:")
                lines.extend(f"    - {v}" for v in xs)

        return "\n".join(lines)

    parts: List[str] = []
    if in_hits:
        parts.append("INPUT FILES")
        parts.extend(_fmt_one(h) for h in in_hits)
    if in_hits and out_hits:
        parts.append("-------------")
    if out_hits:
        if engine:
            parts.append(f"OUTPUT FILES (engine: {engine})")
        else:
            parts.append("OUTPUT FILES")
        parts.extend(_fmt_one(h) for h in out_hits)

    if not parts:
        return "❌ No matches."

    parts.append("")
    parts.append(
        "Tip: use `reaxkit <filename> -h` or `reaxkit <filename> <task> -h` "
        "to see a more comprehensive description of available options, "
        "file_templates, and usage details.\n"
    )

    return "\n".join(parts)


# ----------------------------
# Command / capability search
# ----------------------------

@dataclass(frozen=True)
class CommandHit:
    """Ranked command/capability match for `reaxkit help` queries."""

    command: str
    kind: str
    score: float
    help_text: str
    aliases: Tuple[str, ...]
    module_path: str
    related_commands: Tuple[str, ...]


def _read_yaml_from_package(package: str, filename: str) -> Dict[str, Any]:
    """Read and parse a packaged YAML file."""
    try:
        from importlib import resources
        text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
    except Exception as e:
        raise FileNotFoundError(
            f"Could not read '{filename}' from package '{package}'."
        ) from e

    obj = yaml.safe_load(text) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/dict in '{filename}'.")
    return obj


@lru_cache(maxsize=1)
def load_command_metadata() -> Dict[str, Any]:
    """Load command metadata used for capability discovery."""
    return _read_yaml_from_package("reaxkit", "data/command_metadata.yaml")


@lru_cache(maxsize=1)
def load_engine_data_maps() -> Dict[str, Any]:
    """Load mapping of engine name -> loader/dataclass map source."""
    return _read_yaml_from_package("reaxkit.help.data", "engine_data_maps.yaml")


def _resolve_engine_loader_map(engine: str) -> Dict[str, Any]:
    """Load the configured loader/dataclass map for an engine."""
    engine_key = _norm(engine)
    maps = load_engine_data_maps().get("engines") or {}
    for name, meta in maps.items():
        if _norm(str(name)) != engine_key:
            continue
        package = str((meta or {}).get("package") or "").strip()
        filename = str((meta or {}).get("loader_map_file") or "").strip()
        if not package or not filename:
            return {}
        return _read_yaml_from_package(package, filename)
    return {}


def _resolve_module_path(command: str, kind: str) -> str:
    """Resolve module path for a command via routing registries."""
    try:
        from reaxkit.core.analysis_cli_routing_registry import get_registered_analysis_commands
        from reaxkit.core.generator_cli_routing_registry import get_registered_generators
        from reaxkit.core.workflow_cli_routing_registry import get_registered_workflows
    except Exception:
        return ""

    if kind == "analysis":
        spec = get_registered_analysis_commands().get(command)
        return str(spec.module_path) if spec is not None else ""
    if kind == "generator":
        spec = get_registered_generators().get(command)
        return str(spec.module_path) if spec is not None else ""
    if kind == "workflow":
        spec = get_registered_workflows().get(command)
        return str(spec.module_path) if spec is not None else ""
    return ""


def _related_commands_for_module(module_path: str, kind: str) -> Tuple[str, ...]:
    """List peer commands that share the same module path."""
    if not module_path:
        return tuple()
    try:
        from reaxkit.core.analysis_cli_routing_registry import get_registered_analysis_commands
        from reaxkit.core.generator_cli_routing_registry import get_registered_generators
        from reaxkit.core.workflow_cli_routing_registry import get_registered_workflows
    except Exception:
        return tuple()

    related: List[str] = []
    if kind == "analysis":
        for name, spec in get_registered_analysis_commands().items():
            if str(spec.module_path) == module_path:
                related.append(str(name))
    elif kind == "generator":
        for name, spec in get_registered_generators().items():
            if str(spec.module_path) == module_path:
                related.append(str(name))
    elif kind == "workflow":
        for name, spec in get_registered_workflows().items():
            if str(spec.module_path) == module_path:
                related.append(str(name))
    return tuple(sorted(set(related)))


def search_help_commands(
    query: str,
    *,
    top_k: int = 5,
    min_score: float = 35.0,
) -> List[CommandHit]:
    """Search command metadata from natural-language queries."""
    q = _norm(query)
    q_toks = set(_tokens(query))
    raw = load_command_metadata().get("commands") or {}
    hits: List[CommandHit] = []

    for command_name, meta in raw.items():
        aliases = [str(a) for a in _as_list((meta or {}).get("aliases"))]
        help_text = str((meta or {}).get("help") or "")
        kind = str((meta or {}).get("kind") or "analysis").strip().lower()

        names_blob = " ".join([str(command_name)] + aliases)
        names_tokens = set(_tokens(names_blob))
        desc_tokens = set(_tokens(help_text))

        score = 0.0
        overlap_names = q_toks & names_tokens
        overlap_desc = q_toks & desc_tokens
        if overlap_names:
            score += 45.0 + 6.0 * len(overlap_names)
        if overlap_desc:
            score += 12.0 + 3.0 * len(overlap_desc)

        score += 0.40 * _fuzzy_ratio(q, names_blob)
        score += 0.15 * _fuzzy_ratio(q, help_text)

        if q and q in _norm(str(command_name)):
            score += 45.0
        elif q and q in _norm(names_blob):
            score += 20.0

        if score >= min_score:
            module_path = _resolve_module_path(str(command_name), kind)
            hits.append(
                CommandHit(
                    command=str(command_name),
                    kind=kind,
                    score=score,
                    help_text=help_text,
                    aliases=tuple(aliases),
                    module_path=module_path,
                    related_commands=_related_commands_for_module(module_path, kind),
                )
            )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def _engine_sources_for_dataclass(engine_loader_map: Dict[str, Any], dataclass_name: str) -> List[Dict[str, Any]]:
    """Find loader entries that produce the given dataclass."""
    loaders = (engine_loader_map or {}).get("loaders") or {}
    out: List[Dict[str, Any]] = []
    for loader_name, meta in loaders.items():
        if str((meta or {}).get("dataclass") or "") != dataclass_name:
            continue
        out.append(
            {
                "loader": str(loader_name),
                "primary_files": [str(x) for x in _as_list((meta or {}).get("primary_files"))],
                "supplemental_files": [str(x) for x in _as_list((meta or {}).get("supplemental_files"))],
                "fallback_files": [str(x) for x in _as_list((meta or {}).get("fallback_files"))],
            }
        )
    return out


def _format_command_hits_legacy(hits: List[CommandHit]) -> str:
    """Format command/capability matches for CLI output."""
    if not hits:
        return ""
    return _format_command_hits(hits)

    parts: List[str] = ["COMMAND MATCHES"]

    for hit in hits:
        parts.append(f"• {hit.intent} (score={hit.score:.1f})")
        if hit.description:
            parts.append(f"  {hit.description}")

        parts.append(f"  dataclasses: {list(hit.dataclasses)}")

        analysis_cmds = _analysis_commands_for_dataclasses(hit.dataclasses)
        if registered:
            analysis_cmds = [c for c in analysis_cmds if c in registered]
            workflows = [w for w in hit.workflows if w in registered]
            generators = [g for g in hit.generators if g in registered]
        else:
            workflows = list(hit.workflows)
            generators = list(hit.generators)

        parts.append(f"  analysis tasks: {analysis_cmds or []}")
        parts.append(f"  workflows: {workflows or []}")
        parts.append(f"  generators: {generators or []}")

        if engine:
            parts.append(f"  engine sources ({engine}):")
            any_row = False
            for dataclass_name in hit.dataclasses:
                rows = _engine_sources_for_dataclass(engine_map, dataclass_name)
                if not rows:
                    continue
                any_row = True
                for row in rows:
                    parts.append(
                        "    - "
                        f"{dataclass_name} via {row['loader']} "
                        f"(primary={row['primary_files']}, "
                        f"supplemental={row['supplemental_files']}, "
                        f"fallback={row['fallback_files']})"
                    )
            if not any_row:
                parts.append("    - no engine loader mapping found for this intent/dataclass set")

    return "\n".join(parts)


def _engine_targets_for_file(engine_loader_map: Dict[str, Any], file_name: str) -> List[Dict[str, str]]:
    """Find loader entries that consume the given file."""
    file_key = _norm(file_name)
    if not file_key:
        return []
    loaders = (engine_loader_map or {}).get("loaders") or {}
    out: List[Dict[str, str]] = []
    for loader_name, meta in loaders.items():
        dataclass_name = str((meta or {}).get("dataclass") or "").strip()
        if not dataclass_name:
            continue
        for role in ("primary_files", "supplemental_files", "fallback_files"):
            files = [str(x) for x in _as_list((meta or {}).get(role))]
            if any(_norm(x) == file_key for x in files):
                out.append(
                    {
                        "loader": str(loader_name),
                        "dataclass": dataclass_name,
                        "role": role.replace("_files", ""),
                    }
                )
                break
    return out


def _format_command_hits(hits: List[CommandHit]) -> str:
    """Format command/capability matches for CLI output."""
    if not hits:
        return ""

    parts: List[str] = ["COMMAND MATCHES"]
    for hit in hits:
        parts.append(f"- {hit.command} --{hit.kind} (score={hit.score:.1f})")
        if hit.help_text:
            parts.append(f"  {hit.help_text}")
        if hit.aliases:
            parts.append(f"  aliases: {list(hit.aliases)}")
        if hit.module_path:
            parts.append(f"  workflow module: {hit.module_path}")
        if hit.related_commands:
            parts.append(f"  related commands: {list(hit.related_commands)}")
    return "\n".join(parts)
