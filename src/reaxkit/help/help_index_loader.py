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
from pathlib import Path
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
        tokens = {k: _token_set(v) for k, v in blobs.items()}

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


def _token_set(s: str) -> set[str]:
    """
    Token set with lightweight singular normalization.

    This keeps original tokens and adds common singular variants so
    queries like "coordinates" can match entries using "coordinate".
    """
    raw = _tokens(s)
    out: set[str] = set()
    for tok in raw:
        if not tok:
            continue
        out.add(tok)

        # plural -> singular variants (conservative rules)
        if len(tok) > 3 and tok.endswith("ies"):
            out.add(tok[:-3] + "y")
        if len(tok) > 3 and tok.endswith("es"):
            stem = tok[:-2]
            if stem.endswith(("s", "x", "z", "ch", "sh")):
                out.add(stem)
        if len(tok) > 2 and tok.endswith("s") and not tok.endswith("ss"):
            out.add(tok[:-1])
    return out


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
    desc = str(entry.get("description") or entry.get("desc") or "")
    mandatory = str(entry.get("mandatory") or "")
    tags = _as_list(entry.get("tags"))
    core_vars = _as_list(entry.get("core_vars"))
    optional_vars = _as_list(entry.get("optional_vars"))
    derived_vars = _as_list(entry.get("derived_vars"))
    best_for = _as_list(entry.get("best_for"))
    # some YAMLs might use related_run or related_runs
    related = _as_list(entry.get("related_runs") or entry.get("related_run"))
    notes = _as_list(entry.get("notes"))
    examples = _as_list(entry.get("help_search_examples") or entry.get("file_templates"))

    return {
        "names": " ".join([file_key] + aliases),
        "desc": desc,
        "mandatory": mandatory,
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
    q_toks = _token_set(_expand_query_text(query) or query)

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
        _overlap("mandatory", 10.0)

        if fast_score < 10.0:
            continue

        # 2) fuzzy matching over key fields (weighted)
        score = fast_score

        score += 0.35 * _fuzzy_ratio(q, blobs["tags"])
        score += 0.30 * _fuzzy_ratio(q, blobs["names"])
        score += 0.22 * _fuzzy_ratio(q, blobs["core"])
        score += 0.10 * _fuzzy_ratio(q, blobs["desc"])
        score += 0.08 * _fuzzy_ratio(q, blobs["mandatory"])
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
        mandatory = e.get("mandatory")
        if mandatory:
            lines.append(f"  mandatory: {mandatory}")
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
    examples: Tuple[str, ...]
    directory: str
    module_path: str
    related_commands: Tuple[str, ...]


def _read_yaml_from_package(package: str, filename: str) -> Dict[str, Any]:
    """Read and parse a packaged YAML file."""
    try:
        from importlib import resources
        text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
    except Exception as e:
        # Fallback for local/dev environments where importing package resources
        # can fail due optional runtime deps in package __init__ side-effects.
        from pathlib import Path

        pkg = str(package or "").strip()
        pkg_parts = pkg.split(".")
        if pkg_parts and pkg_parts[0] == "reaxkit":
            rel_parts = pkg_parts[1:]
            local_root = Path(__file__).resolve().parents[1]  # .../reaxkit
            candidate = local_root.joinpath(*rel_parts, filename)
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
            else:
                raise FileNotFoundError(
                    f"Could not read '{filename}' from package '{package}' "
                    f"or local fallback path '{candidate}'."
                ) from e
        else:
            raise FileNotFoundError(
                f"Could not read '{filename}' from package '{package}'."
            ) from e

    obj = yaml.safe_load(text) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/dict in '{filename}'.")
    return obj


def _read_map_from_package(package: str, filename: str) -> Dict[str, Any]:
    """
    Read and parse a packaged structured map file.

    Supports:
    - .yaml/.yml mapping files
    - .py module files exporting one of: REAXFF_MAP, MAP, DATA
    """
    lower = str(filename).lower()
    if lower.endswith(".yaml") or lower.endswith(".yml"):
        return _read_yaml_from_package(package, filename)

    if lower.endswith(".py"):
        module_name = str(filename).rsplit("/", 1)[-1].rsplit("\\", 1)[-1][:-3]
        if not module_name:
            raise ValueError(f"Invalid module filename: {filename!r}")
        try:
            from importlib import import_module

            module = import_module(f"{package}.{module_name}")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not import map module '{module_name}' from package '{package}'."
            ) from e

        for attr in ("REAXFF_MAP", "MAP", "DATA"):
            obj = getattr(module, attr, None)
            if isinstance(obj, dict):
                return obj
        raise ValueError(
            f"Module '{package}.{module_name}' does not export a dict via REAXFF_MAP, MAP, or DATA."
        )

    raise ValueError(f"Unsupported map file type for '{filename}'. Expected .yaml/.yml/.py")


@lru_cache(maxsize=1)
def load_engine_data_maps() -> Dict[str, Any]:
    """Load mapping of engine name -> loader/dataclass map source from help information sources."""
    out: Dict[str, Any] = {}
    src = load_help_information_sources()
    mapping_rows = (((src.get("source_files_by_layer") or {}).get("file_level") or {}).get("mapping") or [])
    if not isinstance(mapping_rows, list):
        return out
    for row in mapping_rows:
        if not isinstance(row, dict):
            continue
        for engine_name, entries in row.items():
            if not isinstance(entries, list):
                continue
            for item in entries:
                if not isinstance(item, dict):
                    continue
                rel_path = str(item.get("file") or "").strip()
                if not rel_path:
                    continue
                filename = Path(rel_path).name
                engine_key = _norm(str(engine_name))
                out[engine_key] = {
                    "package": _package_from_reaxkit_rel_path(rel_path),
                    "loader_map_file": filename,
                }
    return out


@lru_cache(maxsize=1)
def load_help_search_index() -> Dict[str, Any]:
    """Load optional query-expansion and command enrichment metadata for help search."""
    try:
        return _read_yaml_from_package("reaxkit.help.data", "help_search_index.yaml")
    except Exception:
        return {}


@lru_cache(maxsize=1)
def load_help_information_sources() -> Dict[str, Any]:
    """Load layer/source wiring metadata used by `reaxkit help`."""
    return _read_yaml_from_package("reaxkit.help.data", "help_information_sources.yaml")


@lru_cache(maxsize=1)
def load_help_intents() -> Dict[str, Any]:
    """Load canonical intents and wording aliases used for query normalization."""
    return _read_yaml_from_package("reaxkit.help.data", "help_intents.yaml")


def _package_from_reaxkit_rel_path(rel_path: str) -> str:
    """Convert a reaxkit-relative source path to import package form."""
    text = str(rel_path or "").strip().replace("\\", "/")
    if not text.endswith(".yaml") and not text.endswith(".yml") and not text.endswith(".py"):
        return ""
    parts = [p for p in text.split("/") if p]
    if not parts or parts[0] != "reaxkit" or len(parts) < 2:
        return ""
    # drop filename
    return ".".join(parts[:-1])


def _repo_src_root() -> Path:
    """Return repository `src` root inferred from this module location."""
    # .../src/reaxkit/help/help_index_loader.py -> .../src
    return Path(__file__).resolve().parents[2]


def _read_yaml_from_reaxkit_rel_path(rel_path: str) -> Dict[str, Any]:
    """Read YAML from a `reaxkit/...` path relative to repository `src` root."""
    path = _repo_src_root() / str(rel_path).replace("\\", "/")
    text = path.read_text(encoding="utf-8")
    obj = yaml.safe_load(text) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/dict in '{rel_path}'.")
    return obj


def _unique_strs(values: Iterable[Any]) -> List[str]:
    """Return unique, non-empty strings while preserving first-seen order."""
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = _norm(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _word_alias_expand(query: str) -> str:
    """
    Expand query by applying `wording_alias` aliases token-by-token.

    Expected YAML shape:
      wording_alias:
        heat of formation:
          - heatfo
          - hof
    """
    q_raw = str(query or "").strip()
    if not q_raw:
        return ""

    data = load_help_intents()
    wording_alias = (data.get("wording_alias") or {})
    if not isinstance(wording_alias, dict):
        return q_raw

    # Build reverse alias map: alias token -> canonical phrase
    reverse_map: Dict[str, str] = {}
    for canonical, aliases in wording_alias.items():
        canon_text = str(canonical or "").strip()
        if not canon_text:
            continue
        canon_norm = _norm(canon_text)
        reverse_map[canon_norm] = canon_text
        for alias in _as_list(aliases):
            alias_norm = _norm(alias)
            if alias_norm:
                reverse_map[alias_norm] = canon_text

    expanded: List[str] = []
    for tok in _tokens(q_raw):
        expanded.append(tok)
        mapped = reverse_map.get(_norm(tok))
        if mapped:
            expanded.append(mapped)

    if not expanded:
        return q_raw
    return " ".join(_unique_strs(expanded))


@lru_cache(maxsize=1)
def _load_spacy_nlp():
    """
    Lazily load a spaCy pipeline for lemmatization.

    Falls back to ``None`` when spaCy/model/lookups are unavailable.
    """
    try:
        import spacy  # type: ignore
    except Exception:
        return None

    # Preferred: installed English pipeline
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except Exception:
        pass

    # Fallback: blank pipeline with lookup lemmatizer (requires lookups package).
    try:
        nlp = spacy.blank("en")
        if "lemmatizer" not in nlp.pipe_names:
            nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        nlp.initialize()
        return nlp
    except Exception:
        return None


def _lemmatize_with_spacy(text: str) -> str:
    """Lemmatize a text with spaCy; fallback to normalized text when unavailable."""
    nlp = _load_spacy_nlp()
    if nlp is None:
        return _norm(text)
    try:
        doc = nlp(text)
    except Exception:
        return _norm(text)

    lemmas: List[str] = []
    for tok in doc:
        lemma = str(getattr(tok, "lemma_", "") or "").strip().lower()
        if not lemma or lemma == "-pron-":
            lemma = str(tok.text).strip().lower()
        if lemma:
            lemmas.append(lemma)
    return " ".join(lemmas) if lemmas else _norm(text)


def _expand_query_text(query: str) -> str:
    """
    Query preprocessing pipeline for help search:
    1) normalize + lemmatize
    2) word-level alias expansion over both raw and lemmatized tokens
    3) merge all variants for robust fuzzy matching
    """
    raw = _norm(str(query or ""))
    if not raw:
        return ""
    lemma = _lemmatize_with_spacy(raw)
    aliased_raw = _word_alias_expand(raw)
    aliased_lemma = _word_alias_expand(lemma)
    return " ".join(_unique_strs([raw, lemma, aliased_raw, aliased_lemma]))


def _token_set_fuzzy_ratio(a: str, b: str) -> float:
    """Token-set fuzzy similarity in the range 0-100 with a pure-Python fallback."""
    a = _norm(a)
    b = _norm(b)
    if not a or not b:
        return 0.0

    try:
        from rapidfuzz.fuzz import token_set_ratio
        return float(token_set_ratio(a, b))
    except Exception:
        ta = set(_tokens(a))
        tb = set(_tokens(b))
        if not ta or not tb:
            return 0.0
        return 100.0 * (len(ta & tb) / float(len(ta | tb)))


def _infer_command_kind(command_name: str) -> str:
    """Infer command kind from routing registries."""
    try:
        from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
        from reaxkit.core.registry.generator_cli_routing_registry import get_registered_generators
        from reaxkit.core.registry.workflow_cli_routing_registry import get_registered_workflows
    except Exception:
        return ""

    if command_name in get_registered_analysis_commands():
        return "analysis"
    if command_name in get_registered_generators():
        return "generator"
    if command_name in get_registered_workflows():
        return "workflow"
    return ""


def _infer_directory_for_command(command_name: str, kind: str) -> str:
    """
    Infer a directory pointer describing where authoritative command details live.

    Formats:
    - analysis_task_dataclass_map.yaml:<task_key>
    - workflow_dataclass_map.yaml:command_to_workflow_module:<command_name>
    """
    task_map = (load_analysis_task_dataclass_map().get("tasks") or {})
    for key in (command_name, command_name.replace("-", "_")):
        if key in task_map:
            return f"analysis_task_dataclass_map.yaml:{key}"

    if kind in {"generator", "workflow", "analysis"}:
        return f"workflow_dataclass_map.yaml:command_to_workflow_module:{command_name}"
    return f"workflow_dataclass_map.yaml:command_to_workflow_module:{command_name}"


@lru_cache(maxsize=1)
def load_help_command_index() -> Dict[str, Dict[str, Any]]:
    """
    Build command index used by help search from ``help_search_index.yaml``.
    """
    ext = (load_help_search_index().get("commands") or {})
    if not isinstance(ext, dict):
        ext = {}

    all_names = sorted({str(x) for x in list(ext.keys()) if str(x).strip()})
    out: Dict[str, Dict[str, Any]] = {}

    for command_name in all_names:
        ext_meta = ext.get(command_name) if isinstance(ext.get(command_name), dict) else {}

        kind = str(
            ext_meta.get("kind")
            or _infer_command_kind(command_name)
            or "analysis"
        ).strip().lower()
        desc = str(
            ext_meta.get("desc")
            or ext_meta.get("description")
            or ""
        ).strip()
        aliases = _unique_strs(_as_list(ext_meta.get("aliases")))
        examples = _unique_strs(_as_list(ext_meta.get("examples")))
        directory = str(ext_meta.get("directory") or "").strip()
        if not directory:
            directory = _infer_directory_for_command(command_name, kind)

        out[command_name] = {
            "kind": kind,
            "desc": desc,
            "aliases": aliases,
            "examples": examples,
            "directory": directory,
        }

    return out


def _resolve_engine_loader_map(engine: str) -> Dict[str, Any]:
    """Load the configured loader/dataclass map for an engine."""
    engine_key = _norm(engine)
    maps = load_engine_data_maps() or {}
    for name, meta in maps.items():
        if _norm(str(name)) != engine_key:
            continue
        package = str((meta or {}).get("package") or "").strip()
        filename = str((meta or {}).get("loader_map_file") or "").strip()
        if not package or not filename:
            return {}
        return _read_map_from_package(package, filename)
    return {}


def _resolve_module_path(command: str, kind: str) -> str:
    """Resolve module path for a command via routing registries."""
    try:
        from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
        from reaxkit.core.registry.generator_cli_routing_registry import get_registered_generators
        from reaxkit.core.registry.workflow_cli_routing_registry import get_registered_workflows
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
        from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
        from reaxkit.core.registry.generator_cli_routing_registry import get_registered_generators
        from reaxkit.core.registry.workflow_cli_routing_registry import get_registered_workflows
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
    """Search command index from natural-language queries."""
    q_raw = str(query or "")
    q_proc = _expand_query_text(q_raw) or q_raw
    q = _norm(q_proc)
    q_toks = _token_set(q_proc)
    raw = load_help_command_index()
    hits: List[CommandHit] = []

    for command_name, meta in raw.items():
        aliases = _unique_strs(_as_list(meta.get("aliases")))
        examples = _unique_strs(_as_list(meta.get("examples")))
        help_text = str(meta.get("desc") or "")
        kind = str(meta.get("kind") or "analysis").strip().lower()
        directory = str(meta.get("directory") or "").strip()

        alias_blob = " ".join(aliases) if aliases else str(command_name)
        examples_blob = " ".join(examples)
        combined_blob = " ".join([alias_blob, help_text, examples_blob])
        alias_tokens = _token_set(alias_blob)
        example_tokens = _token_set(examples_blob)
        desc_tokens = _token_set(help_text)

        score = 0.0
        overlap_aliases = q_toks & alias_tokens
        overlap_examples = q_toks & example_tokens
        overlap_desc = q_toks & desc_tokens
        if overlap_aliases:
            score += 45.0 + 6.0 * len(overlap_aliases)
        if overlap_examples:
            score += 24.0 + 3.0 * len(overlap_examples)
        if overlap_desc:
            score += 18.0 + 3.0 * len(overlap_desc)

        score += 0.34 * _token_set_fuzzy_ratio(q, alias_blob)
        score += 0.30 * _token_set_fuzzy_ratio(q, examples_blob)
        score += 0.24 * _token_set_fuzzy_ratio(q, help_text)
        score += 0.18 * _token_set_fuzzy_ratio(q, combined_blob)

        if q and q in _norm(alias_blob):
            score += 30.0

        if score >= min_score:
            module_path = _resolve_module_path(str(command_name), kind)
            hits.append(
                CommandHit(
                    command=str(command_name),
                    kind=kind,
                    score=score,
                    help_text=help_text,
                    aliases=tuple(aliases),
                    examples=tuple(examples),
                    directory=directory,
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
        parts.append(f"  - more help: reaxkit {hit.command} -h")
    return "\n".join(parts)


# ----------------------------
# Integrated relationship search
# ----------------------------

@dataclass(frozen=True)
class GeneratorCapabilityHit:
    """Ranked generator capability match."""

    capability: str
    score: float
    description: str
    implementation_module: str
    implementation_symbol: str
    related_commands: Tuple[str, ...]
    related_workflow_module: str
    tags: Tuple[str, ...]
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class LoaderMapHit:
    """Ranked loader/writer map match."""

    entry_type: str  # "loader" | "writer"
    name: str
    score: float
    dataclass: str
    description: str
    generator: str
    primary_files: Tuple[str, ...]
    supplemental_files: Tuple[str, ...]
    fallback_files: Tuple[str, ...]
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class AnalysisTaskMapHit:
    """Ranked analysis-task map match."""

    task: str
    score: float
    dataclass: str
    module: str
    workflow_links: Tuple[Tuple[str, str, str, str], ...]  # (command, kind, module_path, help_text)


def _module_path_to_file(module_path: str) -> str:
    """Convert dotted module path to repository-like file path."""
    mod = str(module_path or "").strip()
    if not mod:
        return ""
    return f"{mod.replace('.', '/')}.py"


@lru_cache(maxsize=1)
def load_analysis_task_dataclass_map() -> Dict[str, Any]:
    """Load analysis task -> dataclass map used by analyzers."""
    return _read_yaml_from_package("reaxkit.analysis.data", "analysis_task_dataclass_map.yaml")


def _resolve_engine_generator_capability_map(engine: str) -> Dict[str, Any]:
    """Load engine-specific generator capability map, if configured."""
    engine_key = _norm(engine)
    maps = load_engine_data_maps() or {}
    for name, meta in maps.items():
        if _norm(str(name)) != engine_key:
            continue
        package = str((meta or {}).get("package") or "").strip()
        filename = str((meta or {}).get("generator_capability_file") or "").strip()
        if not package or not filename:
            return {}
        return _read_yaml_from_package(package, filename)
    return {}


def _command_entry(command_name: str) -> Tuple[str, str, str]:
    """
    Return (kind, help_text, module_path) for a command.

    Falls back to registries when command metadata does not define a command.
    """
    meta = load_help_command_index().get(command_name) or {}
    kind = str(meta.get("kind") or "").strip().lower()
    help_text = str(meta.get("desc") or "")

    if not kind:
        try:
            from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
            from reaxkit.core.registry.generator_cli_routing_registry import get_registered_generators
            from reaxkit.core.registry.workflow_cli_routing_registry import get_registered_workflows
            if command_name in get_registered_analysis_commands():
                kind = "analysis"
            elif command_name in get_registered_generators():
                kind = "generator"
            elif command_name in get_registered_workflows():
                kind = "workflow"
        except Exception:
            kind = ""

    module_path = _resolve_module_path(command_name, kind) if kind else ""
    return kind, help_text, module_path


def search_generator_capabilities(
    query: str,
    *,
    engine: str = "reaxff",
    top_k: int = 5,
    min_score: float = 35.0,
) -> List[GeneratorCapabilityHit]:
    """Search engine generator-capability index."""
    q = _norm(query)
    q_toks = _token_set(_expand_query_text(query) or query)
    raw = (_resolve_engine_generator_capability_map(engine).get("generators") or {})
    hits: List[GeneratorCapabilityHit] = []

    for capability, meta in raw.items():
        desc = str((meta or {}).get("description") or "")
        tags = tuple(str(x) for x in _as_list((meta or {}).get("tags")))
        impl_module = str((meta or {}).get("implementation_module") or "")
        impl_symbol = str((meta or {}).get("implementation_symbol") or "")
        related_commands = tuple(str(x) for x in _as_list((meta or {}).get("related_commands")))
        related_workflow = str((meta or {}).get("related_workflow_module") or "")
        notes = tuple(str(x) for x in _as_list((meta or {}).get("notes")))

        names_blob = " ".join(
            [
                str(capability),
                impl_symbol,
                impl_module,
                related_workflow,
                " ".join(related_commands),
            ]
        )
        text_blob = " ".join([desc, " ".join(tags), " ".join(notes)])
        names_tokens = _token_set(names_blob)
        text_tokens = _token_set(text_blob)

        score = 0.0
        overlap_names = q_toks & names_tokens
        overlap_text = q_toks & text_tokens
        if overlap_names:
            score += 45.0 + 6.0 * len(overlap_names)
        if overlap_text:
            score += 14.0 + 3.0 * len(overlap_text)

        score += 0.40 * _fuzzy_ratio(q, names_blob)
        score += 0.18 * _fuzzy_ratio(q, text_blob)

        if q and q in _norm(str(capability)):
            score += 45.0
        elif q and q in _norm(names_blob):
            score += 20.0

        if score >= min_score:
            hits.append(
                GeneratorCapabilityHit(
                    capability=str(capability),
                    score=score,
                    description=desc,
                    implementation_module=impl_module,
                    implementation_symbol=impl_symbol,
                    related_commands=related_commands,
                    related_workflow_module=related_workflow,
                    tags=tags,
                    notes=notes,
                )
            )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def search_loader_map(
    query: str,
    *,
    engine: str = "reaxff",
    top_k: int = 8,
    min_score: float = 35.0,
) -> List[LoaderMapHit]:
    """Search loader/writer map entries directly."""
    q = _norm(query)
    q_toks = _token_set(_expand_query_text(query) or query)
    raw_map = _resolve_engine_loader_map(engine)
    raw_loaders = raw_map.get("loaders") or {}
    raw_writers = raw_map.get("writers") or {}
    hits: List[LoaderMapHit] = []

    for loader, meta in raw_loaders.items():
        dataclass_name = str((meta or {}).get("dataclass") or "")
        description = ""
        generator = ""
        primary_files = tuple(str(x) for x in _as_list((meta or {}).get("primary_files")))
        supplemental_files = tuple(str(x) for x in _as_list((meta or {}).get("supplemental_files")))
        fallback_files = tuple(str(x) for x in _as_list((meta or {}).get("fallback_files")))
        notes = tuple(str(x) for x in _as_list((meta or {}).get("notes")))

        names_blob = " ".join(
            [
                str(loader),
                dataclass_name,
                description,
                generator,
                " ".join(primary_files),
                " ".join(supplemental_files),
                " ".join(fallback_files),
            ]
        )
        text_blob = " ".join(notes)
        names_tokens = _token_set(names_blob)
        text_tokens = _token_set(text_blob)

        score = 0.0
        overlap_names = q_toks & names_tokens
        overlap_text = q_toks & text_tokens
        if overlap_names:
            score += 40.0 + 5.0 * len(overlap_names)
        if overlap_text:
            score += 12.0 + 3.0 * len(overlap_text)

        score += 0.42 * _fuzzy_ratio(q, names_blob)
        score += 0.12 * _fuzzy_ratio(q, text_blob)

        if q and q in _norm(str(loader)):
            score += 40.0
        if q and q in _norm(dataclass_name):
            score += 30.0

        if score >= min_score:
            hits.append(
                LoaderMapHit(
                    entry_type="loader",
                    name=str(loader),
                    score=score,
                    dataclass=dataclass_name,
                    description=description,
                    generator=generator,
                    primary_files=primary_files,
                    supplemental_files=supplemental_files,
                    fallback_files=fallback_files,
                    notes=notes,
                )
            )

    for writer, meta in raw_writers.items():
        dataclass_name = str((meta or {}).get("dataclass") or "")
        description = str((meta or {}).get("description") or "")
        generator = str((meta or {}).get("generator") or (meta or {}).get("generator_function") or "")
        primary_files = tuple(str(x) for x in _as_list((meta or {}).get("output_files")))
        supplemental_files = tuple()
        fallback_files = tuple()
        notes = tuple(str(x) for x in _as_list((meta or {}).get("notes")))

        names_blob = " ".join(
            [
                str(writer),
                dataclass_name,
                description,
                generator,
                str((meta or {}).get("generator_module") or ""),
                str((meta or {}).get("workflow_command") or ""),
                str((meta or {}).get("workflow_module") or ""),
                " ".join(primary_files),
            ]
        )
        text_blob = " ".join(notes)
        names_tokens = _token_set(names_blob)
        text_tokens = _token_set(text_blob)

        score = 0.0
        overlap_names = q_toks & names_tokens
        overlap_text = q_toks & text_tokens
        if overlap_names:
            score += 40.0 + 5.0 * len(overlap_names)
        if overlap_text:
            score += 12.0 + 3.0 * len(overlap_text)

        score += 0.42 * _fuzzy_ratio(q, names_blob)
        score += 0.12 * _fuzzy_ratio(q, text_blob)

        if q and q in _norm(str(writer)):
            score += 40.0
        if q and q in _norm(dataclass_name):
            score += 30.0

        if score >= min_score:
            hits.append(
                LoaderMapHit(
                    entry_type="writer",
                    name=str(writer),
                    score=score,
                    dataclass=dataclass_name,
                    description=description,
                    generator=generator,
                    primary_files=primary_files,
                    supplemental_files=supplemental_files,
                    fallback_files=fallback_files,
                    notes=notes,
                )
            )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def _workflow_links_for_task(task_name: str, task_module: str) -> Tuple[Tuple[str, str, str, str], ...]:
    """Return workflow/command links for an analyzer task."""
    links: List[Tuple[str, str, str, str]] = []
    seen: set[Tuple[str, str]] = set()

    try:
        from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
        analysis_routes = get_registered_analysis_commands()
    except Exception:
        analysis_routes = {}

    if task_name in analysis_routes:
        kind, help_text, module_path = _command_entry(task_name)
        kind = kind or "analysis"
        module_path = module_path or str(analysis_routes[task_name].module_path)
        key = (task_name, module_path)
        if key not in seen:
            seen.add(key)
            links.append((task_name, kind, module_path, help_text))

    # Timeseries tasks are dispatched through the `timeseries` workflow command.
    task_module_base, _ = _split_module_ref(str(task_module))
    if str(task_module_base).startswith("reaxkit.analysis.timeseries."):
        kind, help_text, module_path = _command_entry("timeseries")
        if not module_path:
            module_path = "reaxkit.workflows.timeseries_workflow"
        key = ("timeseries", module_path)
        if key not in seen:
            seen.add(key)
            links.append(("timeseries", kind or "workflow", module_path, help_text))

    return tuple(links)


def search_analysis_task_map(
    query: str,
    *,
    top_k: int = 8,
    min_score: float = 35.0,
) -> List[AnalysisTaskMapHit]:
    """Search analysis-task dataclass mapping."""
    q = _norm(query)
    q_toks = _token_set(_expand_query_text(query) or query)
    raw = (load_analysis_task_dataclass_map().get("tasks") or {})
    hits: List[AnalysisTaskMapHit] = []

    for task_name, meta in raw.items():
        dataclass_name = str((meta or {}).get("consumes_dataclass") or "")
        module_path = str((meta or {}).get("implementation_module") or (meta or {}).get("module") or "")
        desc = str((meta or {}).get("description") or (meta or {}).get("desc") or "")
        names_blob = " ".join([str(task_name), dataclass_name, module_path])
        text_blob = desc
        names_tokens = _token_set(names_blob)
        text_tokens = _token_set(text_blob)

        score = 0.0
        overlap_names = q_toks & names_tokens
        overlap_text = q_toks & text_tokens
        if overlap_names:
            score += 38.0 + 5.0 * len(overlap_names)
        if overlap_text:
            score += 14.0 + 3.0 * len(overlap_text)

        score += 0.44 * _fuzzy_ratio(q, names_blob)
        score += 0.20 * _fuzzy_ratio(q, text_blob)

        if q and q in _norm(str(task_name)):
            score += 45.0
        if q and q in _norm(dataclass_name):
            score += 25.0
        if q and q in _norm(desc):
            score += 20.0

        if score >= min_score:
            hits.append(
                AnalysisTaskMapHit(
                    task=str(task_name),
                    score=score,
                    dataclass=dataclass_name,
                    module=module_path,
                    workflow_links=_workflow_links_for_task(str(task_name), module_path),
                )
            )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def _tasks_for_dataclass(dataclass_name: str) -> List[AnalysisTaskMapHit]:
    """Resolve analyzer tasks that consume a dataclass."""
    data = load_analysis_task_dataclass_map()
    tasks_map = data.get("tasks") or {}
    usage = data.get("dataclass_usage_across_tasks") or data.get("dataclass_usage") or {}

    task_names = [str(x) for x in _as_list(usage.get(dataclass_name))]
    if not task_names:
        task_names = [
            str(name)
            for name, meta in tasks_map.items()
            if str((meta or {}).get("consumes_dataclass") or "") == dataclass_name
        ]

    out: List[AnalysisTaskMapHit] = []
    for task_name in task_names:
        meta = tasks_map.get(task_name) or {}
        module = str((meta or {}).get("implementation_module") or (meta or {}).get("module") or "")
        out.append(
            AnalysisTaskMapHit(
                task=task_name,
                score=0.0,
                dataclass=dataclass_name,
                module=module,
                workflow_links=_workflow_links_for_task(task_name, module),
            )
        )
    return out


def _parse_directory_ref(ref: str) -> Tuple[str, str]:
    """Parse a directory reference in the form '<file>:<key>'."""
    text = str(ref or "").strip()
    if not text:
        return "", ""
    if ":" not in text:
        return text, ""
    left, right = text.split(":", 1)
    return left.strip(), right.strip()


def _split_module_ref(value: str) -> Tuple[str, str]:
    """
    Split a module reference of the form ``module.path: token``.

    Returns
    -------
    tuple[str, str]
        ``(module_path, token)`` where token is often a command/function hint.
    """
    text = str(value or "").strip()
    if not text:
        return "", ""
    if ":" not in text:
        return text, ""
    left, right = text.split(":", 1)
    return left.strip(), right.strip()


def _workflow_command_from_ref(ref: str) -> str:
    """Extract CLI command hint from ``related_workflow_module`` style refs."""
    _, token = _split_module_ref(ref)
    return token


def _analysis_hit_for_task(task_name: str) -> Optional[AnalysisTaskMapHit]:
    """Build an analysis task hit from task metadata by exact name."""
    tasks_map = (load_analysis_task_dataclass_map().get("tasks") or {})
    if not isinstance(tasks_map, dict):
        return None
    meta = tasks_map.get(task_name) or tasks_map.get(task_name.replace("-", "_")) or {}
    if not isinstance(meta, dict):
        return None
    dataclass_name = str(meta.get("consumes_dataclass") or "")
    module_path = str(meta.get("implementation_module") or meta.get("module") or "")
    if not dataclass_name:
        return None
    return AnalysisTaskMapHit(
        task=str(task_name),
        score=200.0,  # directory-exact pointer is treated as a very strong match
        dataclass=dataclass_name,
        module=module_path,
        workflow_links=_workflow_links_for_task(str(task_name), module_path),
    )


def _resolve_tasks_from_command_hits(command_hits: List[CommandHit]) -> List[AnalysisTaskMapHit]:
    """
    Resolve analyzer tasks from command directories.

    Directory values are expected to look like:
    - ``analysis_task_dataclass_map.yaml:trainset_group_comments``
    """
    out: List[AnalysisTaskMapHit] = []
    seen: set[str] = set()
    for hit in command_hits:
        file_name, key = _parse_directory_ref(hit.directory)
        if _norm(file_name) != "analysis task dataclass map.yaml" and _norm(file_name) != "analysis_task_dataclass_map.yaml":
            continue
        if not key:
            continue
        task = _analysis_hit_for_task(key)
        if task is None:
            continue
        if task.task in seen:
            continue
        seen.add(task.task)
        out.append(task)
    return out


def _score_label(score: float) -> str:
    """Map numeric score to a user-facing qualitative label."""
    if score >= 120.0:
        return "very strong match"
    if score >= 90.0:
        return "strong match"
    if score >= 60.0:
        return "likely relevant"
    if score >= 35.0:
        return "possible weak/contextual match"
    return "below default threshold"


def _collect_file_dataclass_links(
    file_hits: List[HelpHit],
    engine: str,
) -> Tuple[List[Tuple[HelpHit, Dict[str, str]]], Dict[str, List[AnalysisTaskMapHit]]]:
    """
    Build file -> dataclass links and dataclass -> analyzer-task links.

    Returns
    -------
    tuple
        ``(rows, tasks_by_dataclass)``, where rows is ordered by file-hit ranking.
    """
    engine_map = _resolve_engine_loader_map(engine)
    rows: List[Tuple[HelpHit, Dict[str, str]]] = []
    tasks_by_dataclass: Dict[str, List[AnalysisTaskMapHit]] = {}
    for hit in file_hits:
        targets = _engine_targets_for_file(engine_map, hit.file)
        for row in targets:
            dataclass_name = str(row.get("dataclass") or "")
            rows.append((hit, row))
            tasks_by_dataclass.setdefault(dataclass_name, _tasks_for_dataclass(dataclass_name))
    return rows, tasks_by_dataclass


def _normalize_help_query(query: str) -> str:
    """
    Normalize query with canonical alias expansion and lemmatization.

    Sources are loaded from `help_intents.yaml` under `index_and_intent`.
    """
    raw = _norm(str(query or ""))
    if not raw:
        return ""
    lemma = _lemmatize_with_spacy(raw)
    aliased_raw = _word_alias_expand(raw)
    aliased_lemma = _word_alias_expand(lemma)
    return " ".join(_unique_strs([raw, lemma, aliased_raw, aliased_lemma]))


def _searchable_text(value: Any) -> str:
    """Convert a metadata field value into plain searchable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x is not None)
    return str(value)


def _score_metadata_entry(name: str, entry: Dict[str, Any], q: str, q_toks: set[str]) -> float:
    """
    Weighted ranking over standardized help fields.

    Priority:
    aliases/help_search_examples/tags > description > notes
    """
    field_weights = {
        "aliases": 56.0,
        "help_search_examples": 52.0,
        "tags": 48.0,
        "description": 24.0,
        "notes": 16.0,
    }

    score = 0.0
    name_blob = _searchable_text(name)
    name_toks = _token_set(name_blob)
    overlap_name = q_toks & name_toks
    if overlap_name:
        score += 44.0 + 6.0 * len(overlap_name)
    score += 0.30 * _token_set_fuzzy_ratio(q, name_blob)

    for field, weight in field_weights.items():
        blob = _searchable_text((entry or {}).get(field))
        if not blob:
            continue
        toks = _token_set(blob)
        overlap = q_toks & toks
        if overlap:
            score += weight + 3.5 * len(overlap)
        score += 0.08 * weight * (_token_set_fuzzy_ratio(q, blob) / 100.0)

    if q and q in _norm(name_blob):
        score += 35.0
    return score


def _iter_engine_generator_sources(engine: str) -> List[str]:
    """Return generator capability source files for common + selected engine."""
    src = load_help_information_sources()
    layer = (((src.get("source_files_by_layer") or {}).get("generator_level")) or {})
    out: List[str] = []

    for item in (layer.get("common") or []):
        if isinstance(item, dict) and item.get("file"):
            out.append(str(item["file"]))

    for item in (layer.get(_norm(engine)) or layer.get(str(engine)) or []):
        if isinstance(item, dict) and item.get("file"):
            out.append(str(item["file"]))

    return _unique_strs(out)


def _iter_file_level_io_sources(engine: str) -> List[str]:
    """Return file-level IO source files for selected engine."""
    src = load_help_information_sources()
    source_layers = src.get("source_files_by_layer") or {}
    file_level = (source_layers.get("file_level") or {}) if isinstance(source_layers, dict) else {}
    engine_block = file_level.get(_norm(engine)) or file_level.get(str(engine)) or {}
    engine_rows = (engine_block.get("io") or []) if isinstance(engine_block, dict) else []
    out: List[str] = []
    for item in engine_rows:
        if isinstance(item, dict) and item.get("file"):
            out.append(str(item["file"]))
    return _unique_strs(out)


def _file_level_mapping_source(engine: str) -> str:
    """Return file-level mapping source file for selected engine."""
    src = load_help_information_sources()
    file_level = (((src.get("source_files_by_layer") or {}).get("file_level")) or {})
    if not isinstance(file_level, dict):
        return ""
    engine_block = file_level.get(_norm(engine)) or file_level.get(str(engine)) or {}
    if not isinstance(engine_block, dict):
        return ""
    for item in (engine_block.get("mapping") or []):
        if isinstance(item, dict) and item.get("file"):
            return str(item["file"])
    return ""


def _iter_analyzer_sources() -> List[str]:
    """Return analyzer-level mapping sources."""
    src = load_help_information_sources()
    rows = (((src.get("source_files_by_layer") or {}).get("analyzer_level")) or [])
    out: List[str] = []
    for item in rows:
        if isinstance(item, dict) and item.get("file"):
            out.append(str(item["file"]))
    return _unique_strs(out)


def _iter_utility_sources() -> List[str]:
    """Return utility-level metadata sources."""
    src = load_help_information_sources()
    rows = (((src.get("source_files_by_layer") or {}).get("utility_level")) or [])
    out: List[str] = []
    for item in rows:
        if isinstance(item, dict) and item.get("file"):
            out.append(str(item["file"]))
    return _unique_strs(out)


def _iter_workflow_sources() -> List[str]:
    """Return workflow-level mapping sources."""
    src = load_help_information_sources()
    rows = (((src.get("source_files_by_layer") or {}).get("workflow_level")) or [])
    out: List[str] = []
    for item in rows:
        if isinstance(item, dict) and item.get("file"):
            out.append(str(item["file"]))
    return _unique_strs(out)


@lru_cache(maxsize=64)
def _load_yaml_rel_cached(rel_path: str) -> Dict[str, Any]:
    """Cached YAML loader for `reaxkit/...` repository-relative paths."""
    return _read_yaml_from_reaxkit_rel_path(rel_path)


def _search_named_section(
    rel_path: str,
    section_key: str,
    q: str,
    q_toks: set[str],
    *,
    top_k: int,
    min_score: float,
    exact_match: bool = False,
    exact_query: str = "",
) -> List[Tuple[str, float, Dict[str, Any], str]]:
    """Search a named mapping section and return ranked `(name, score, entry, source)` rows."""
    data = _load_yaml_rel_cached(rel_path)
    section = (data.get(section_key) or {})
    if not isinstance(section, dict):
        return []

    hits: List[Tuple[str, float, Dict[str, Any], str]] = []
    for name, entry in section.items():
        if not isinstance(entry, dict):
            continue
        if exact_match:
            name_norm = _norm(str(name))
            alias_norms = {_norm(a) for a in _as_list(entry.get("aliases"))}
            if exact_query and exact_query not in ({name_norm} | alias_norms):
                continue
            score = 1000.0 if exact_query == name_norm else 950.0
            hits.append((str(name), float(score), entry, rel_path))
            continue
        score = _score_metadata_entry(str(name), entry, q, q_toks)
        if score >= min_score:
            hits.append((str(name), float(score), entry, rel_path))
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits[:top_k]


def _find_file_usage_refs(file_name: str, mapping_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Resolve `file_to_dataclass_across_loaders_and_writers` references for a file key."""
    refs = (mapping_data.get("file_to_dataclass_across_loaders_and_writers") or {})
    if not isinstance(refs, dict):
        return {"primary_files": [], "supplemental_files": [], "output_files": []}

    target_key = ""
    for key in refs.keys():
        if _norm(str(key)) == _norm(file_name):
            target_key = str(key)
            break
    if not target_key:
        return {"primary_files": [], "supplemental_files": [], "output_files": []}

    row = refs.get(target_key) or {}
    return {
        "primary_files": list(row.get("primary_files") or []),
        "supplemental_files": list(row.get("supplemental_files") or []),
        "output_files": list(row.get("output_files") or []),
    }


def _search_mapping_file_refs(
    mapping_data: Dict[str, Any],
    q: str,
    q_toks: set[str],
    *,
    top_k: int,
    min_score: float,
    exact_match: bool = False,
    exact_query: str = "",
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Search `file_to_dataclass_across_loaders_and_writers` directly.

    Used as fallback for engines without file-level IO index.
    """
    refs = (mapping_data.get("file_to_dataclass_across_loaders_and_writers") or {})
    if not isinstance(refs, dict):
        return []

    hits: List[Tuple[str, float, Dict[str, Any]]] = []
    for file_name, row in refs.items():
        if not isinstance(row, dict):
            continue

        file_key = str(file_name)
        if exact_match:
            primary = row.get("primary_files") or []
            supplemental = row.get("supplemental_files") or []
            output_files = row.get("output_files") or []
            field_vals: List[str] = [file_key]
            for block in (primary, supplemental, output_files):
                if not isinstance(block, list):
                    continue
                for item in block:
                    if not isinstance(item, dict):
                        continue
                    for k in ("kind", "name", "dataclass"):
                        if item.get(k):
                            field_vals.append(str(item.get(k)))
            exact_pool = {_norm(v) for v in field_vals if str(v).strip()}
            if exact_query not in exact_pool:
                continue
            hits.append((file_key, 1000.0, row))
            continue

        # Build searchable mapping entry from primary/supplemental (+ output) usage rows.
        primary = row.get("primary_files") or []
        supplemental = row.get("supplemental_files") or []
        output_files = row.get("output_files") or []

        def _row_blob(items: Any) -> str:
            if not isinstance(items, list):
                return ""
            chunks: List[str] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                for k in ("kind", "name", "dataclass"):
                    v = item.get(k)
                    if v is not None:
                        chunks.append(str(v))
            return " ".join(chunks)

        pseudo_entry = {
            "aliases": [file_key],
            "help_search_examples": [],
            "tags": ["mapping", "loader", "writer"],
            "description": " ".join([
                _row_blob(primary),
                _row_blob(supplemental),
                _row_blob(output_files),
            ]).strip(),
            "notes": [],
        }

        # Reuse weighted scorer: title + aliases + description text from mapping rows.
        score = _score_metadata_entry(file_key, pseudo_entry, q, q_toks)
        if score >= min_score:
            hits.append((file_key, float(score), row))

    hits.sort(key=lambda x: x[1], reverse=True)
    return hits[:top_k]


def _format_notes_text(notes: Any) -> str:
    """Render notes as a concise single line."""
    vals = _as_list(notes)
    return "; ".join(vals) if vals else "None"


def _format_all_fields(entry: Dict[str, Any]) -> str:
    """Render all fields of an entry as YAML-like text."""
    dumped = yaml.safe_dump(entry or {}, sort_keys=False, allow_unicode=False).rstrip()
    return dumped if dumped else "{}"


def _append_all_fields_bulleted(
    lines: List[str],
    data: Dict[str, Any],
    *,
    base_indent: str = "  ",
) -> None:
    """Render all fields using bullet/indent style (o -> • -> -)."""
    if not isinstance(data, dict) or not data:
        lines.append(f"{base_indent}• all fields: None")
        return

    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{base_indent}• {key}:")
            if not value:
                lines.append(f"{base_indent}  - None")
            else:
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{base_indent}  -")
                        for sub_key, sub_val in item.items():
                            lines.append(f"{base_indent}    • {sub_key}: {sub_val}")
                    else:
                        lines.append(f"{base_indent}  - {item}")
        elif isinstance(value, dict):
            lines.append(f"{base_indent}• {key}:")
            if not value:
                lines.append(f"{base_indent}  - None")
            else:
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, list):
                        lines.append(f"{base_indent}  • {sub_key}:")
                        if not sub_val:
                            lines.append(f"{base_indent}    - None")
                        else:
                            for item in sub_val:
                                lines.append(f"{base_indent}    - {item}")
                    else:
                        lines.append(f"{base_indent}  • {sub_key}: {sub_val}")
        else:
            lines.append(f"{base_indent}• {key}: {value}")


def _search_score_label(score: float) -> str:
    """Map numeric search score to qualitative likelihood label."""
    s = float(score)
    if s >= 121.0:
        return "extremely likely"
    if 86.0 <= s <= 120.0:
        return "very likely"
    if 61.0 <= s <= 85.0:
        return "likely"
    if 35.0 <= s <= 60.0:
        return "probably"
    return "possible"


def _append_section_header(lines: List[str], title: str) -> None:
    """Append a section title and visual separator."""
    lines.append("")
    lines.append(title)
    lines.append("----------------------------")


def _append_notes_lines(lines: List[str], notes: Any, *, indent: str = "  o ") -> None:
    """Append notes with one note per line instead of a joined sentence."""
    vals = _as_list(notes)
    if not vals:
        lines.append(f"{indent}notes: None")
        return
    lines.append(f"{indent}notes:")
    for note in vals:
        lines.append(f"    - {note}")


def build_help_relationship_report(
    query: str,
    *,
    top_k: int = 8,
    min_score: float = 35.0,
    engine: str = "reaxff",
    all_info: bool = False,
    exact_match: bool = False,
) -> str:
    """Build layered help report from `help_information_sources.yaml`."""
    q_proc = _norm(query) if exact_match else _normalize_help_query(query)
    if not q_proc:
        return "No matches."
    q = _norm(q_proc)
    q_toks = _token_set(q_proc)
    q_exact = _norm(query)

    out: List[str] = [f"Normalized query: {q_proc}"]

    # 1) generator level
    generator_hits: List[Tuple[str, float, Dict[str, Any], str]] = []
    for rel_path in _iter_engine_generator_sources(engine):
        generator_hits.extend(
            _search_named_section(
                rel_path,
                "generators",
                q,
                q_toks,
                top_k=top_k,
                min_score=min_score,
                exact_match=exact_match,
                exact_query=q_exact,
            )
        )
    generator_hits.sort(key=lambda x: x[1], reverse=True)
    generator_hits = generator_hits[:top_k]

    if generator_hits:
        _append_section_header(out, "GENERATOR LEVEL")
        for name, score, entry, _ in generator_hits:
            score_label = _search_score_label(score)
            if all_info:
                out.append(f"o (search score: {score_label}) {name} (score={score:.1f})")
                _append_all_fields_bulleted(out, entry, base_indent="  ")
            else:
                out.append(f"o (search score: {score_label}) {name}")
                out.append(f"  • description: {str(entry.get('description') or '').strip()}")
                _append_notes_lines(out, entry.get("notes"), indent="  • ")
                workflow_cmd = _workflow_command_from_ref(str(entry.get("related_workflow_module") or ""))
                if not workflow_cmd:
                    workflow_cmd = str(name)
                out.append(f"  • How to use the workflow: reaxkit {workflow_cmd} -h\n")

    # 2) file level (io then mapping lookup)
    file_hits: List[Tuple[str, float, Dict[str, Any], str]] = []
    for rel_path in _iter_file_level_io_sources(engine):
        file_hits.extend(
            _search_named_section(
                rel_path,
                "files",
                q,
                q_toks,
                top_k=top_k,
                min_score=min_score,
                exact_match=exact_match,
                exact_query=q_exact,
            )
        )
    file_hits.sort(key=lambda x: x[1], reverse=True)
    file_hits = file_hits[:top_k]
    mapping_path = _file_level_mapping_source(engine)
    mapping_data = _load_yaml_rel_cached(mapping_path) if mapping_path else {}

    # Engines like ams/lammps may not have IO index files; search mapping directly.
    if not file_hits and mapping_data:
        mapping_hits = _search_mapping_file_refs(
            mapping_data,
            q,
            q_toks,
            top_k=top_k,
            min_score=min_score,
            exact_match=exact_match,
            exact_query=q_exact,
        )
        file_hits = [(fname, sc, {}, mapping_path) for fname, sc, _ in mapping_hits]

    if file_hits:
        _append_section_header(out, "FILE LEVEL -> DATACLASS")
        for file_name, score, file_entry, file_src in file_hits:
            score_label = _search_score_label(score)
            refs = _find_file_usage_refs(file_name, mapping_data)
            primary_refs = refs.get("primary_files") or []
            supplemental_refs = refs.get("supplemental_files") or []
            output_refs = refs.get("output_files") or []
            all_refs = [*primary_refs, *supplemental_refs, *output_refs]

            if not all_info:
                if all_refs:
                    for ref in all_refs:
                        kind = str(ref.get("kind") or "")
                        method = str(ref.get("name") or "")
                        dataclass_name = str(ref.get("dataclass") or "")
                        if kind and method and dataclass_name:
                            out.append(
                                f"o (search score: {score_label}) Related data is in file {file_name}, and its content is mapped to {dataclass_name} dataclass through the {method} method."
                            )
                else:
                    out.append(f"o (search score: {score_label}) {file_name} has no loader/writer mapping entry for engine '{engine}'.")
                continue

            out.append(f"o (search score: {score_label}) {file_name} (score={score:.1f})")
            out.append(f"  • source_file: {file_src}")
            out.append("  • file_fields:")
            _append_all_fields_bulleted(out, file_entry, base_indent="    ")
            out.append("  • mapping_usage:")
            _append_all_fields_bulleted(out, refs, base_indent="    ")

            loaders = mapping_data.get("loaders") or {}
            writers = mapping_data.get("writers") or {}
            for ref in all_refs:
                ref_kind = str(ref.get("kind") or "").strip().lower()
                ref_name = str(ref.get("name") or "").strip()
                if not ref_name:
                    continue
                if ref_kind == "loader" and isinstance(loaders, dict) and ref_name in loaders:
                    out.append(f"  • loader::{ref_name}")
                    _append_all_fields_bulleted(out, loaders.get(ref_name) or {}, base_indent="    ")
                if ref_kind == "writer" and isinstance(writers, dict) and ref_name in writers:
                    out.append(f"  • writer::{ref_name}")
                    _append_all_fields_bulleted(out, writers.get(ref_name) or {}, base_indent="    ")
    out.append(" ")

    # 3) utility level
    utility_hits: List[Tuple[str, float, Dict[str, Any], str]] = []
    for rel_path in _iter_utility_sources():
        utility_hits.extend(
            _search_named_section(
                rel_path,
                "utilities",
                q,
                q_toks,
                top_k=top_k,
                min_score=min_score,
                exact_match=exact_match,
                exact_query=q_exact,
            )
        )
    utility_hits.sort(key=lambda x: x[1], reverse=True)
    utility_hits = utility_hits[:top_k]

    if utility_hits:
        _append_section_header(out, "UTILITY LEVEL")
        for name, score, entry, _ in utility_hits:
            score_label = _search_score_label(score)
            if all_info:
                out.append(f"o (search score: {score_label}) {name} (score={score:.1f})")
                _append_all_fields_bulleted(out, entry, base_indent="  ")
            else:
                out.append(f"o (search score: {score_label}) {name}")
                out.append(f"  o description: {str(entry.get('description') or '').strip()}")
                out.append(f"  o implementation_module: {str(entry.get('implementation_module') or '').strip()}")
                _append_notes_lines(out, entry.get("notes"), indent="  o ")
                out.append(" ")

    # 4) analyzer level
    analyzer_hits: List[Tuple[str, float, Dict[str, Any], str]] = []
    for rel_path in _iter_analyzer_sources():
        analyzer_hits.extend(
            _search_named_section(
                rel_path,
                "tasks",
                q,
                q_toks,
                top_k=top_k,
                min_score=min_score,
                exact_match=exact_match,
                exact_query=q_exact,
            )
        )
    analyzer_hits.sort(key=lambda x: x[1], reverse=True)
    analyzer_hits = analyzer_hits[:top_k]

    if analyzer_hits:
        _append_section_header(out, "ANALYZER LEVEL")
        for name, score, entry, _ in analyzer_hits:
            score_label = _search_score_label(score)
            if all_info:
                out.append(f"o (search score: {score_label}) {name} (score={score:.1f})")
                _append_all_fields_bulleted(out, entry, base_indent="  ")
            else:
                out.append(f"o (search score: {score_label}) {name}")
                out.append(f"  • consumes_dataclass: {str(entry.get('consumes_dataclass') or '').strip()}")
                out.append(f"  • description: {str(entry.get('description') or '').strip()}")
                _append_notes_lines(out, entry.get("notes"), indent="  • ")
                workflow_cmd = _workflow_command_from_ref(str(entry.get("related_workflow_module") or ""))
                if not workflow_cmd:
                    workflow_cmd = str(name)
                out.append(f"  • How to use the workflow: reaxkit {workflow_cmd} -h")
                out.append(" ")

    # 5) workflow level
    workflow_hits: List[Tuple[str, float, Dict[str, Any], str]] = []
    for rel_path in _iter_workflow_sources():
        workflow_hits.extend(
            _search_named_section(
                rel_path,
                "workflows",
                q,
                q_toks,
                top_k=top_k,
                min_score=min_score,
                exact_match=exact_match,
                exact_query=q_exact,
            )
        )
    workflow_hits.sort(key=lambda x: x[1], reverse=True)
    workflow_hits = workflow_hits[:top_k]

    if workflow_hits:
        _append_section_header(out, "WORKFLOW LEVEL")
        for name, score, entry, _ in workflow_hits:
            score_label = _search_score_label(score)
            if all_info:
                out.append(f"o (search score: {score_label}) {name} (score={score:.1f})")
                _append_all_fields_bulleted(out, entry, base_indent="  ")
            else:
                out.append(f"o (search score: {score_label}) {name}")
                out.append(f"  • description: {str(entry.get('description') or '').strip()}")
                _append_notes_lines(out, entry.get("notes"), indent="  • ")
                out.append(" ")

    if len(out) == 1:
        return f"No matches for: {query!r}"
    return "\n".join(out).rstrip()
