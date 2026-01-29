from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import re

try:
    import yaml  # pyyaml
except Exception as e:  # pragma: no cover
    raise ImportError("PyYAML is required to use ReaxKit help index. Install with: pip install pyyaml") from e


# ----------------------------
# Data access (package files)
# ----------------------------

def _read_yaml_from_pkg_data(filename: str) -> Dict[str, Any]:
    """
    Load a YAML file shipped inside `reaxkit/data/` using importlib.resources.
    Works in editable installs and normal pip installs.
    """
    try:
        from importlib import resources
        data_pkg = resources.files("reaxkit.data")
        path = data_pkg / filename
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        raise FileNotFoundError(
            f"Could not read '{filename}' from package 'reaxkit.data'. "
            f"Make sure it exists under src/reaxkit/data/ and is included in package data."
        ) from e

    obj = yaml.safe_load(text) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/dict in '{filename}'.")
    return obj


def load_input_index() -> Dict[str, Any]:
    return _read_yaml_from_pkg_data("reaxff_input_files_contents.yaml")


def load_output_index() -> Dict[str, Any]:
    return _read_yaml_from_pkg_data("reaxff_output_files_contents.yaml")


# ----------------------------
# Search / ranking
# ----------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    s = s.lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> List[str]:
    return _WORD_RE.findall(_norm(s))


def _fuzzy_ratio(a: str, b: str) -> float:
    """
    Returns 0..100 similarity.
    Uses rapidfuzz if available; falls back to difflib.
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
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    return [str(v)]


def _entry_search_blobs(file_key: str, entry: Dict[str, Any]) -> Dict[str, str]:
    """
    Create concatenated strings for different searchable fields.
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
    examples = _as_list(entry.get("examples"))

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
        "examples": " ".join(examples),
    }


@dataclass(frozen=True)
class HelpHit:
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
    Search across BOTH input and output YAML indices and return ranked hits.

    Returns a flat list of HelpHit with hit.kind in {"input","output"}.
    Call `group_hits()` to split into input/output sections for printing.
    """
    q = _norm(query)
    q_toks = set(_tokens(query))

    in_idx = load_input_index()
    out_idx = load_output_index()

    hits: List[HelpHit] = []
    hits.extend(_search_one_index("input", in_idx, q, q_toks))
    hits.extend(_search_one_index("output", out_idx, q, q_toks))

    # overall top_k across both
    hits.sort(key=lambda h: h.score, reverse=True)
    hits = [h for h in hits if h.score >= min_score]
    return hits[:top_k]


def _search_one_index(kind: str, idx: Dict[str, Any], q: str, q_toks: set[str]) -> List[HelpHit]:
    files = idx.get("files", {}) or {}
    if not isinstance(files, dict):
        return []

    res: List[HelpHit] = []

    for file_key, entry in files.items():
        if not isinstance(entry, dict):
            continue

        blobs = _entry_search_blobs(file_key, entry)

        score = 0.0
        why: List[str] = []

        # 1) deterministic boosts
        names_norm = _norm(blobs["names"])
        if q and q in names_norm.split():
            score += 120.0
            why.append("exact file/alias match")

        # token overlaps (fast and robust)
        def _overlap(field_name: str, weight: float) -> None:
            nonlocal score
            toks = set(_tokens(blobs[field_name]))
            ov = q_toks & toks
            if ov:
                score += weight + 4.0 * len(ov)
                why.append(f"{field_name} match: {', '.join(sorted(list(ov))[:5])}")

        _overlap("tags", 30.0)
        _overlap("best_for", 22.0)
        _overlap("core", 18.0)
        _overlap("optional", 8.0)   # smaller boost; optional-only matches should rank lower
        _overlap("derived", 18.0)
        _overlap("related", 14.0)
        _overlap("desc", 10.0)

        # 2) fuzzy matching over key fields (weighted)
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


def group_hits(hits: Iterable[HelpHit]) -> Tuple[List[HelpHit], List[HelpHit]]:
    """
    Split hits into (input_hits, output_hits), each sorted by score desc.
    """
    ins = [h for h in hits if h.kind == "input"]
    outs = [h for h in hits if h.kind == "output"]
    ins.sort(key=lambda h: h.score, reverse=True)
    outs.sort(key=lambda h: h.score, reverse=True)
    return ins, outs


def format_hits(
    hits: List[HelpHit],
    *,
    show_why: bool = True,
    show_examples: bool = False,
    show_tags: bool = False,
    show_core_vars: bool = False,
    show_optional_vars: bool = False,
    show_derived_vars: bool = False,
    show_notes: bool = False,
) -> str:

    """
    Pretty output that divides INPUT and OUTPUT with a dashed line.
    If show_kind_flag=True, shows '--input'/'--output' next to each result.
    """
    in_hits, out_hits = group_hits(hits)

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
            ex = e.get("examples") or []
            if ex:
                lines.append(f"  ex:  {ex[0]}")
        if show_tags:
            tags = e.get("tags") or []
            if tags:
                lines.append(f"  tags: {tags}")

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
        parts.append("OUTPUT FILES")
        parts.extend(_fmt_one(h) for h in out_hits)

    if not parts:
        return "❌ No matches."

    parts.append("")
    parts.append(
        "Tip: use `reaxkit <filename> -h` or `reaxkit <filename> <task> -h` "
        "to see a more comprehensive description of available options, "
        "examples, and usage details."
    )

    return "\n".join(parts)
