# _gen_mkdocs.py
"""
Generate mkdocs.yml nav from Markdown files under <repo_root>/docs/api/.

Fixes
-----
1) No top-level "API:" wrapper; nav lists IO/Analysis/Utils/Workflows directly.
2) Top-level section order: IO -> Analysis -> Utils -> Workflows (then others).
3) Removes previous auto-generated sections (IO/Analysis/Utils/Workflows and legacy names),
   leaving only Home/Installation/Quickstart + regenerated API sections.
4) Writes mkdocs.yml with blank-line separation between major blocks (theme/plugins/nav).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

try:
    import yaml  # PyYAML
except ImportError as e:  # pragma: no cover
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e


NavItem = Union[str, Dict[str, Any]]


# -------------------- Titles --------------------

def _to_title_from_stem(stem: str) -> str:
    """
    Convert filename stem to a clean MkDocs nav title.

    Rules:
    - index -> Overview
    - strip trailing structural suffixes:
        *_handler_doc
        *_generator_doc
        *_workflow_doc
        *_analyzer_doc
        *_doc
    - snake_case -> Title Case
    """
    s = stem.lower()

    if s == "index":
        return "🔎 Overview"

    # ordered from most specific → least specific
    SUFFIXES = [
        "_handler_doc",
        "_generator_doc",
        "_workflow_doc",
        "_analyzer_doc",
        "_doc",
    ]

    for suf in SUFFIXES:
        if s.endswith(suf):
            stem = stem[: -len(suf)]
            break

    if stem.isupper():
        return stem

    parts = [p for p in stem.split("_") if p]
    return " ".join(p.capitalize() for p in parts)



# -------------------- Repo root discovery --------------------

def _find_repo_root(start: Path) -> Path:
    """
    Find repo root by walking upward until we find:
      <root>/src/reaxkit
    """
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "src" / "reaxkit").is_dir():
            return p
    raise FileNotFoundError(
        "Could not locate repo root containing 'src/reaxkit'. "
        f"Started search from: {start}"
    )


def _auto_paths(script_path: Path) -> tuple[Path, Path]:
    """
    Auto-detect:
      api_dir    = <repo_root>/docs/api
      mkdocs_yml = <repo_root>/mkdocs.yml
    """
    repo_root = _find_repo_root(script_path.parent)
    return repo_root / "docs" / "api", repo_root / "mkdocs.yml"


# -------------------- Tree builders --------------------

def _insert_path_tree(root: dict, parts: List[str], md_rel_path: str) -> None:
    """Insert a path into a nested dict tree (folders->dict, files->path str)."""
    node = root
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = md_rel_path


def _tree_to_nav(tree: dict) -> List[NavItem]:
    """
    Convert a nested dict tree into MkDocs nav format.

    Ordering:
    - index.md (Overview) first within each folder
    - then alphabetical (files, then folders)
    """
    items: List[NavItem] = []

    files: List[tuple[str, str]] = []
    folders: List[tuple[str, dict]] = []
    for k, v in tree.items():
        if isinstance(v, dict):
            folders.append((k, v))
        else:
            files.append((k, str(v)))

    def file_key(item: tuple[str, str]) -> tuple[int, str]:
        fname = item[0].lower()
        return (0, fname) if fname == "index.md" else (1, fname)

    files.sort(key=file_key)
    folders.sort(key=lambda kv: kv[0].lower())

    for filename, md_path in files:
        title = _to_title_from_stem(Path(filename).stem)
        items.append({title: md_path})

    for folder_name, subtree in folders:
        title = _to_title_from_stem(folder_name)
        items.append({title: _tree_to_nav(subtree)})

    return items


def build_sections_from_api_dir(api_dir: Path) -> Dict[str, List[NavItem]]:
    """
    Build top-level nav sections from docs/api folder.

    Returns a mapping:
      {"IO": [...], "Analysis": [...], "Utils": [...], "Workflows": [...], ...}
    """
    if not api_dir.exists():
        raise FileNotFoundError(f"Could not find api directory: {api_dir}")

    # tree keyed by api subfolders (e.g., io/, analysis/, utils/, workflows/)
    top_tree: Dict[str, Any] = {}

    for md in sorted(api_dir.rglob("*.md")):
        rel_to_api = md.relative_to(api_dir)           # e.g., analysis/composed/index.md
        rel_to_docs = Path("api") / rel_to_api         # mkdocs path: api/analysis/...
        rel_str = rel_to_docs.as_posix()

        parts = list(rel_to_api.parts)                 # ("analysis","composed","index.md")
        if not parts:
            continue

        top = parts[0]                                 # analysis
        rest = parts[1:]                               # composed/index.md

        if top not in top_tree:
            top_tree[top] = {}
        if not rest:
            # api/<top>/some.md case (rare)
            top_tree[top][rel_to_api.name] = rel_str
        else:
            _insert_path_tree(top_tree[top], rest, rel_str)

    # Convert each top folder to a nav list, with desired title casing
    out: Dict[str, List[NavItem]] = {}
    for top, subtree in top_tree.items():
        title = _to_title_from_stem(top)               # "Io" -> not desired; map below
        # enforce exact labels for your main buckets
        if top.lower() == "io":
            title = "IO"
        elif top.lower() == "analysis":
            title = "Analysis"
        elif top.lower() == "utils":
            title = "Utils"
        elif top.lower() == "workflows":
            title = "Workflows"

        out[title] = _tree_to_nav(subtree) if isinstance(subtree, dict) else [{title: str(subtree)}]

    return out


def _sorted_top_sections(sections: Dict[str, List[NavItem]]) -> List[NavItem]:
    """
    Return top-level section entries sorted:
      IO -> Analysis -> Utils -> Workflows -> (others alphabetical)
    """
    priority = ["IO", "Analysis", "Utils", "Workflows"]
    keys = list(sections.keys())

    def key_fn(k: str) -> tuple[int, str]:
        if k in priority:
            return (priority.index(k), "")
        return (len(priority), k.lower())

    ordered = sorted(keys, key=key_fn)

    nav: List[NavItem] = []
    for k in ordered:
        if sections[k]:  # omit empty sections
            nav.append({k: sections[k]})
    return nav


# -------------------- Nav rewriting --------------------

def _ensure_basic_nav_entries(nav: List[NavItem]) -> List[NavItem]:
    """Ensure Home/Installation/Quickstart exist (no duplicates)."""
    want = [
        ("Home", "index.md"),
        ("Installation", "installation.md"),
        ("Quickstart", "quickstart.md"),
    ]
    existing = set()
    for item in nav:
        if isinstance(item, dict):
            existing.update(item.keys())

    out = list(nav)
    for key, path in reversed(want):
        if key not in existing:
            out.insert(0, {key: path})
    return out


def _strip_old_generated_sections(old_nav: List[NavItem]) -> List[NavItem]:
    """
    Remove previously generated sections that should not persist.

    Removes these top-level keys if present:
      - IO, Analysis, Utils, Workflows
      - legacy: "Analysis Routines", "Workflows" (kept), but we want to regenerate Workflows too,
        so remove it as well.
    """
    to_remove = {
        "API",
        "IO",
        "Analysis",
        "Utils",
        "Workflows",
        "Analysis Routines",
        "Analysis routines",
        "AnalysisRoutines",
    }

    new_nav: List[NavItem] = []
    for item in old_nav:
        if isinstance(item, dict):
            k = next(iter(item.keys()), None)
            if k in to_remove:
                continue
        new_nav.append(item)

    return new_nav


def _dump_yaml_block(obj: Any) -> str:
    """YAML dump helper with stable ordering and nice formatting."""
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).rstrip() + "\n"


def write_mkdocs_yml_with_spacing(mkdocs_path: Path, data: dict) -> None:
    """
    Write mkdocs.yml with an empty line separating major sections:
      [other keys]

      theme:

      plugins:

      nav:
    """
    # Separate major blocks
    other: dict = {}
    theme = None
    plugins = None
    nav = None

    for k, v in data.items():
        if k == "theme":
            theme = v
        elif k == "plugins":
            plugins = v
        elif k == "nav":
            nav = v
        else:
            other[k] = v

    blocks: list[str] = []

    # Dump each block without trailing whitespace, then join with TWO newlines
    def dump_block(obj: dict) -> str:
        return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).strip()

    if other:
        blocks.append(dump_block(other))
    if theme is not None:
        blocks.append(dump_block({"theme": theme}))
    if plugins is not None:
        blocks.append(dump_block({"plugins": plugins}))
    if nav is not None:
        blocks.append(dump_block({"nav": nav}))

    # Ensure a blank line between blocks, and a final newline at EOF
    text = "\n\n".join(blocks).rstrip() + "\n"
    mkdocs_path.write_text(text, encoding="utf-8")


def regenerate_nav(
    *,
    mkdocs_path: Path,
    api_dir: Path,
) -> None:
    """Load mkdocs.yml, rebuild nav from docs/api, and write back."""
    if not mkdocs_path.exists():
        raise FileNotFoundError(f"mkdocs.yml not found: {mkdocs_path}")

    data = yaml.safe_load(mkdocs_path.read_text(encoding="utf-8")) or {}
    old_nav = data.get("nav") or []

    # Keep non-generated nav items; remove old generated buckets
    base_nav = _strip_old_generated_sections(old_nav)
    base_nav = _ensure_basic_nav_entries(base_nav)

    # Build fresh sections from docs/api
    sections = build_sections_from_api_dir(api_dir)
    api_sections_nav = _sorted_top_sections(sections)

    # Final nav
    # Insert API sections right after "Examples" and before "For Developers" (if present).
    # This keeps the sidebar ordering:
    #   Home/Installation/Quickstart/Tutorials/Examples -> IO/Analysis/Utils/Workflows -> For Developers/...
    def _find_top_key(item):
        if isinstance(item, dict) and item:
            return next(iter(item.keys()))
        return None

    insert_after = "Examples"
    insert_before = "For Developers"

    after_idx = None
    before_idx = None
    for i, it in enumerate(base_nav):
        k = _find_top_key(it)
        if k == insert_after:
            after_idx = i
        if k == insert_before and before_idx is None:
            before_idx = i

    if before_idx is None:
        before_idx = len(base_nav)

    if after_idx is None:
        # If "Examples" isn't present, insert before "For Developers" (or append)
        ins = before_idx
    else:
        # Insert immediately after Examples, but never after For Developers.
        ins = min(after_idx + 1, before_idx)

    data["nav"] = base_nav[:ins] + api_sections_nav + base_nav[ins:]
    # Write with spacing
    write_mkdocs_yml_with_spacing(mkdocs_path, data)


# -------------------- CLI --------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate mkdocs.yml nav from docs/api Markdown structure."
    )
    parser.add_argument(
        "--mkdocs-yml",
        default=None,
        help="Path to mkdocs.yml (auto-detected if omitted).",
    )
    parser.add_argument(
        "--api-dir",
        default=None,
        help="Path to docs/api directory (auto-detected if omitted).",
    )
    args = parser.parse_args()

    auto_api_dir, auto_mkdocs = _auto_paths(Path(__file__))

    api_dir = Path(args.api_dir).resolve() if args.api_dir else auto_api_dir
    mkdocs_path = Path(args.mkdocs_yml).resolve() if args.mkdocs_yml else auto_mkdocs

    regenerate_nav(mkdocs_path=mkdocs_path, api_dir=api_dir)

    print(f"[Done] Updated nav in: {mkdocs_path}")
    print(f"       API scanned from: {api_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
