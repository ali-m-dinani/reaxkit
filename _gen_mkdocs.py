"""Generate `mkdocs.yml` navigation from the current docs tree.

Keep this script outside `docs/scripts/` so it is not auto-executed by
`docs/hooks.py` during `mkdocs serve`.
"""

from __future__ import annotations

from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
MKDOCS_FILE = ROOT / "mkdocs.yml"

API_SECTIONS = ("analysis", "engine", "utils", "workflows")
OVERVIEW_LABEL = "📍 Overview"

TITLE_OVERRIDES = {
    "ams": "AMS",
    "lammps": "LAMMPS",
    "msd": "MSD",
    "rdf": "RDF",
    "fft": "FFT",
    "mm": "MM",
    "io": "IO",
    "gui": "GUI",
    "cli": "CLI",
    "api": "API",
    "reaxff": "ReaxFF",
    "xmolout": "XMolout",
}

DIR_TITLE_OVERRIDES = {
    "active_sites": "Active Sites",
    "force_field": "Force Field",
    "molecular_analysis": "Molecular Analysis",
    "file_tools": "File Tools",
    "study_design": "Study Design",
    "rules_and_conventions": "Rules and Conventions",
    "for_developers": "For Developers",
    "getting_started": "Getting Started",
}


def to_title(raw: str, *, strip_kind_suffix: bool = True) -> str:
    """Convert file/dir slugs to display titles."""
    key = raw.strip().replace("-", "_")
    if not key:
        return ""

    if strip_kind_suffix:
        key = re.sub(r"_(analysis|workflow|doc)$", "", key, flags=re.IGNORECASE)
    if key in DIR_TITLE_OVERRIDES:
        return DIR_TITLE_OVERRIDES[key]

    words = [w for w in key.split("_") if w]
    titled: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in TITLE_OVERRIDES:
            titled.append(TITLE_OVERRIDES[lower])
        elif word.isupper() and len(word) <= 5:
            titled.append(word)
        else:
            titled.append(word.capitalize())
    return " ".join(titled)


def rel_docs(path: Path) -> str:
    """Return POSIX path relative to docs dir."""
    return path.relative_to(DOCS_DIR).as_posix()


def sorted_md_files(directory: Path) -> list[Path]:
    """Return markdown files with `index.md` first, then alphabetical."""
    files = [p for p in directory.glob("*.md") if p.is_file()]
    index = [p for p in files if p.name.lower() == "index.md"]
    others = sorted([p for p in files if p.name.lower() != "index.md"], key=lambda p: p.name.lower())
    return index + others


def build_dir_nav(directory: Path) -> list[dict[str, object]]:
    """Build nested nav entries for a directory."""
    entries: list[dict[str, object]] = []

    for md_file in sorted_md_files(directory):
        label = OVERVIEW_LABEL if md_file.name.lower() == "index.md" else to_title(md_file.stem, strip_kind_suffix=True)
        entries.append({label: rel_docs(md_file)})

    subdirs = sorted([d for d in directory.iterdir() if d.is_dir()], key=lambda d: d.name.lower())
    for subdir in subdirs:
        sub_entries = build_dir_nav(subdir)
        if sub_entries:
            entries.append({to_title(subdir.name, strip_kind_suffix=False): sub_entries})

    return entries


def build_api_nav() -> list[dict[str, object]]:
    """Build API nav from docs/api directory."""
    api_dir = DOCS_DIR / "api"
    items: list[dict[str, object]] = []

    api_index = api_dir / "index.md"
    if api_index.is_file():
        items.append({OVERVIEW_LABEL: rel_docs(api_index)})

    for section in API_SECTIONS:
        section_dir = api_dir / section
        if section_dir.is_dir():
            items.append({to_title(section, strip_kind_suffix=False): build_dir_nav(section_dir)})

    return items


def build_tutorials_nav() -> list[dict[str, object]]:
    """Build tutorials nav including overview plus each tutorial page."""
    tutorials_dir = DOCS_DIR / "tutorials"
    if not tutorials_dir.is_dir():
        return []

    entries: list[dict[str, object]] = []
    for md_file in sorted_md_files(tutorials_dir):
        if md_file.name.lower() == "index.md":
            entries.append({OVERVIEW_LABEL: rel_docs(md_file)})
            continue

        stem = md_file.stem
        match = re.match(r"^(\d+)[_-](.+)$", stem)
        if match:
            order = match.group(1)
            title = to_title(match.group(2), strip_kind_suffix=False)
            label = f"{order} - {title}"
        else:
            label = to_title(stem, strip_kind_suffix=False)
        entries.append({label: rel_docs(md_file)})
    return entries


def build_rules_nav() -> list[dict[str, object]]:
    """Build rules-and-conventions nav from docs/rules_and_conventions."""
    rules_dir = DOCS_DIR / "rules_and_conventions"
    if not rules_dir.is_dir():
        return []
    return build_dir_nav(rules_dir)


def build_resources_nav() -> list[dict[str, object]]:
    """Build resources nav from docs/resources."""
    resources_dir = DOCS_DIR / "resources"
    if not resources_dir.is_dir():
        return []
    return build_dir_nav(resources_dir)


def build_nav() -> list[dict[str, object]]:
    """Build full mkdocs nav structure."""
    nav: list[dict[str, object]] = []

    if (DOCS_DIR / "index.md").is_file():
        nav.append({"Home": "index.md"})

    getting_started: list[dict[str, object]] = []
    if (DOCS_DIR / "getting_started/index.md").is_file():
        getting_started.append({OVERVIEW_LABEL: "getting_started/index.md"})
    if (DOCS_DIR / "installation.md").is_file():
        getting_started.append({"Installation": "installation.md"})
    if (DOCS_DIR / "quickstart.md").is_file():
        getting_started.append({"Quickstart": "quickstart.md"})
    tutorials_nav = build_tutorials_nav()
    if tutorials_nav:
        getting_started.append({"Tutorials": tutorials_nav})
    if (DOCS_DIR / "examples/README.md").is_file():
        getting_started.append({"Examples": "examples/README.md"})
    if getting_started:
        nav.append({"Getting Started": getting_started})

    api_nav = build_api_nav()
    if api_nav:
        nav.append({"API": api_nav})

    dev_items: list[dict[str, object]] = []
    if (DOCS_DIR / "for_developers/index.md").is_file():
        dev_items.append({OVERVIEW_LABEL: "for_developers/index.md"})
    if (DOCS_DIR / "contributing.md").is_file():
        dev_items.append({"Contributing": "contributing.md"})
    if (DOCS_DIR / "file_templates/index.md").is_file():
        dev_items.append({"File Templates": "file_templates/index.md"})
    rules_nav = build_rules_nav()
    if rules_nav:
        dev_items.append({"Rules and Conventions": rules_nav})
    if dev_items:
        nav.append({"For Developers": dev_items})

    resources_nav = build_resources_nav()
    if resources_nav:
        nav.append({"Resources": resources_nav})

    return nav


def dump_nav(nav: list[dict[str, object]]) -> str:
    """Render nav as YAML block (without top-level `nav:` key)."""
    dumped = yaml.safe_dump(nav, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return dumped.rstrip()


def replace_nav_block(original: str, nav_block: str) -> str:
    """Replace the `nav:` block in mkdocs.yml."""
    lines = original.splitlines()
    nav_start = None
    for i, line in enumerate(lines):
        if line.strip() == "nav:" and not line.startswith((" ", "\t")):
            nav_start = i
            break
    if nav_start is None:
        raise RuntimeError("Could not find top-level `nav:` in mkdocs.yml")

    nav_end = len(lines)
    top_key_pattern = re.compile(r"^[A-Za-z0-9_\"'.-]+:\s*$")
    for i in range(nav_start + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        if line.startswith((" ", "\t")):
            continue
        if top_key_pattern.match(line):
            nav_end = i
            break

    new_lines = lines[:nav_start] + ["nav:"] + nav_block.splitlines() + lines[nav_end:]
    return "\n".join(new_lines).rstrip() + "\n"


def main() -> None:
    nav = build_nav()
    nav_block = dump_nav(nav)
    original = MKDOCS_FILE.read_text(encoding="utf-8")
    updated = replace_nav_block(original, nav_block)
    MKDOCS_FILE.write_text(updated, encoding="utf-8")
    print(f"Updated {MKDOCS_FILE} with {len(nav)} top-level nav entries.")


if __name__ == "__main__":
    main()
