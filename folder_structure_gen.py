from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable


def save_folder_structure(
    root_dir: str | Path,
    output_file: str | Path = "folder_structure.md",
    *,
    skipped_folders: Iterable[str] = None,
    include_files: bool = True,
    sort_entries: bool = True,
) -> None:
    """
    Walk through a directory and save its folder+file structure
    in a tree-like format.

    Key behavior:
      - If a folder is in `skipped_folders`, it and ALL its subfolders are skipped.
      - Default output is Markdown (.md) for better readability/rendering.
    """
    root = Path(root_dir).resolve()
    out = Path(output_file)

    skipped = set(skipped_folders)

    def _rel_depth(p: Path) -> int:
        rel = p.relative_to(root)
        return 0 if rel == Path(".") else len(rel.parts)

    with out.open("w", encoding="utf-8", newline="\n") as f:
        # Markdown renders this nicely; in txt it's still readable.
        if out.suffix.lower() in {".md", ".markdown"}:
            f.write(f"# Folder structure\n\nRoot: `{root}`\n\n```\n")
        else:
            f.write(f"Folder structure\nRoot: {root}\n\n")

        for current_path, dirs, files in os.walk(root):
            current = Path(current_path)

            # --- IMPORTANT PART: prune traversal so skipped folders' subtrees are not visited ---
            # Remove skipped dirs in-place so os.walk will not descend into them.
            dirs[:] = [d for d in dirs if d not in skipped]

            if sort_entries:
                dirs.sort()
                files.sort()

            depth = _rel_depth(current)
            indent = "    " * depth
            name = current.name if current != root else str(root)

            f.write(f"{indent}{name}/\n")

            if include_files:
                sub_indent = "    " * (depth + 1)
                for filename in files:
                    f.write(f"{sub_indent}{filename}\n")

        if out.suffix.lower() in {".md", ".markdown"}:
            f.write("```\n")

    print(f"[Done] Folder structure saved to: {out}")


if __name__ == "__main__":
    skipped_folders = ["__pycache__", "examples_to_test", "full_sim_examples", ".git",
                                      ".github", "trash", "under_dev", "webui", "reaxkit.egg-info"]
    save_folder_structure(".", "folder_structure.md", skipped_folders=skipped_folders)
