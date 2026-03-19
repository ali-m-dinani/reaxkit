"""Management helpers for study relocation and cleanup."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from reaxkit.core.study.io import read_json, write_json
from reaxkit.core.study.naming import canonical_token, case_label_from_params_compact


def looks_like_abs_path(text: str) -> bool:
    s = str(text).strip()
    if not s:
        return False
    if s.startswith("/"):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", s):
        return True
    return False


def replace_path_prefix(value: str, old_prefix: str | None, new_prefix: str | None) -> str:
    if not old_prefix or new_prefix is None:
        return value
    if value == old_prefix:
        return new_prefix
    if value.startswith(old_prefix.rstrip("/\\") + "/") or value.startswith(old_prefix.rstrip("/\\") + "\\"):
        suffix = value[len(old_prefix.rstrip("/\\")) :]
        return new_prefix.rstrip("/\\") + suffix
    return value


def remap_study_internal_path(value: str, *, new_study_root: str | None, study_name: str | None) -> str:
    if new_study_root is None or not study_name:
        return value
    if not looks_like_abs_path(value):
        return value
    normalized = value.replace("\\", "/")
    token = f"/{study_name}/"
    rel_parts: list[str]
    idx = normalized.find(token)
    if idx >= 0:
        rel = normalized[idx + len(token) :].strip("/")
        rel_parts = [p for p in rel.split("/") if p]
    elif normalized.rstrip("/").endswith(f"/{study_name}"):
        rel_parts = []
    else:
        return value
    target = Path(new_study_root)
    if rel_parts:
        target = target.joinpath(*rel_parts)
    return str(target)


def rewrite_json_paths(
    obj: Any,
    *,
    old_source_yaml: str | None,
    new_source_yaml: str | None,
    old_study_root: str | None,
    new_study_root: str | None,
    old_template_dir: str | None,
    new_template_dir: str | None,
    study_name: str | None,
) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if key == "source_yaml":
                out[key] = new_source_yaml
                continue
            if key == "study_root":
                out[key] = new_study_root
                continue
            if key == "template_dir":
                out[key] = new_template_dir
                continue
            out[key] = rewrite_json_paths(
                v,
                old_source_yaml=old_source_yaml,
                new_source_yaml=new_source_yaml,
                old_study_root=old_study_root,
                new_study_root=new_study_root,
                old_template_dir=old_template_dir,
                new_template_dir=new_template_dir,
                study_name=study_name,
            )
        return out
    if isinstance(obj, list):
        return [
            rewrite_json_paths(
                v,
                old_source_yaml=old_source_yaml,
                new_source_yaml=new_source_yaml,
                old_study_root=old_study_root,
                new_study_root=new_study_root,
                old_template_dir=old_template_dir,
                new_template_dir=new_template_dir,
                study_name=study_name,
            )
            for v in obj
        ]
    if isinstance(obj, str):
        text = obj
        if looks_like_abs_path(text):
            text = replace_path_prefix(text, old_study_root, new_study_root)
            text = replace_path_prefix(text, old_source_yaml, new_source_yaml)
            text = replace_path_prefix(text, old_template_dir, new_template_dir)
            text = remap_study_internal_path(text, new_study_root=new_study_root, study_name=study_name)
        return text
    return obj


def rewrite_paths_by_map(obj: Any, path_map: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: rewrite_paths_by_map(v, path_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_paths_by_map(v, path_map) for v in obj]
    if isinstance(obj, str):
        text = obj
        for old, new in path_map.items():
            text = replace_path_prefix(text, old, new)
        return text
    return obj


def json_path_matches_filters(
    path: Path,
    *,
    case_filter: str | None,
    replicate_filter: str | None,
    replicate_variants_fn: Callable[[str | None], set[str]],
) -> bool:
    if not case_filter and not replicate_filter:
        return True
    text = str(path).replace("\\", "/")
    case_ok = True
    if case_filter:
        case_ok = canonical_token(case_filter) in canonical_token(text)
    rep_ok = True
    if replicate_filter:
        variants = replicate_variants_fn(replicate_filter)
        rep_ok = any(canonical_token(v) in canonical_token(text) for v in variants)
    return case_ok and rep_ok


def update_study_directory_paths(
    *,
    study_root: Path,
    case_filter: str | None = None,
    replicate_filter: str | None = None,
    dry_run: bool = False,
    prompt_with_default_fn: Callable[[str, str | None], str | None],
    replicate_variants_fn: Callable[[str | None], set[str]],
) -> dict[str, Any]:
    manifest_path = study_root / "study_manifest.json"
    manifest = read_json(manifest_path)
    study_name = str(manifest.get("study_name") or "").strip() or study_root.name
    old_source_yaml = str(manifest.get("source_yaml") or "").strip() or None
    old_study_root = str(manifest.get("study_root") or "").strip() or str(study_root.resolve())
    old_template_dir = str(manifest.get("template_dir") or "").strip() or None

    print("Update study path references (press Enter to keep current value).")
    new_source_yaml = prompt_with_default_fn("source_yaml", old_source_yaml)
    new_study_root_in = prompt_with_default_fn("study_root", str(study_root.resolve()))
    if not new_study_root_in:
        raise ValueError("study_root cannot be empty.")
    new_study_root = str(Path(new_study_root_in).resolve())
    new_template_dir = prompt_with_default_fn("template_dir", old_template_dir)

    json_files = sorted(study_root.rglob("*.json"))
    updated = 0
    scanned = 0
    for path in json_files:
        if not json_path_matches_filters(
            path,
            case_filter=case_filter,
            replicate_filter=replicate_filter,
            replicate_variants_fn=replicate_variants_fn,
        ):
            continue
        scanned += 1
        try:
            payload = read_json(path)
        except Exception:
            continue
        rewritten = rewrite_json_paths(
            payload,
            old_source_yaml=old_source_yaml,
            new_source_yaml=new_source_yaml,
            old_study_root=old_study_root,
            new_study_root=new_study_root,
            old_template_dir=old_template_dir,
            new_template_dir=new_template_dir,
            study_name=study_name,
        )
        if rewritten != payload:
            if not isinstance(rewritten, dict):
                continue
            if not dry_run:
                write_json(path, rewritten)
            updated += 1

    return {
        "study_root": new_study_root,
        "source_yaml": new_source_yaml,
        "template_dir": new_template_dir,
        "json_files_scanned": scanned,
        "json_files_updated": updated,
        "dry_run": bool(dry_run),
    }


def rename_case_directories(
    *,
    study_root: Path,
    case_filter: str | None = None,
    dry_run: bool = False,
    case_matches_selector_fn: Callable[[dict[str, Any], str | None], bool],
) -> dict[str, Any]:
    manifest_path = study_root / "study_manifest.json"
    manifest = read_json(manifest_path)
    cases = manifest.get("cases") or []
    if not isinstance(cases, list):
        raise ValueError("Invalid study_manifest.json: 'cases' must be a list.")

    rename_map: dict[str, str] = {}
    case_updates: list[dict[str, str]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        if not case_matches_selector_fn(case, case_filter):
            continue
        case_id = str(case.get("case_id") or "").strip()
        params = case.get("parameters") if isinstance(case.get("parameters"), dict) else {}
        old_path = str(case.get("path") or "").strip()
        if not case_id or not old_path:
            continue
        new_slug = case_label_from_params_compact(params)
        new_dir_name = f"{case_id}_{new_slug}" if new_slug else case_id
        old_dir = Path(old_path)
        new_dir = old_dir.parent / new_dir_name
        if str(old_dir) == str(new_dir):
            continue
        if new_dir.exists() and str(new_dir.resolve()) != str(old_dir.resolve()):
            raise FileExistsError(f"Target case directory already exists: {new_dir}")
        rename_map[str(old_dir)] = str(new_dir)
        case_updates.append({"case_id": case_id, "from": str(old_dir), "to": str(new_dir), "combo_slug": new_slug})

    if not case_updates:
        return {"renamed": [], "json_files_updated": 0, "dry_run": bool(dry_run)}

    for old, new in sorted(rename_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        old_p, new_p = Path(old), Path(new)
        if not old_p.exists():
            continue
        if not dry_run:
            old_p.rename(new_p)

    json_files = sorted(study_root.rglob("*.json"))
    json_updated = 0
    for path in json_files:
        try:
            payload = read_json(path)
        except Exception:
            continue
        rewritten = rewrite_paths_by_map(payload, rename_map)
        if isinstance(rewritten, dict) and rewritten != payload:
            if path == manifest_path and isinstance(rewritten.get("cases"), list):
                for entry in rewritten["cases"]:
                    if isinstance(entry, dict):
                        cid = str(entry.get("case_id") or "").strip()
                        for upd in case_updates:
                            if upd["case_id"] == cid:
                                entry["combo_slug"] = upd["combo_slug"]
                                break
            if not dry_run:
                write_json(path, rewritten)
            json_updated += 1

    return {
        "renamed": case_updates,
        "json_files_updated": json_updated,
        "dry_run": bool(dry_run),
    }

