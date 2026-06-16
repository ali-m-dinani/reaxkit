from __future__ import annotations

from pathlib import Path

from reaxkit.core.runtime.generator_runtime import maybe_copy_output_to_dot


def test_maybe_copy_output_to_dot_copies_file_to_current_directory(
    tmp_path: Path,
    monkeypatch,
):
    run_dir = tmp_path / "reaxkit_workspace" / "inputs" / "run_1"
    run_dir.mkdir(parents=True)
    source = run_dir / "geo"
    source.write_text("geo content\n", encoding="utf-8")
    cwd = tmp_path / "full_sim_examples" / "kyle_space"
    cwd.mkdir(parents=True)
    parent = cwd.parent
    monkeypatch.chdir(cwd)

    copied = maybe_copy_output_to_dot(source, enabled=True)

    assert copied == cwd / "geo"
    assert (cwd / "geo").read_text(encoding="utf-8") == "geo content\n"
    assert not (parent / "geo").exists()


def test_maybe_copy_output_to_dot_copies_directory_to_current_directory(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "reaxkit_workspace" / "inputs" / "run_1" / "bundle"
    source.mkdir(parents=True)
    (source / "item.txt").write_text("content\n", encoding="utf-8")
    cwd = tmp_path / "full_sim_examples" / "kyle_space"
    cwd.mkdir(parents=True)
    parent = cwd.parent
    monkeypatch.chdir(cwd)

    copied = maybe_copy_output_to_dot(source, enabled=True)

    assert copied == cwd / "bundle"
    assert (cwd / "bundle" / "item.txt").read_text(encoding="utf-8") == "content\n"
    assert not (parent / "bundle").exists()
