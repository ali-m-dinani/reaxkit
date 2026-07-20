"""Tests for Jaguar isomer job creation."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from reaxkit.analysis.molecular_analysis import jaguar_isomer_jobs
from reaxkit.analysis.molecular_analysis.jaguar_isomer_jobs import (
    create_jaguar_isomer_jobs,
    load_slurm_jaguar_job_config,
    render_slurm_jaguar_script,
)
from reaxkit.workflows.file_tools import jaguar_isomer_workflow


def _write_isomer_inputs(root: Path, *, names: tuple[str, ...] = ("C8H13O3B5_0", "C8H13O3B5_1")) -> Path:
    isomer_dir = root / "isomers"
    for name in names:
        folder = isomer_dir / name
        folder.mkdir(parents=True)
        (folder / "xmolout").write_text(
            "\n".join(
                [
                    "C    0.00000    0.00000    0.00000",
                    "H    0.00000    0.00000    1.00000",
                    "&",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return isomer_dir


def _write_hf_base(root: Path) -> Path:
    hf_base = root / "hf_base.in"
    hf_base.write_text(
        "\n".join(
            [
                "&gen",
                "igeopt=1",
                "molchg=0",
                "multip=1",
                "&",
                "&zmat",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return hf_base


def _write_job_config(root: Path) -> Path:
    config = root / "slurm_job_config.yaml"
    config.write_text(
        "\n".join(
            [
                "job_system: slurm",
                "partition: debug",
                'time: "00:30:00"',
                "nodes: 1",
                "ntasks_per_node: 1",
                "cpus_per_task: 4",
                'mem: "2G"',
                "module_load:",
                "  - schrodinger",
                "environment:",
                "  SCHRODINGER: /opt/schrodinger",
                "pre_commands:",
                '  - echo "Preparing Jaguar"',
                "jaguar_command: jaguar",
                "jaguar_args:",
                "  - run",
                "  - hf.in",
                "  - -WAIT",
                "  - -PARALLEL",
                '  - "{cpus_per_task}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_load_slurm_jaguar_job_config_reads_yaml(tmp_path: Path) -> None:
    config = load_slurm_jaguar_job_config(_write_job_config(tmp_path))

    assert config.job_system == "slurm"
    assert config.partition == "debug"
    assert config.cpus_per_task == 4
    assert config.module_load == ["schrodinger"]
    assert config.environment == {"SCHRODINGER": "/opt/schrodinger"}
    assert config.jaguar_args[-1] == "{cpus_per_task}"


def test_render_slurm_jaguar_script_uses_config_values(tmp_path: Path) -> None:
    config = load_slurm_jaguar_job_config(_write_job_config(tmp_path))

    script = render_slurm_jaguar_script(structure_name="C8H13O3B5_0", config=config)

    assert "#SBATCH --partition=debug" in script
    assert "#SBATCH --time=00:30:00" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=2G" in script
    assert "#SBATCH --job-name=C8H13O3B5_0" in script
    assert "module load schrodinger" in script
    assert "export SCHRODINGER=/opt/schrodinger" in script
    assert "jaguar run hf.in -WAIT -PARALLEL 4" in script


def test_create_jaguar_isomer_jobs_writes_hf_input_scripts_and_manifest(tmp_path: Path) -> None:
    isomer_dir = _write_isomer_inputs(tmp_path)
    hf_base = _write_hf_base(tmp_path)
    config = _write_job_config(tmp_path)
    output_dir = tmp_path / "jaguar_jobs"

    result = create_jaguar_isomer_jobs(
        isomer_dir=isomer_dir,
        hf_base_path=hf_base,
        job_config_path=config,
        output_dir=output_dir,
    )

    assert len(result.records) == 2
    first_dir = output_dir / "C8H13O3B5_0"
    assert (first_dir / "xmolout").read_text(encoding="utf-8") == (
        isomer_dir / "C8H13O3B5_0" / "xmolout"
    ).read_text(encoding="utf-8")
    assert (first_dir / "hf.in").read_text(encoding="utf-8") == (
        hf_base.read_text(encoding="utf-8") + (isomer_dir / "C8H13O3B5_0" / "xmolout").read_text(encoding="utf-8")
    )
    assert "#SBATCH --partition=debug" in (first_dir / "jaguar.job").read_text(encoding="utf-8")
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert "structure_name,source_xmolout,job_dir,hf_input,submit_script,submitted,slurm_job_id,skipped,skip_reason" in manifest
    assert "C8H13O3B5_0" in manifest
    assert "submitted" in manifest
    assert result.warnings == [jaguar_isomer_jobs.JAGUAR_SETTINGS_WARNING]


def test_create_jaguar_isomer_jobs_validates_required_inputs(tmp_path: Path) -> None:
    isomer_dir = _write_isomer_inputs(tmp_path)
    hf_base = _write_hf_base(tmp_path)
    config = _write_job_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="isomer directory not found"):
        create_jaguar_isomer_jobs(
            isomer_dir=tmp_path / "missing_isomers",
            hf_base_path=hf_base,
            job_config_path=config,
            output_dir=tmp_path / "out",
        )

    with pytest.raises(FileNotFoundError, match="hf_base.in file not found"):
        create_jaguar_isomer_jobs(
            isomer_dir=isomer_dir,
            hf_base_path=tmp_path / "missing_hf_base.in",
            job_config_path=config,
            output_dir=tmp_path / "out",
        )

    with pytest.raises(FileNotFoundError, match="job config file not found"):
        create_jaguar_isomer_jobs(
            isomer_dir=isomer_dir,
            hf_base_path=hf_base,
            job_config_path=tmp_path / "missing_config.yaml",
            output_dir=tmp_path / "out",
        )


def test_create_jaguar_isomer_jobs_fails_when_isomer_subfolder_lacks_xmolout(tmp_path: Path) -> None:
    isomer_dir = tmp_path / "isomers"
    (isomer_dir / "C8H13O3B5_0").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="missing xmolout"):
        create_jaguar_isomer_jobs(
            isomer_dir=isomer_dir,
            hf_base_path=_write_hf_base(tmp_path),
            job_config_path=_write_job_config(tmp_path),
            output_dir=tmp_path / "out",
        )


def test_create_jaguar_isomer_jobs_refuses_nonempty_output_without_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "jaguar_jobs"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep me\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        create_jaguar_isomer_jobs(
            isomer_dir=_write_isomer_inputs(tmp_path),
            hf_base_path=_write_hf_base(tmp_path),
            job_config_path=_write_job_config(tmp_path),
            output_dir=output_dir,
        )


def test_create_jaguar_isomer_jobs_mocks_slurm_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def fake_run(command, *, cwd=None, capture_output, text, check):
        calls.append((list(command), str(cwd)))
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 12345\n", stderr="")

    monkeypatch.setattr(
        jaguar_isomer_jobs.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"squeue", "sbatch"} else None,
    )
    monkeypatch.setattr(jaguar_isomer_jobs.subprocess, "run", fake_run)

    result = create_jaguar_isomer_jobs(
        isomer_dir=_write_isomer_inputs(tmp_path, names=("C8H13O3B5_0",)),
        hf_base_path=_write_hf_base(tmp_path),
        job_config_path=_write_job_config(tmp_path),
        output_dir=tmp_path / "jaguar_jobs",
        submit=True,
    )

    assert calls == [
        (["squeue", "--name=C8H13O3B5_0", "--noheader"], "None"),
        (["sbatch", "jaguar.job"], str(tmp_path / "jaguar_jobs" / "C8H13O3B5_0")),
    ]
    assert result.records[0].submitted is True
    assert result.records[0].slurm_job_id == "12345"
    assert result.warnings == [jaguar_isomer_jobs.JAGUAR_SETTINGS_WARNING, jaguar_isomer_jobs.SUBMISSION_WARNING]


def test_create_jaguar_isomer_jobs_skips_completed_hf_output(tmp_path: Path) -> None:
    isomer_dir = _write_isomer_inputs(tmp_path, names=("C8H13O3B5_0",))
    (isomer_dir / "C8H13O3B5_0" / "hf.out").write_text("final geometry:\n", encoding="utf-8")

    result = create_jaguar_isomer_jobs(
        isomer_dir=isomer_dir,
        hf_base_path=_write_hf_base(tmp_path),
        job_config_path=_write_job_config(tmp_path),
        output_dir=tmp_path / "jaguar_jobs",
    )

    assert result.records[0].skipped is True
    assert result.records[0].skip_reason == "completed_hf_output"
    assert not (tmp_path / "jaguar_jobs" / "C8H13O3B5_0" / "hf.in").exists()
    assert "completed_hf_output" in result.manifest_path.read_text(encoding="utf-8")


def test_create_jaguar_isomer_jobs_skips_queued_slurm_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, capture_output, text, check):
        calls.append(list(command))
        return SimpleNamespace(returncode=0, stdout="12345 debug C8H13O3B5_0\n", stderr="")

    monkeypatch.setattr(
        jaguar_isomer_jobs.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"squeue", "sbatch"} else None,
    )
    monkeypatch.setattr(jaguar_isomer_jobs.subprocess, "run", fake_run)

    result = create_jaguar_isomer_jobs(
        isomer_dir=_write_isomer_inputs(tmp_path, names=("C8H13O3B5_0",)),
        hf_base_path=_write_hf_base(tmp_path),
        job_config_path=_write_job_config(tmp_path),
        output_dir=tmp_path / "jaguar_jobs",
        submit=True,
    )

    assert calls == [["squeue", "--name=C8H13O3B5_0", "--noheader"]]
    assert result.records[0].skipped is True
    assert result.records[0].skip_reason == "queued_slurm_job"
    assert result.records[0].submitted is False


def test_create_jaguar_isomer_jobs_submit_requires_sbatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jaguar_isomer_jobs.shutil, "which", lambda name: None)

    with pytest.raises(FileNotFoundError, match="sbatch command not found"):
        create_jaguar_isomer_jobs(
            isomer_dir=_write_isomer_inputs(tmp_path),
            hf_base_path=_write_hf_base(tmp_path),
            job_config_path=_write_job_config(tmp_path),
            output_dir=tmp_path / "jaguar_jobs",
            submit=True,
        )


def test_create_jaguar_isomer_jobs_rejects_unsupported_job_system(tmp_path: Path) -> None:
    config = _write_job_config(tmp_path)
    config.write_text(config.read_text(encoding="utf-8").replace("job_system: slurm", "job_system: pbs"), encoding="utf-8")

    with pytest.raises(ValueError, match="Only job_system='slurm'"):
        create_jaguar_isomer_jobs(
            isomer_dir=_write_isomer_inputs(tmp_path),
            hf_base_path=_write_hf_base(tmp_path),
            job_config_path=config,
            output_dir=tmp_path / "jaguar_jobs",
        )


def test_load_slurm_jaguar_job_config_rejects_scalar_list_fields(tmp_path: Path) -> None:
    config = _write_job_config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace("module_load:\n  - schrodinger", "module_load: schrodinger"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="module_load"):
        load_slurm_jaguar_job_config(config)


def test_jaguar_isomer_workflow_runs_file_generation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        isomer_dir=str(_write_isomer_inputs(tmp_path)),
        hf_base=str(_write_hf_base(tmp_path)),
        job_config=str(_write_job_config(tmp_path)),
        output_dir=str(tmp_path / "jaguar_jobs"),
        submit=False,
        force=False,
        no_skip_completed=False,
        no_skip_queued=False,
        write_example_config=None,
    )

    assert jaguar_isomer_workflow.run_main("create-jaguar-isomer-jobs", args) == 0

    captured = capsys.readouterr()
    assert "WARNING: hf_base.in contains Jaguar/DFT settings" in captured.out
    assert "create-jaguar-isomer-jobs: processed 2 isomers; generated 2 jobs; submitted 0; skipped 0" in captured.out


def test_jaguar_isomer_workflow_writes_example_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        isomer_dir=None,
        hf_base=None,
        job_config=None,
        output_dir="unused",
        submit=False,
        force=False,
        no_skip_completed=False,
        no_skip_queued=False,
        write_example_config=str(tmp_path / "slurm_job_config.yaml"),
    )

    assert jaguar_isomer_workflow.run_main("create-jaguar-isomer-jobs", args) == 0

    captured = capsys.readouterr()
    assert "wrote example Slurm/Jaguar config" in captured.out
    assert "job_system: slurm" in (tmp_path / "slurm_job_config.yaml").read_text(encoding="utf-8")
