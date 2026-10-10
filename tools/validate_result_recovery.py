"""Compare baseline, checkpointed and re-exported h-BN CLI sample results."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import psutil


def execute(arguments, log):
    started = time.perf_counter()
    peak = 0
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, "-c", "from reaxkit.cli_startup import main; raise SystemExit(main())", *arguments],
                                   stdout=stream, stderr=subprocess.STDOUT)
        observed = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [observed, *observed.children(recursive=True)]
                peak = max(peak, sum(child.memory_info().rss for child in processes if child.is_running()))
            except psutil.NoSuchProcess:
                pass
            try:
                process.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                pass
    if process.returncode:
        raise RuntimeError(f"CLI failed with exit {process.returncode}; see {log}")
    return {"wall_seconds": time.perf_counter() - started, "peak_rss_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--engine", choices=["reaxff", "ams"], required=True)
    parser.add_argument("--charge-source", choices=["formal", "reaxff"], default="formal")
    parser.add_argument("--frames", default="0:3")
    parser.add_argument("--replication", nargs=3, default=[19, 19, 10], type=int)
    parser.add_argument("--command", default="get-hbn-reference-polarization",
                        choices=["get-hbn-reference-polarization", "get-hbn-reference-local-polarization", "get-hbn-reference-projected-polarity"])
    args = parser.parse_args()
    source = args.input_dir.resolve()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    common = [args.command, "--engine", args.engine, "--input", str(source / "reaxout.kf" if args.engine == "ams" else source),
              "--run-dir", str(source),
              "--replication", *map(str, args.replication), "--frames", args.frames, "--charge-source", args.charge_source,
              "--volume-method", "cell", "--output-profile", "minimal", "--execution", "serial"]
    if args.engine == "reaxff":
        common.extend(["--xmolout", str(source / "xmolout")])
        if args.charge_source == "reaxff":
            common.extend(["--fort7", str(source / "fort.7")])
    report = {"engine": args.engine, "frames": args.frames, "command": args.command, "charge_source": args.charge_source}
    for mode in ("baseline", "checkpoint"):
        directory = workspace / mode
        flags = ["--no-checkpoint"] if mode == "baseline" else ["--checkpoint", "--checkpoint-buffer-mb", ".1"]
        print(f"Running {args.engine} {args.charge_source} {mode}", flush=True)
        report[mode] = execute([*common, "--project-root", str(directory), *flags], workspace / f"{mode}.log")
    manifests = list((workspace / "checkpoint" / "analysis").rglob("checkpoint/manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Use a fresh validation workspace containing exactly one checkpoint run.")
    checkpoint = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    assert manifest["state"] == "complete"
    report["committed_frames"] = manifest["frame_count"]
    report["checkpoint_bytes"] = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    print(f"Resuming completed {args.engine} {args.charge_source} checkpoint", flush=True)
    report["resume"] = execute([*common, "--project-root", str(workspace / "resumed"), "--resume", str(checkpoint)], workspace / "resume.log")
    baseline_tables = list((workspace / "baseline" / "analysis" / args.command).rglob("*.csv"))
    compared = []
    for path in baseline_tables:
        expected = pd.read_csv(path)
        for mode in ("checkpoint", "resumed"):
            matches = list((workspace / mode / "analysis" / args.command).rglob(path.name))
            assert len(matches) == 1, (mode, path.name, matches)
            pd.testing.assert_frame_equal(pd.read_csv(matches[0]), expected, rtol=1e-10, atol=1e-10)
        compared.append(path.name)
    assert compared
    report["compared_csvs"] = compared
    (workspace / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
