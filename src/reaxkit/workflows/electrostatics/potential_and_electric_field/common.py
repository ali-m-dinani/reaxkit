"""Shared documented CLI arguments for local ReaxFF electrostatics workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, add_storage_cli_arguments
from reaxkit.core.utils.frame_utils import parse_frame_indices


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Add input, frame-selection, electrostatics, and storage arguments."""
    parser.add_argument(
        "--engine", choices=["reaxff"], default="reaxff",
        help="Select the simulation-data adapter. Example: --engine reaxff, reads standalone ReaxFF trajectory and charge files.",
    )
    parser.add_argument(
        "--input", default=".",
        help="Set the path used for ReaxFF input discovery. Example: --input ./run, discovers simulation files under ./run.",
    )
    parser.add_argument(
        "--run-dir", default=".",
        help="Set the directory used to resolve relative input filenames. Example: --run-dir ./run, resolves ffield and energylog under ./run.",
    )
    parser.add_argument(
        "--fort7", default="fort.7",
        help="Select the file containing frame-aligned atomic charges. Example: --fort7 ./run/fort.7, reads charges from that file.",
    )
    parser.add_argument(
        "--xmolout", default="xmolout",
        help="Select the coordinate and atom-label trajectory. Example: --xmolout ./run/xmolout, evaluates fields at coordinates in that trajectory.",
    )
    parser.add_argument(
        "--summary", default=None,
        help="Select optional simulation and cell metadata. Example: --summary ./run/summary.txt, reads available iteration and cell information from that file.",
    )
    parser.add_argument(
        "--ffield", default="ffield",
        help="Select the force field containing taper radii and gammaEEM values. Example: --ffield ./run/ffield, uses electrostatic parameters from that file.",
    )
    parser.add_argument("--energylog", default=None,
                        help="Select the partial-energy log used to compare Ecoul values. Example: --energylog ./run/energylog, matches each selected iteration to its logged Ecoul value.")
    parser.add_argument(
        "--periodic", choices=["none", "x", "y", "z", "xy", "xz", "yz", "xyz"], default="xyz",
        help="Choose periodic lattice directions. Example: --periodic xy, includes periodic images along x and y while treating z as nonperiodic.",
    )
    parser.add_argument(
        "--cell-lengths", nargs=3, type=float, default=None,
        help="Override cell lengths in angstrom. Example: --cell-lengths 30 30 60, uses those lengths for every selected frame.",
    )
    parser.add_argument(
        "--cell-angles", nargs=3, type=float, default=(90.0, 90.0, 90.0),
        help="Set cell angles in degrees for --cell-lengths. Example: --cell-angles 90 90 120, defines a hexagonal cell geometry.",
    )
    parser.add_argument("--frames", nargs="*", default=None,
                        help="Select zero-based source frames or slices. Example: --frames ::20, evaluates every twentieth saved frame through the end of the trajectory.")
    parser.add_argument(
        "--every", type=int, default=1,
        help="Stride the selected frames again after --frames is applied. Example: --every 5, keeps every fifth frame from the current selection.",
    )
    parser.add_argument("--probe-elements", nargs="+", default=None,
                        help="Choose the hypothetical +1e probe species that supply gamma_i. Example: --probe-elements Al N, writes Al-probe, N-probe, and equal-average results.")
    parser.add_argument(
        "--field-method", choices=["analytic", "numerical"], default="analytic",
        help="Choose how the potential gradient is evaluated. Example: --field-method analytic, uses the exact derivative of the shielded tapered kernel.",
    )
    parser.add_argument(
        "--field-step", type=float, default=0.001,
        help="Set the Cartesian displacement in angstrom for numerical gradients. Example: --field-method numerical --field-step 0.0005, uses a centered difference with h=0.0005 angstrom.",
    )
    parser.add_argument("--disable-taper", action="store_true",
                        help="Disable tapering and cutoff filtering. Example: --disable-taper, uses Tap(r)=1 and every listed atom once with minimum-image periodic displacements.")
    parser.add_argument(
        "--log", choices=["verbose", "quiet"], default="quiet",
        help="Choose console logging detail. Example: --log verbose, prints detailed loading and calculation progress.",
    )
    add_storage_cli_arguments(parser)
    storage_help = {
        "run_id": (
            "Select an existing run identifier for workspace organization. "
            "Example: --run-id run_91ac0e, stores artifacts under that run identity."
        ),
        "project_root": (
            "Set the ReaxKit workspace root containing data and analysis directories. "
            "Example: --project-root ./reaxkit_workspace, writes canonical artifacts under that workspace."
        ),
        "analysis_id": (
            "Set the analysis artifact identifier instead of deriving it from the run. "
            "Example: --analysis-id field_profile_01, stores results under that analysis name."
        ),
    }
    for action in parser._actions:
        if action.dest in storage_help:
            action.help = storage_help[action.dest]


def resolve_input_path(args, value: str) -> Path:
    path = Path(value)
    if path.is_file(): return path.resolve()
    candidate = Path(args.run_dir) / path
    return candidate.resolve() if candidate.is_file() else path.resolve()


def request_kwargs(args) -> dict:
    from reaxkit.analysis.electrostatics.potential_and_electric_field.parameters import ReaxFFCoulombParameters
    parameters = ReaxFFCoulombParameters.from_ffield(resolve_input_path(args, args.ffield))
    raw_frames = list(args.frames or ())
    open_stride = None
    if len(raw_frames) == 1:
        token = str(raw_frames[0]).strip()
        parts = token.split(":")
        if len(parts) == 3 and parts[0] in {"", "0"} and parts[1] == "" and parts[2]:
            open_stride = int(parts[2])
            if open_stride < 1:
                raise ValueError("The open-ended frame-slice step must be positive.")
    return {
        "lower_taper_radius": parameters.lower_taper_radius,
        "upper_taper_radius": parameters.upper_taper_radius,
        "gamma_by_symbol": dict(parameters.gamma_by_symbol),
        "parameter_source": parameters.source,
        "probe_elements": tuple(args.probe_elements or ()),
        "periodic": tuple(axis in str(args.periodic) for axis in "xyz"),
        "cell_lengths": None if args.cell_lengths is None else tuple(args.cell_lengths),
        "cell_angles": tuple(args.cell_angles),
        "frames": None if open_stride is not None else parse_frame_indices(args.frames),
        "every": int(args.every) * (open_stride or 1), "field_method": str(args.field_method),
        "field_step": float(args.field_step), "disable_taper": bool(args.disable_taper),
    }


def runtime_arguments(args) -> dict:
    values = vars(args).copy(); values.update({"scope": "total", "cache": False, "_quick_charge_only": True})
    return values


def attach_energylog_reference(result, args) -> None:
    """Match Ecoul through ReaxKit's canonical partial-energy loader."""
    configured = Path(args.energylog) if args.energylog else Path(args.run_dir) / "energylog"
    if not configured.is_file():
        return
    import numpy as np
    from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
    partial = ReaxFFAdapter().load_partial_energy({"fort73": str(configured.resolve()),
                                                   "run_dir": str(Path(args.run_dir).resolve())})
    names = {str(name).casefold(): index for index, name in enumerate(partial.components)}
    component = next((names[key] for key in ("ecoul", "coul", "coulomb") if key in names), None)
    if component is None:
        raise ValueError(f"No Ecoul component was found in {configured}.")
    references = {int(iteration): float(value) for iteration, value in
                  zip(partial.iterations, partial.values[:, component], strict=False)}
    matched = result.totals["iter"].map(references)
    if matched.isna().any():
        missing = result.totals.loc[matched.isna(), "iter"].astype(int).tolist()
        raise ValueError(f"Energylog contains no Coulomb value for iteration(s): {missing}.")
    result.totals["energylog_coulomb (kcal/mol)"] = matched.to_numpy(dtype=float)
    reference = matched.to_numpy(dtype=float)
    calculated = result.totals["coulomb (kcal/mol)"].to_numpy(dtype=float)
    result.totals["relative_difference_percent"] = np.divide(
        calculated - reference, np.abs(reference),
        out=np.full_like(reference, np.nan), where=reference != 0.0,
    ) * 100.0


def artifact_directory(args, command: str) -> Path:
    if getattr(args, "output_dir", None) is not None: return Path(args.output_dir).resolve()
    analysis_id = args.analysis_id or args.run_id or getattr(args, "_analysis_id", None) or "analysis"
    return ReaxkitStorageLayout(project_root=Path(args.project_root)).analysis_root / command / str(analysis_id)


__all__ = ["add_input_arguments", "artifact_directory", "attach_energylog_reference",
           "request_kwargs", "runtime_arguments"]
