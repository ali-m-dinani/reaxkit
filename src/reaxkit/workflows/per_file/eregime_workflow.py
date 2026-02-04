"""
Electric-field regime (eregime) workflow for ReaxKit.

This workflow provides tools to read, analyze, visualize, and generate
`eregime.in` files, which define time- or iteration-dependent electric-field
schedules in ReaxFF simulations.

It supports:
- Extracting electric-field components (e.g. E1, E2) versus iteration, frame,
  or physical time, with optional plotting and CSV export.
- Selecting subsets of data by frame ranges for focused analysis.
- Generating new `eregime.in` files using standard field profiles, including
  sinusoidal waves, smooth pulses, or user-defined analytic functions.

The workflow is designed to bridge ReaxFF electric-field protocols with
analysis and visualization, enabling reproducible setup and interpretation
of field-driven simulations.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Callable, Dict, Any
from reaxkit.utils.units import unit_for

from reaxkit.io.handlers.eregime_handler import EregimeHandler
from reaxkit.analysis.per_file.eregime_analyzer import get_eregime_data
from reaxkit.utils.media.plotter import single_plot
from reaxkit.io.generators.eregime_generator import (
    write_eregime_sinusoidal,
    write_eregime_smooth_pulse,
    write_eregime_from_function,
)
from reaxkit.utils.alias import normalize_choice
from reaxkit.utils.frame_utils import parse_frames, select_frames
from reaxkit.utils.path import resolve_output_path


# ====================================================================
#                           GET (read/plot/export)
# ====================================================================

def _eregime_get_task(args: argparse.Namespace) -> int:
    """
    Parse eregime.in and return Y vs chosen X axis (iter/frame/time), with optional plotting/export.
    """
    handler = EregimeHandler(args.file)

    # Build series with alias-aware resolver and x-axis conversion (iter/frame/time)
    df = get_eregime_data(
        handler,
        y=args.column,
        xaxis=args.xaxis,
        control_file=args.control,  # used when xaxis='time'
    )  # -> DataFrame with columns [<x_label>, <y (as requested)>]

    # Optional row selection via frames (position-based)
    frames = parse_frames(args.frames)
    df = select_frames(df, frames)

    x_label = df.columns[0]
    y_label = df.columns[1]
    canonical_label = normalize_choice(args.column)  # resolve like E1 → field1
    unit = unit_for(canonical_label) or unit_for(y_label) or ""
    title = f"{canonical_label} vs {x_label}"

    workflow_name = args.kind
    # Export CSV
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[Done] Exported data to {out}")

    # Save static plot
    if args.save:
        out = resolve_output_path(args.save, workflow_name)
        single_plot(
            df[x_label], df[y_label],
            title=title, xlabel=x_label, ylabel=f"{canonical_label} ({unit})", save=out
        )

    # Show interactive plot
    if args.plot:
        single_plot(
            df[x_label], df[y_label],
            title=title, xlabel=x_label, ylabel=f"{canonical_label} ({unit})", save=None
        )

    # Guidance if no action flags
    if not (args.export or args.save or args.plot):
        print("ℹ️ No action chosen. Add one or more of: --plot, --save PATH, --export PATH")
        print(f"Available columns in file: {', '.join(df.columns)}")

    return 0

# ====================================================================
#                       GENERATORS (using official API)
# ====================================================================

def _task_gen_sin(args: argparse.Namespace) -> int:
    write_eregime_sinusoidal(
        args.output,
        max_magnitude=args.max_magnitude,
        step_angle=args.step_angle,
        iteration_step=args.iteration_step,
        num_cycles=args.num_cycles,
        direction=args.direction,
        voltage_idx=args.V,
        phase=args.phase,
        dc_offset=args.dc_offset,
        start_iter=args.start_iter,
    )
    print(f"       This file is saved in {Path(args.output).resolve()}")
    return 0


def _task_gen_pulse(args: argparse.Namespace) -> int:
    write_eregime_smooth_pulse(
        args.output,
        amplitude=args.amplitude,
        width=args.width,
        period=args.period,
        slope=args.slope,
        iteration_step=args.iteration_step,
        num_of_cycles=args.num_cycles,
        step_size=args.step_size,
        direction=args.direction,
        voltage_idx=args.V,
        baseline=args.baseline,
        start_iter=args.start_iter,
    )
    print(f"       This file is saved in {Path(args.output).resolve()}")
    return 0


def _safe_build_func(expr: str) -> Callable[[float], float]:
    """
    Build func(t) from a simple expression like '0.003*sin(2*pi*t/100) + 0.0005'.
    Allowed names: from math + numpy as (optional) 'np'.
    """
    import math, numpy as np
    allowed: Dict[str, Any] = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["np"] = np
    code = compile(expr, "<func-expr>", "eval")

    def f(t: float) -> float:
        return float(eval(code, {"__builtins__": {}}, {**allowed, "t": float(t)}))

    return f


def _task_gen_func(args: argparse.Namespace) -> int:
    func = _safe_build_func(args.expr)
    write_eregime_from_function(
        args.output,
        func=func,
        t_end=args.t_end,
        dt=args.dt,
        iteration_step=args.iteration_step,
        direction=args.direction,
        voltage_idx=args.V,
        start_iter=args.start_iter,
    )
    print(f"       This file is saved in {Path(args.output).resolve()}")
    return 0


# ====================================================================
#                       CLI WIRING (single entry)
# ====================================================================

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Wire subcommands under the *existing* 'eregime' command provided by the top-level CLI.
    """
    # ---- 'get' ----
    p_get = subparsers.add_parser(
        "get",
        help="Get a column (like E1) vs iter/frame/time and optionally plot, save, or export it.\n",
        description=(
            "Examples:\n"
            "  reaxkit eregime get --column E1 --xaxis time --export eregime_E1_vs_time.csv "
            "--save eregime_E1_vs_time.png\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_get.add_argument("--file", default="eregime.in", help="Path to eregime.in file")
    p_get.add_argument("--column", required=True,
        help="Y column to extract (aliases supported, e.g., E, E1, E2, direction, direction1, direction2)"
    )
    p_get.add_argument("--xaxis", default="iter", choices=("iter", "frame", "time"),
        help="X axis for the output: 'iter' (default), 'frame', or 'time' (needs --control)"
    )
    p_get.add_argument("--control", default="control",
        help="Control file used to convert iteration → time when --xaxis time (default: control)"
    )
    p_get.add_argument("--frames", default=None,
        help="Row selector (position-based): 'start:stop[:step]' or 'i,j,k' (default: all rows)"
    )
    p_get.add_argument("--plot", action="store_true", help="Show the plot interactively.")
    p_get.add_argument("--save", default=None, help="Save the plot image to this path.")
    p_get.add_argument("--export", default=None, help="Export data CSV to this path.")
    p_get.set_defaults(_run=_eregime_get_task)


    # ---- 'gen' parent ----
    p_gen = subparsers.add_parser(
        "gen",
        help="Generate eregime.in via official generators (sin, pulse, func).\n",
        description=(
            "Examples:\n"
            "  reaxkit eregime gen sin --output eregime_sin.in --max-magnitude 0.004 --step-angle 0.05 "
            "--iteration-step 500 --num-cycles 2 --direction z --V 1\n"
            "  reaxkit eregime gen pulse --output eregime_pulse.in --amplitude 0.003 --width 50 "
            "--period 200 --slope 20 --iteration-step 250 --num-cycles 5 --direction z --V 1\n"
            "  reaxkit eregime gen func --output eregime_func.in --expr '0.003*cos(2*pi*t/100)' "
            "--t-end 1000 --dt 1 --iteration-step 250 --direction z --V 1"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    gen = p_gen.add_subparsers(dest="gen_cmd", required=True)

    # gen sin (make_eregime_sinusoidal)
    p_sin = gen.add_parser("sin")
    p_sin.add_argument("--output", default="eregime.in", help="Output file path.")
    p_sin.add_argument("--max-magnitude", type=float, required=True, help="Peak amplitude (V/Å).")
    p_sin.add_argument("--step-angle", type=float, required=True, help="Sampling step (radians).")
    p_sin.add_argument("--iteration-step", type=int, required=True, help="Iterations between samples.")
    p_sin.add_argument("--num-cycles", type=float, required=True, help="Number of cycles.")
    p_sin.add_argument("--direction", default="z", help="x|y|z")
    p_sin.add_argument("--V", type=int, default=1, help="Voltage index (integer).")
    p_sin.add_argument("--phase", type=float, default=0.0, help="Phase offset (radians).")
    p_sin.add_argument("--dc-offset", type=float, default=0.0, help="DC offset (V/Å).")
    p_sin.add_argument("--start-iter", type=int, default=0, help="Starting iteration.")
    p_sin.set_defaults(_run=_task_gen_sin)

    # gen pulse (make_eregime_smooth_pulse)
    p_pulse = gen.add_parser("pulse")
    p_pulse.add_argument("--output", default="eregime.in", help="Output file path.")
    p_pulse.add_argument("--amplitude", type=float, required=True, help="Peak amplitude (V/Å).")
    p_pulse.add_argument("--width", type=float, required=True, help="Flat-top width (time units).")
    p_pulse.add_argument("--period", type=float, required=True, help="Full-cycle period (time units).")
    p_pulse.add_argument("--slope", type=float, required=True, help="Ramp duration up/down (time units).")
    p_pulse.add_argument("--iteration-step", type=int, required=True, help="Iterations per sample.")
    p_pulse.add_argument("--num-cycles", type=float, required=True, help="Number of cycles.")
    p_pulse.add_argument("--step-size", type=float, default=0.1, help="Temporal resolution.")
    p_pulse.add_argument("--direction", default="z", help="x|y|z")
    p_pulse.add_argument("--V", type=int, default=1, help="Voltage index (integer).")
    p_pulse.add_argument("--baseline", type=float, default=0.0, help="Baseline (V/Å).")
    p_pulse.add_argument("--start-iter", type=int, default=0, help="Starting iteration.")
    p_pulse.set_defaults(_run=_task_gen_pulse)

    # gen func (make_eregime_from_function)
    p_func = gen.add_parser("func")
    p_func.add_argument("--output", default="eregime.in", help="Output file path.")
    p_func.add_argument(
        "--expr", required=True,
        help="Python expression in t (math + np available), e.g. '0.003*sin(2*pi*t/100) + 0.0005'"
    )
    p_func.add_argument("--t-end", type=float, required=True, help="End time.")
    p_func.add_argument("--dt", type=float, required=True, help="Time step.")
    p_func.add_argument("--iteration-step", type=int, required=True, help="Iterations per sample.")
    p_func.add_argument("--direction", default="z", help="x|y|z")
    p_func.add_argument("--V", type=int, default=1, help="Voltage index (integer).")
    p_func.add_argument("--start-iter", type=int, default=0, help="Starting iteration.")
    p_func.set_defaults(_run=_task_gen_func)
