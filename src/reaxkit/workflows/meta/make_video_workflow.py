"""
Video generation workflow for ReaxKit.

This workflow provides utilities for creating video files (e.g. MP4)
from a sequence of image files stored in a directory.

It is primarily intended for post-processing and visualization of
ReaxKit outputs, such as:
- frame-by-frame 3D scatter plots,
- time-resolved heatmaps,
- or any other analysis that produces ordered image sequences.

The workflow wraps simple image-to-video functionality and exposes it
through a CLI task for reproducible and scriptable media generation.
"""


from __future__ import annotations
import argparse
from reaxkit.presentation.movie.make_video import images_to_video


def _make_task(args: argparse.Namespace) -> int:
    """
    Make task.

    Works on
    --------
    CLI workflow task arguments and helper utilities

    Parameters
    ----------
    args : argparse.Namespace
        Parameter description.

    Returns
    -------
    int
        Return value description.

    Examples
    --------
    >>>
    """
    exts = tuple(e.strip() for e in args.ext.split(","))
    out_path = images_to_video(
        folder_path=args.folder,
        output_file=args.output,
        fps=args.fps,
        ext=exts,
    )
    print(f"[Done] Video saved to {args.output}")
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Register `video` workflow task subcommands.
    """
    p = subparsers.add_parser(
        "make",
        help="Create a video from a sequence of image files",
        description=(
            "Create a video file from an ordered sequence of images in a folder.\n"
            "Use this command for post-processing figure frames into a playable animation.\n"
            "It scans the target folder for matching image extensions and writes one output video.\n\n"
            "Examples:\n"
            "  1. Build an MP4 from analysis figures with custom FPS:\n"
            "   reaxkit video make --folder reaxkit_workspace/analysis/msd/run_xxx/figures "
            "--output reaxkit_workspace/analysis/msd/run_xxx/figures/msd.mp4 --fps 8\n\n"
            "  2. Build a video from current directory images using defaults:\n"
            "   reaxkit video make\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--folder",
        default=".",
        help="Folder containing image files. Example: --folder results/figures, which tells the command where to collect frames.",
    )
    p.add_argument(
        "--output",
        default="reaxkit_outputs/video/output_video.mp4",
        help="Output video filename. Example: --output movies/msd.mp4, which writes the generated video to that path.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second. Example: --fps 8, which plays 8 frames each second for a slower animation than the default 10.",
    )
    p.add_argument(
        "--ext",
        default=".png,.jpg,.jpeg",
        help="Comma-separated list of accepted image extensions. Example: --ext .png,.jpg, which limits frame discovery to PNG and JPG files.",
    )
    p.set_defaults(_run=_make_task)
