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
from reaxkit.utils.media.make_video import images_to_video


def _make_task(args: argparse.Namespace) -> int:
    """Create a video from images in a folder."""
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
    Register coordination-related CLI subcommands.

    This function defines the task-level interface for the
    `reaxkit file` workflow and attaches its tasks (i.e., subcommands).

    Each task may share common input arguments and defines task-specific options as needed.
    """
    p = subparsers.add_parser(
        "make",
        help="Create a video from a sequence of image files",
        description=(
            "Examples:\n"
            "  reaxkit video make --folder reaxkit_outputs/elect/local_mu_3d "
            "--output reaxkit_outputs/video/local_mu_3D.mp4 --fps 5\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--folder", default=".", help="Folder containing image files (default: current directory)")
    p.add_argument("--output", default="reaxkit_outputs/video/output_video.mp4", help="Output video filename")
    p.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    p.add_argument("--ext", default=".png,.jpg,.jpeg",
                   help="Comma-separated list of accepted image extensions (default: .png,.jpg,.jpeg)",
    )
    p.set_defaults(_run=_make_task)
