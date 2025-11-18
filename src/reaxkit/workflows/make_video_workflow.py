# reaxkit/workflows/make_video_workflow.py

from __future__ import annotations
import argparse
from reaxkit.utils.make_video import images_to_video


def make_task(args: argparse.Namespace) -> int:
    """Create a video from images in a folder."""
    exts = tuple(e.strip() for e in args.ext.split(","))
    out_path = images_to_video(
        folder_path=args.folder,
        output_file=args.output,
        fps=args.fps,
        ext=exts,
    )
    print(f"[Done] Video saved to {out_path}")
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # video make: reaxkit video make --folder ./figs/local_pol3d --output local_polarization_3D.mp4 --fps 5
    p = subparsers.add_parser("make", help="Create a video from a sequence of image files")
    p.add_argument("--folder", default=".", help="Folder containing image files (default: current directory)")
    p.add_argument("--output", default="output_video.mp4", help="Output video filename (default: output_video.mp4)")
    p.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    p.add_argument("--ext", default=".png,.jpg,.jpeg",
                   help="Comma-separated list of accepted image extensions (default: .png,.jpg,.jpeg)",
    )
    p.set_defaults(_run=make_task)
