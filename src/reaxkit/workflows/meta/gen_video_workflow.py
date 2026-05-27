"""Direct command workflow for generating videos from image sequences."""

from __future__ import annotations

import argparse

from reaxkit.presentation.movie.make_video import images_to_video

ALL_COMMANDS = ("gen-video",)
ALL_LEGACY_COMMANDS = ("gen_video", "video")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command="gen-video")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Create a video file from an ordered sequence of images in a folder.\n"
        "Use this command for post-processing analysis frames into a playable animation.\n"
        "It scans the target folder for matching image extensions and writes one output video.\n\n"
        "Examples:\n"
        "  1. Build an MP4 from analysis figures with custom FPS:\n"
        "   reaxkit gen-video --folder reaxkit_workspace/analysis/msd/run_xxx/figures "
        "--output reaxkit_workspace/analysis/msd/run_xxx/figures/msd.mp4 --fps 8\n\n"
        "  2. Build a video from current directory images using defaults:\n"
        "   reaxkit gen-video"
    )
    parser.add_argument(
        "--folder",
        default=".",
        help="Folder containing image files. Example: --folder results/figures, which tells the command where to collect frames.",
    )
    parser.add_argument(
        "--output",
        default="reaxkit_outputs/video/output_video.mp4",
        help="Output video filename. Example: --output movies/msd.mp4, which writes the generated video to that path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second. Example: --fps 8, which plays 8 frames each second for a slower animation than the default 10.",
    )
    parser.add_argument(
        "--ext",
        default=".png,.jpg,.jpeg",
        help="Comma-separated list of accepted image extensions. Example: --ext .png,.jpg, which limits frame discovery to PNG and JPG files.",
    )
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    _ = command
    exts = tuple(e.strip() for e in args.ext.split(","))
    out_path = images_to_video(
        folder_path=args.folder,
        output_file=args.output,
        fps=args.fps,
        ext=exts,
    )
    print(f"[Done] Video saved to {out_path}")
    return 0
