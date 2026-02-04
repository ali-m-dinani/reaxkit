"""
Image-to-video conversion utilities.

This module provides lightweight helpers for assembling a sequence of image
files into a video file, primarily for visualization of simulation snapshots,
plots, or frame-based outputs.

Typical use cases include:

- creating MP4 videos from sequentially saved plot images
- visualizing time evolution of simulation frames
- generating animations for presentations or reports
"""


import os
import re
import imageio


def _extract_numeric_index(filename: str) -> int:
    """
    Extract the first numeric index from a filename for sorting purposes.
    """
    nums = re.findall(r"\d+", filename)
    return int(nums[0]) if nums else -1


def images_to_video(
    folder_path: str,
    output_file: str = "output_video.mp4",
    fps: int = 10,
    ext: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> str:
    """
    Create a video from an ordered sequence of images in a folder.

    Images are collected from the specified directory, filtered by extension,
    sorted numerically based on the first number appearing in each filename,
    and encoded into a video file.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing image files.
    output_file : str, optional
        Output video filename, including extension (e.g., ``.mp4`` or ``.avi``).
    fps : int, optional
        Frames per second for the generated video.
    ext : tuple[str, ...], optional
        Accepted image file extensions.

    Returns
    -------
    str
        Absolute path to the saved video file.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist or contains no valid image files.

    Examples
    --------
    >>> images_to_video("frames/", output_file="traj.mp4", fps=15)
    """

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Collect image files
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(ext)
    ]
    if not files:
        raise FileNotFoundError(f"No image files with extensions {ext} in {folder_path}")

    # Sort numerically
    files.sort(key=_extract_numeric_index)

    # Read images
    images = [
        imageio.v2.imread(os.path.join(folder_path, f))
        for f in files
    ]

    # Save video
    imageio.mimsave(output_file, images, fps=fps)

    return os.path.abspath(output_file)
