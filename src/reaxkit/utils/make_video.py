"""utility used to make videos out of images"""

import os
import re
import imageio


def _extract_numeric_index(filename: str) -> int:
    """Extract first numeric occurrence from filename for sorting."""
    nums = re.findall(r"\d+", filename)
    return int(nums[0]) if nums else -1


def images_to_video(
    folder_path: str,
    output_file: str = "output_video.mp4",
    fps: int = 10,
    ext: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> str:
    """
    Create a video from image sequence in a folder.

    Parameters
    ----------
    folder_path : str
        Directory containing image files.
    output_file : str
        Output video filename (with .mp4/.avi).
    fps : int
        Frame rate.
    ext : tuple[str, ...]
        Accepted image extensions.

    Returns
    -------
    str
        Path to saved video file.

    Raises
    ------
    FileNotFoundError
        If folder doesn't exist or no valid images found.
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
