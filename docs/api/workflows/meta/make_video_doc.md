# Make Video Workflow

CLI namespace: `reaxkit make_video <task> [flags]`

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

## Available tasks

### `make`

#### Examples

- `reaxkit video make --folder reaxkit_outputs/elect/local_mu_3d --output reaxkit_outputs/video/local_mu_3D.mp4 --fps 5`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--folder FOLDER` | Folder containing image files (default: current directory) |
| `--output OUTPUT` | Output video filename |
| `--fps FPS` | Frames per second (default: 10) |
| `--ext EXT` | Comma-separated list of accepted image extensions (default: .png,.jpg,.jpeg) |
