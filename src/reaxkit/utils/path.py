"""
Output path resolution utilities for ReaxKit workflows.

This module centralizes the logic used to determine where analysis results,
exports, and generated files are written. By default, outputs are placed
under a standardized directory structure:

    reaxkit_outputs/<workflow>/<filename>

If the user supplies an explicit path (absolute or containing directories),
that path is respected exactly.
"""


from pathlib import Path

DEFAULT_OUTROOT = Path("reaxkit_outputs")

def resolve_output_path(user_value: str, workflow: str) -> Path:
    """
    Resolve the output path for a workflow result.

    If the user provides only a bare filename, the file is written under
    ``reaxkit_outputs/<workflow>/``. If the user provides an absolute path
    or a path containing directories, that path is used directly.

    Parameters
    ----------
    user_value : str
        User-specified output path or filename.
    workflow : str
        Name of the workflow producing the output.

    Returns
    -------
    pathlib.Path
        Resolved output path with parent directories created if needed.

    Examples
    --------
    >>> resolve_output_path("results.csv", workflow="xmolout")
    >>> resolve_output_path("out/results.csv", workflow="fort7")
    """
    p = Path(user_value)

    # If user gave an absolute path or a relative path with dirs,
    # respect it exactly.
    if p.is_absolute() or p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # Otherwise, user gave just a bare filename -> use default tree
    outdir = DEFAULT_OUTROOT / workflow
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / p.name
