from pathlib import Path

DEFAULT_OUTROOT = Path("reaxkit_outputs")

def resolve_output_path(user_value: str, workflow: str) -> Path:
    """
    Put outputs under reaxkit_outputs/<workflow> by default,
    unless the user explicitly gave a directory.
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
