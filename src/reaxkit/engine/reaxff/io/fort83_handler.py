"""Handler for extracting the optimized force field from ``fort.83``."""

from __future__ import annotations

from pathlib import Path


ERROR_FORCE_FIELD_MARKER = "Error force field"
_ERROR_FORCE_FIELD_MARKER_BYTES = ERROR_FORCE_FIELD_MARKER.encode("ascii")
DEFAULT_OPTIMIZED_FFIELD_NAME = "fort83_optimized_ffield"


class Fort83Handler:
    """Read a ReaxFF ``fort.83`` optimization history.

    The optimized force field is the block after the last line containing
    ``Error force field``.
    """

    def __init__(self, file_path: str | Path = "fort.83") -> None:
        self.path = Path(file_path)

    def get_optimized_ffield(self) -> str:
        """Return the force-field block following the last error marker."""
        return self._get_optimized_ffield_bytes().decode("utf-8")

    def _get_optimized_ffield_bytes(self) -> bytes:
        """Return the final force-field block without changing line endings."""
        lines = self.path.read_bytes().splitlines(keepends=True)

        for index in range(len(lines) - 1, -1, -1):
            if _ERROR_FORCE_FIELD_MARKER_BYTES in lines[index]:
                return b"".join(lines[index + 1 :])

        raise ValueError(
            f"'{ERROR_FORCE_FIELD_MARKER}' was not found in fort.83 file "
            f"'{self.path}'."
        )

    def write_optimized_ffield(
        self,
        output_path: str | Path = DEFAULT_OPTIMIZED_FFIELD_NAME,
    ) -> Path:
        """Write the optimized force field and return its output path."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self._get_optimized_ffield_bytes())
        return destination


def extract_optimized_ffield(
    fort83_path: str | Path = "fort.83",
    output_path: str | Path = DEFAULT_OPTIMIZED_FFIELD_NAME,
) -> Path:
    """Extract the final force field from ``fort83_path`` into ``output_path``."""
    return Fort83Handler(fort83_path).write_optimized_ffield(output_path)


__all__ = [
    "DEFAULT_OPTIMIZED_FFIELD_NAME",
    "ERROR_FORCE_FIELD_MARKER",
    "Fort83Handler",
    "extract_optimized_ffield",
]
