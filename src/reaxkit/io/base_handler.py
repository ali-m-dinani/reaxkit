"""
Base file-handler abstraction for ReaxKit.

This module defines the abstract ``FileHandler`` class, which provides the
common interface and lifecycle used by all ReaxKit file handlers
(e.g., ``XmoloutHandler``, ``Fort7Handler``, ``SummaryHandler``).

The base class standardizes how ReaxFF output files are:

- loaded from disk
- parsed lazily into structured tabular data
- exposed via a uniform DataFrame-based API
- accompanied by lightweight metadata

All ReaxKit analysis functions rely on ``FileHandler`` subclasses to provide
a consistent, predictable view of parsed ReaxFF files.
"""


from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import pandas as pd

class BaseHandler(ABC):
    """
    Abstract base class for ReaxKit file handlers.

    This class defines the minimal public interface that all ReaxKit
    file handlers must implement. Subclasses are responsible for parsing
    a specific ReaxFF file format and exposing its contents as structured
    pandas DataFrames.

    Parsed Data
    -----------
    Main table
        A pandas.DataFrame returned by ``dataframe()``, whose columns
        depend on the specific file type.

    Metadata
        A dictionary of lightweight metadata returned by ``metadata()``,
        typically including global or per-file attributes.

    Notes
    -----
    - Parsing is performed lazily and cached after the first access.
    - Subclasses must implement the private ``_parse()`` method.
    """

    def __init__(self, file_path: str | Path):
        """
        Initialize a file handler with a file path.

        Works on
        --------
        ReaxFF output files on disk

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the file to be parsed.

        Returns
        -------
        None
            Initializes the handler without parsing the file.
        """
        self.path = Path(file_path)
        self._parsed = False
        self._df: pd.DataFrame | None = None
        self._meta: dict[str, Any] = {}

    # ---- public API
    def parse(self) -> None:
        """
            Parse the file contents into structured data.

            Works on
            --------
            FileHandler — ReaxFF output file

            Returns
            -------
            None
                Parses the file and caches the resulting DataFrame and metadata.

            Examples
            --------
            >>> h = SomeHandler("file")
            >>> h.parse()
            """
        if not self._parsed:
            df, meta = self._parse()
            self._df = df
            self._meta = meta or {}
            self._parsed = True

    def dataframe(self) -> pd.DataFrame:
        """
            Return the parsed file contents as a pandas DataFrame.

            Works on
            --------
            FileHandler — ReaxFF output file

            Returns
            -------
            pandas.DataFrame
                Structured table representing the parsed file contents.

            Examples
            --------
            >>> h = SomeHandler("file")
            >>> df = h.dataframe()
            """
        if not self._parsed:
            self.parse()
        assert self._df is not None
        return self._df

    def metadata(self) -> dict[str, Any]:
        """
            Return parsed metadata associated with the file.

            Works on
            --------
            FileHandler — ReaxFF output file

            Returns
            -------
            dict
                Dictionary of metadata values extracted during parsing.

            Examples
            --------
            >>> h = SomeHandler("file")
            >>> meta = h.metadata()
            """
        if not self._parsed:
            self.parse()
        return dict(self._meta)

    # ---- subclasses must implement
    @abstractmethod
    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Parse the file and return structured data and metadata.

        Works on
        --------
        ReaxFF output files

        Parameters
        ----------
        None

        Returns
        -------
        tuple (pandas.DataFrame, dict)
            Parsed data table and associated metadata.

        Notes
        -----
        This method must be implemented by all subclasses."""
        ...
