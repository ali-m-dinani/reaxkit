"""
Custom exception types used throughout ReaxKit.

This module defines lightweight, domain-specific exceptions that allow
ReaxKit workflows to distinguish between parsing failures and analysis
failures, enabling clearer error reporting and recovery.
"""

class ParseError(Exception):
    """
    Error raised when a ReaxFF file cannot be parsed correctly.

    This exception is intended for failures occurring during file reading,
    tokenization, or structural interpretation of input or output files.
    """
    pass


class AnalysisError(Exception):
    """
    Error raised when an analysis step fails.

    This exception should be used for errors occurring after successful
    parsing, such as invalid data assumptions, numerical failures, or
    unsupported analysis configurations.
    """
    pass
