""""managing exceptions to prevent errors while running ReaxKit"""

class ParseError(Exception):
    """Raised when a file cannot be parsed correctly."""
    pass


class AnalysisError(Exception):
    """Raised when an analysis step fails."""
    pass
