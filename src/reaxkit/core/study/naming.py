"""
Naming helpers for study entities.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from typing import Any


def slug(value: Any) -> str:
    """
    Slug.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    value : Any
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.naming import slug
    # Configure required arguments for your case.
    result = slug(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    text = str(value).strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    norm = "".join(out).strip("_")
    return norm or "value"


def slug_underscore(value: Any) -> str:
    """
    Slug underscore.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    value : Any
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.naming import slug_underscore
    # Configure required arguments for your case.
    result = slug_underscore(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return slug(value).replace("-", "_").replace(".", "_")


def canonical_token(text: str) -> str:
    """
    Canonical token.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    text : str
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.naming import canonical_token
    # Configure required arguments for your case.
    result = canonical_token(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def short_param_name(name: str) -> str:
    """
    Short param name.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    name : str
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.naming import short_param_name
    # Configure required arguments for your case.
    result = short_param_name(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    low = str(name).strip().lower()
    if low.endswith("_percent"):
        return low[: -len("_percent")]
    if low == "temperature":
        return "temp"
    return low


def compact_param_name(name: str) -> str:
    """
    Compact param name.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    name : str
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.naming import compact_param_name
    # Configure required arguments for your case.
    result = compact_param_name(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    low = str(name).strip().lower()
    if not low:
        return "p"
    token = low.split("_", 1)[0]
    token = token[:2] if len(token) >= 2 else token
    return token or "p"


def format_param_value_for_case(value: Any) -> str:
    """
    Format param value for case.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    value : Any
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.naming import format_param_value_for_case
    # Configure required arguments for your case.
    result = format_param_value_for_case(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    try:
        f = float(value)
        if f.is_integer():
            i = int(f)
            return f"{i:02d}" if 0 <= i < 100 else str(i)
    except Exception:
        pass
    return slug(value)


def case_label_from_params(params: dict[str, Any]) -> str:
    """
    Case label from params.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    params : dict[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.naming import case_label_from_params
    # Configure required arguments for your case.
    result = case_label_from_params(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    parts = [f"{short_param_name(k)}_{format_param_value_for_case(v)}" for k, v in params.items()]
    return "__".join(parts)


def case_label_from_params_compact(params: dict[str, Any]) -> str:
    """
    Case label from params compact.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    params : dict[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.naming import case_label_from_params_compact
    # Configure required arguments for your case.
    result = case_label_from_params_compact(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    parts = [f"{compact_param_name(k)}_{format_param_value_for_case(v)}" for k, v in params.items()]
    return "_".join(parts)

