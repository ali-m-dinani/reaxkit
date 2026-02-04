# Docstring Templates and Conventions

This page defines the **standard docstring format** used throughout ReaxKit.

All **public APIs** (analysis functions, handlers, workflows) must follow these
templates so that documentation renders cleanly in MkDocs and remains consistent,
concise, and user-friendly.

ReaxKit uses **NumPy-style docstrings**, rendered automatically by `mkdocstrings`.

---

## General Rules

- ✅ Use **NumPy-style** sections (`Parameters`, `Returns`, `Examples`, etc.)
- ✅ Keep docstrings **short, informative, and user-facing**
- ✅ Every **public function** must have a docstring
- ❌ Do not document private helpers (`_internal_function`)
- ❌ Do not describe internal implementation details
- ❌ Do not include long tutorials inside docstrings

Docstrings are part of the **public API**.

---

## File-Level Docstring Template (Analyzer Modules)

Place this at the **top of every `*_analyzer.py` file**.

```python
"""
<Short title of analysis> utilities.

This module provides functions for <what is analyzed> from a parsed
ReaxFF <file(s)> via <Handler name(s)>.

Typical use cases include:

- <use case 1>
- <use case 2>
- <use case 3>
"""
```

### Example

```python
"""
fort.7 analysis utilities.

This module provides functions for extracting atom-level and
iteration-level quantities from a parsed ReaxFF ``fort.7`` file
via ``Fort7Handler``.

Typical use cases include:

- extracting per-atom bond-order features
- computing coordination statistics
- building derived structural descriptors
"""
```

---

## Function-Level Docstring Template (Public Functions)

Use this template for every public function (functions without a leading _).

```text
def function_name(...):
    """
    One-line summary of what this function computes.
    
    Any other explanation can be written here if necessary (examples include explaining the logic, etc.)

    Works on
    --------
    <Handler(s)> — ``file_name`` [+ optional additional files]

    Parameters
    ----------
    param1 : type
        Short, user-facing description.
    param2 : type, optional
        Short description.

    Returns
    -------
    return_type
        What is returned (mention key columns/keys if DataFrame).

    Examples
    --------
    >>> from reaxkit.io.handlers.<handler> import <Handler>
    >>> from reaxkit.analysis.<path> import function_name
    >>> h = <Handler>("file")
    >>> out = function_name(h, ...)
    """
```

### One-line summary (required)

* Start with an imperative verb
* Example:
  * “Extract per-atom partial charges across selected frames.”
  * “Compute the radial distribution function (RDF).”

### “Works on” section (required for analysis functions)

This is ReaxKit-specific and very important.

```text
Works on
--------
Fort7Handler — ``fort.7``
```

Or, for multi-file analyses:

```text
Works on
--------
XmoloutHandler + Fort7Handler — ``xmolout`` + ``fort.7``
```

This allows users to instantly answer: “Which file does this function work on?”

### Parameters

* Include type + meaning
* Do not over-explain obvious parameters
* Group similar parameters together when possible
 
Good:

```text
frames : iterable of int, optional
    Frame indices to include.
```

Avoid:

```text
frames : iterable of int
    This parameter is used to specify which frames are selected
    and is later passed into internal selection logic...
```

### Returns (required)

* Always specify the return type
* If returning a DataFrame, list 2–5 key columns
* Keep it short

Good:

```text
Returns
-------
pandas.DataFrame
    Table with columns: ``frame_idx``, ``iter``, ``sum_BOs``.
```

### Examples (recommended)

* Keep examples 3–6 lines
* Show a happy path
* No long explanations
* No plotting unless absolutely necessary

Good:

```text
Examples
--------
>>> h = Fort7Handler("fort.7")
>>> df = get_sum_bos(h)
```

Avoid:

* Long workflows
* Multiple examples
* Plot-heavy code

---

## What NOT to Put in Docstrings

Do not include:

* Internal algorithm details
* Performance optimizations
* Code comments (“we loop over frames…”)
* Development notes
* CLI-specific explanations

These belong in:

* code comments
* tutorials
* design documents

---

## Private Functions

Private helpers (names starting with _) may have:

* a one-line docstring, or
* no docstring at all

They are not part of the public API and should not be documented extensively.

---

## Class-Level Docstring: Data Description (Handlers)

For **IO handler classes**, the class docstring should briefly document:

1. **What structured data the file is parsed into**
2. **What the class exposes to users** (summary vs per-frame data)

This makes handler behavior clear **without reading the code**.

### Extended Template (Handlers)

```python
class SomeHandler:
    """
    One-line summary of the file this handler parses.

    This class parses ReaxFF ``<file>`` files and exposes their contents
    as structured tabular data for downstream analysis.

    Parsed Data
    -----------
    Summary table
        One row per iteration, returned by ``dataframe()``, with columns:
        [<col1>, <col2>, <col3>, ...]

    Per-frame data
        Stored internally in ``self._frames`` (if applicable), where each
        frame is a ``pandas.DataFrame`` with columns:
        [<col1>, <col2>, <col3>, ...]

    Notes
    -----
    - Brief format quirks or guarantees (optional)
    - Ordering, deduplication, or correction rules (optional)
    """
```

### Example: `Fort7Handler`

```text
class Fort7Handler:
    """
    Parser for ReaxFF ``fort.7`` connectivity files.

    This class parses ReaxFF ``fort.7`` files and exposes atom–atom
    connectivity and bond-order information as structured tabular data.

    Parsed Data
    -----------
    Summary table
        One row per iteration, returned by ``dataframe()``, with columns:
        ["iter", "num_of_atoms", "num_of_bonds", "total_BO", "total_LP",
         "total_BO_uncorrected", "total_charge"]

    Per-frame atom tables
        Stored in ``self._frames``, one table per iteration, with columns:
        ["atom_num", "atom_type_num", "atom_cnn1..nb", "molecule_num",
         "BO1..nb", "sum_BOs", "num_LPs", "partial_charge", ...]

    Notes
    -----
    - Duplicate iterations are resolved by keeping the last occurrence.
    - Corrected bond orders are used unless otherwise specified.
    """
```

### Guidance

✅ List only the most important columns

✅ Use ..nb to indicate variable-length columns

✅ Prefer clarity over completeness

❌ Do not explain how columns are computed

❌ Do not include examples here (methods handle that)



