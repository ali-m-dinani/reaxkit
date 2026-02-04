# Numerical Utilities

This section documents **numerical utilities** in ReaxKit.
These helpers provide **generic numerical operations** used across analyses and workflows,
without embedding ReaxFF-specific physics or file-format assumptions.

They serve as the mathematical and numerical backbone for higher-level analysis code.

---

## Scope of `utils/numerical`

Utilities in this folder typically handle:

- Numerical fitting and regression helpers
- Lightweight statistics
- Array and DataFrame transformations
- Safe wrappers around SciPy / NumPy routines
- Reusable numerical patterns used in multiple analyses

These utilities are **domain-agnostic** and reusable outside ReaxKit.

---

## Typical responsibilities

### Fitting and regression helpers

Common use cases include:

- Equation-of-state fitting (e.g. Vinet, Birch–Murnaghan)
- Linear and nonlinear least-squares fitting
- Residual and goodness-of-fit evaluation

These utilities are often used by:
- Elastic and bulk modulus analyses
- Stress–strain and energy–volume workflows

---

### Statistical helpers

Numerical utilities may include:

- Mean, variance, and RMS calculations
- Rolling or windowed statistics
- Robust estimators for noisy simulation data

They are designed to complement pandas and NumPy,
not replace them.

---

### Array and DataFrame transformations

Examples include:

- Safe slicing and reshaping
- Unit-consistent numerical normalization
- Conversions between NumPy arrays and pandas objects
- Defensive checks for NaNs, infinities, or empty inputs

---

## How numerical utilities fit in the stack

Typical data flow:

1. Handlers load raw data into DataFrames
2. Analyses extract numerical arrays
3. Numerical utilities:
   - perform fitting or statistics
   - return clean numerical results
4. Analyses interpret results physically
5. Workflows handle plotting and export

This keeps numerical logic isolated and testable.

---

## What numerical utilities should *not* do

Numerical utilities should avoid:

- Reading or writing files
- Plotting or visualization
- CLI argument handling
- Encoding ReaxFF-specific semantics

Those concerns belong in other layers.
