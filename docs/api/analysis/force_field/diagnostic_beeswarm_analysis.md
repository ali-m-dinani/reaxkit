# Diagnostic Beeswarm Analysis

::: reaxkit.analysis.force_field.diagnostic_beeswarm

The diagnostic beeswarm analysis combines the optimization diagnostic samples,
force-field parameter tables, and declared optimization parameter bounds. It
normalizes each sampled value with its stored lower and upper bounds and
prepares objective-colored rows for the force-field diagnostic workflow.

The result contains a sample-level `table` and a parameter-level `parameters`
table. Parameters can be sorted by numeric pointer, final value, or starting
value, and objective colors can use either per-parameter ranges or one global
range.
