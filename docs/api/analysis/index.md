# Analysis

This section documents the **analysis layer** in ReaxKit: functions that transform parsed ReaxFF data
into **metrics, summaries, plots, and exportable tables**.

---

## How this docs folder is organized

This folder is split into two documentation groups:

- [Per-file analyses](per_file/index.md) → analyses tied to a *single* ReaxFF output file  
  (for example: “what can I compute from `xmolout` alone?”)

- [Composed analyses](composed/index.md) → higher-level routines that **combine multiple files/handlers**  
  (for example: “use `xmolout` + `fort.7` connectivity to compute local dipoles / cluster properties”)

---

## What you’ll find on each analysis page

Most pages follow the same structure so they are easy to scan:

1. **What it computes** (metric/summary/plot)
2. **Inputs** (which file(s), which handler(s), required columns/metadata)
3. **Outputs** (DataFrame schema, plots, exported CSV structure)
4. **Python example** (direct API usage)
5. **CLI example** (the equivalent `reaxkit <file> <task> ...` usage)
6. **Notes / gotchas** (units, frame/iteration mapping, alias resolution)

---

## Quick usage patterns

### Python
Typical usage is:

```python
from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.analysis.xmolout_analyzer import get_xmolout_data  # example

xh = XmoloutHandler("xmolout")
df = get_xmolout_data(xh)
print(df.head())
```

### CLI
Most analyses are also available as CLI workflows:

```bash
reaxkit <file-kind> <task> [--file ...] [--xaxis ...] [--yaxis ...] [--plot] [--save ...] [--export ...]
```

Exact subcommands/options are documented on the corresponding page under `per_file/` or `composed/`.

---

## Conventions used across analyses

- **Frames vs iterations**  
  ReaxFF often reports *iterations* while plotting is often easier in *time* or *frame index*.
  Many workflows support `--xaxis iter|frame|time` and convert using the run `control` file.

- **Column aliases**  
  Many files use different headers for the same concept (e.g., `Density` vs `Dens(kg/dm3)`).
  Workflows typically resolve these via the alias utilities so users can pass canonical names.


