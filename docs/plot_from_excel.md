# Plot figures prepared in Excel

Use the packaged `src/reaxkit/workflows/force_field_opt/data/template_ffield_opt_figure_generator.xlsx`.
`get_ffield_opt_plots` also copies this template into its output folder. Previously
exported workbooks are not automatically migrated: copy their data into the updated layout.

```powershell
reaxkit plot-from-excel --input training_figures.xlsx --output manuscript_figures --plot-style publication-bold
```

Without `--output`, figures and the manifest are saved in `excel_figures` beside
the input workbook, regardless of the current working directory. An explicit
`--output` selects a different destination; relative paths use the working directory.
Long bar labels tilt 45 degrees to leave more space for the plot.

Each of `bar_plots`, `line_plots`, and `EOS_plots` has two tables:

| Columns A–F | Meaning |
| --- | --- |
| Figure number | Positive integer; groups rows within this sheet only |
| Line in fort.99 | Optional source line number or expression; not used to fetch values |
| Identifier | Bar category, categorical line label, or numeric-curve point identifier |
| X value | Numeric curve coordinate; mandatory for EOS; unused for bars |
| ReaxFF value | Finite numeric prediction |
| QM value | Finite numeric reference |

Columns H–K contain `Figure number`, `Title`, `X-axis label`, `Y-axis label`.
Supply exactly one settings row for each figure, with axis units and energy
normalization explicitly stated. Titles may be blank. Headers must be in row 1;
capitalization does not matter. Blank data rows are ignored. Other sheets are ignored.

Bars retain worksheet order. Numeric curves sort by X (duplicate coordinates
retain row order). For categorical reaction profiles in `line_plots`, leave every
X value in a figure blank: identifiers label evenly spaced points in row order.
Do not mix blank and numeric X values within a line figure. A curve needs at
least two points. EOS curves connect supplied points; no EOS fit is performed.
No signs, units, normalization, or energy references are changed automatically.

For example, rows marked `1` and `2` in `EOS_plots` generate
`EOS_plots/figure_001.png` and `EOS_plots/figure_002.png`. A bar figure numbered `1`
is independent and goes to `bar_plots/figure_001.png`.

The default exports are PNG and SVG. Use `--formats png svg pdf` for all three,
`--width 8 --height 5.2` for figure dimensions in inches, and `--overwrite` to
replace existing outputs. Match plot style and dimensions when mixing these plots
with other ReaxKit outputs; differing manuscript resizing also scales fonts and lines.
The shared renderer uses tight export bounds, so whitespace can vary with labels.

Both `plot-from-excel` and `get-ffield-opt-plots` accept `--color-style default`
(the existing appearance) or `--color-style light-fill`. The latter keeps the
series colors for bar borders and lines, and blends bar and marker interiors
35% toward white. Legends use the same treatment. Color style is independent
of `--plot-style`, so it also works with `publication` and `publication-bold`.

```powershell
reaxkit plot-from-excel --input training_figures.xlsx --color-style light-fill --plot-style publication-bold
reaxkit get-ffield-opt-plots --project-root . --color-style light-fill --plot-style publication-bold
```

The command validates all workbook data before rendering. Errors report sheet/cell
locations. Formula cells use Excel's saved results: recalculate and save the workbook
in Excel first. ReaxKit does not evaluate Excel formulas or detect stale cached results.
`excel_figures.json` records source rows, entered values, settings and exported paths.

The migrated packaged template retains the previous numeric example data. Its
original formulas and charts are preserved in the sibling `_legacy.xlsx` workbook.
Check example units before publication. The legacy line example uses sequential
X coordinates because none were previously supplied. Excel preview charts are
optional and do not control ReaxKit output grouping or appearance.
