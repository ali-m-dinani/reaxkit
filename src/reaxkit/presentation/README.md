# presentation

## Purpose
Formats analysis outputs for user-facing consumption: report payloads, persistence/export outputs, and plot/video rendering integration.

## What Belongs Here
- Report payload builders and registry wiring.
- Presentation specs and conversion/dispatch helpers.
- Plot and movie rendering integration code.

## What Does Not Belong Here
- Core analysis algorithms.
- Engine parsing logic.

## Structure
- `active_sites/`: active-site-specific report and plot exports.
- `plot/`: plot registry and renderer integration.
- `movie/`: video generation helpers.
- `powerpoint.py`: reusable PowerPoint export of categorized PNG/JPEG figures.
- `report_registry.py`, `reporting.py`, `persist.py`, `dispatcher.py`, `convert.py`, `specs.py`, `units.py`, `export_utils.py`.

## Flow
Workflows/core pass result payloads into presentation utilities to create report structures, figures, and persisted output artifacts.

## Extension Points
- Register report payload builders in `report_registry.py`.
- Add new renderer behavior in `plot/` and update dispatch mappings.

## Figure presentations

`write_figure_presentation(categories, destination)` in `powerpoint.py` accepts an
ordered mapping of category titles to image paths. Each nonempty category gets
an editable title slide followed by one figure per widescreen slide. Images are
embedded at their original resolution, centered, and fitted without cropping or
distortion. Category and figure order follow the caller's input. PNG and JPEG are
supported; PowerPoint is not required to generate the file. A failed export leaves
an existing destination intact.

Figure-slide headings are left-aligned. Each nonempty category is also a native
PowerPoint section, allowing it to be expanded or collapsed in the slide pane.
Optional `summary_counts={"Category": 3}` and `warning="..."` keyword arguments
add an opening editable two-column table and a bold red warning below it.

```python
from reaxkit.presentation.powerpoint import write_figure_presentation

write_figure_presentation(
    {"Angle scans": ["angle_plots/angle_example.png"],
     "Bond scans": ["bond_plots/bond_example.png"]},
    "figures.pptx",
)
```

`get-ffield-opt-plots --make-powerpoint` (alias `--make-ppt`) uses this exporter and writes
`ffield_opt_plots.pptx` beside the generated figures. It uses only the current
run's image paths, including EOS material subfolders, and groups both kinds of
`other_bar_plots` into one section. Empty categories are omitted. If all categories
are empty, the workflow reports that PowerPoint generation was skipped.
The opening summary includes every category count from `plot_summary.txt`
(including zeros and the two separate energy-bar types), followed by the same
bold not-plotted warning and CSV path. The summary has its own PowerPoint section.

The exporter writes the standard
[PresentationML package structure](https://learn.microsoft.com/en-us/office/open-xml/presentation/structure-of-a-presentationml-document).
Slide headings remain editable; the scientific plots are embedded images rather
than editable PowerPoint charts.
