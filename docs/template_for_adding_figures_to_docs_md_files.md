# Template For Adding Figures To Docs Markdown Files

Use this template when adding figures from `docs/figures/` to ReaxKit documentation pages.

```md
<a id="figure_anchor_name"></a>

The figure below shows the result produced by this workflow:

<div class="figure-large" markdown="1">

![short_descriptive_alt_text](relative/path/to/figure.png)

*Figure: One-sentence caption explaining what the reader should notice.*

</div>
```

Concrete example:

```md
<a id="bond_events_with_smoothing_and_thresholds"></a>

The figure below shows an example bond-events output plot:

<div class="figure-large" markdown="1">

![bond_events_with_smoothing_and_thresholds](../../../figures/bond_events_with_smoothing_and_thresholds.png)

*Figure: Sample bond events plot with detected bond breakage and formation iterations.*

</div>
```

## Path Rules

Figures should live in:

```text
docs/figures/
```

Choose the relative path from the Markdown file you are editing:

```text
docs/architecture_overview.md                  -> figures/example.png
docs/api/workflows/trajectory_workflow.md      -> ../../figures/example.png
docs/api/workflows/file_tools/ffield_workflow.md -> ../../../figures/example.png
docs/tutorials/01_understanding_quickstart.md  -> ../figures/example.png
```


## Checklist

- Put the image under `docs/figures/`.
- Use a relative path from the current Markdown file.
- Use `style="width:85%; max-width:800px;"` for plain images.
- Use `<div class="figure-large" markdown="1">` when adding captions.
- Keep captions short and focused on what the figure demonstrates.
