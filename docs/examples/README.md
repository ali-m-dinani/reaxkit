# ReaxKit Examples

This directory contains runnable scripts that demonstrate current ReaxKit usage.

## Directory structure

```text
examples/
    data/
        control
        sample_tabular_data.csv
        small_fort.7
        small_xmolout
    xmolout_basic_example.py
    connectivity_multifile_example.py
    gen_plot_presentation_example.py
    multi_engine_get_msd_cli_blueprint.py
    README.md
```

## Example scripts

### [`xmolout_basic_example.py`](https://github.com/ali-m-dinani/reaxkit/blob/new-features/docs/examples/xmolout_basic_example.py)

Demonstrates direct Python-task usage on `xmolout` data:
- load trajectory via `XmoloutHandler`
- extract coordinate series
- compute MSD
- extract cell dimensions
- export CSV outputs

### [`connectivity_multifile_example.py`](https://github.com/ali-m-dinani/reaxkit/blob/new-features/docs/examples/connectivity_multifile_example.py)

Demonstrates multi-file connectivity/charge analysis with `xmolout` + `fort.7`:
- load `ConnectivityData` and `ChargeData`
- extract charge and sum-bond-order features
- build connection lists/tables/stats
- detect bond events
- compute coordination status

Related tutorial: [02_atom_property_and_video_workflows.md](../tutorials/02_atom_property_and_video_workflows.md)

### [`gen_plot_presentation_example.py`](https://github.com/ali-m-dinani/reaxkit/blob/new-features/docs/examples/gen_plot_presentation_example.py)

Demonstrates file-agnostic plotting utilities on tabular data:
- single and multi-series plots
- directed plots
- dual y-axis plots
- 3D scatter
- 2D heatmap projection

Related tutorial: [04_gen_plot_workflow.md](../tutorials/04_gen_plot_workflow.md)

### [`multi_engine_get_msd_cli_blueprint.py`](https://github.com/ali-m-dinani/reaxkit/blob/new-features/docs/examples/multi_engine_get_msd_cli_blueprint.py)

Demonstrates a task-first, multi-engine architecture blueprint for
`reaxkit get_msd` style flows.

## Notes

- These examples are for learning and exploration, not formal tests.
- For conceptual walkthroughs, use [tutorials](../tutorials/index.md).
