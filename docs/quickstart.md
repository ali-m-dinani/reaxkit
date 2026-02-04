# Quick Start

This page walks you through a **minimal, end-to-end ReaxKit workflow**, and shows you how to use ReaxKit **CLI commands**.
By the end, you will have parsed a ReaxFF output file and produced a simple result.

This guide assumes ReaxKit is already installed (If not, see [Installation](installation.md)). Moreover,
if you are not familiar with the data organization by ReaxFF, see [ReaxFF File References](reaxff_reference/index.md).

---

## Step 1: Check that ReaxKit works

Open a terminal and run:

```bash
reaxkit --help
```

You should see the ReaxKit CLI help message listing available workflows.

---

## Step 2: Inspect available data and files

If you already know which file you should process, skip this step and go to step 3.

ReaxKit includes a built-in help system that knows which quantities appear
in which ReaxFF files.

To explore this:

```bash
reaxkit help
```

or search for a specific quantity:

```bash
reaxkit help pressure
```
```bash
reaxkit help "electric field"
```

This helps you decide which file you need for a given analysis.

---

## Step 3: Finding the supported tasks for a given ReaxFF file

If you want to know which task a specific file like `xmolout` supports, you can simply do:

```bash
reaxkit xmolout -h
```

which prints in the terminal a list of positional arguments (i.e., supported tasks) and options (i.e., supported flags).
For example, the above CLI command will show you that `xmolout` workflow supports tasks like `trajget`, `MSD`, `RDF`, etc.

Later on, for a specific task like `trajget` you may request its supported flags simply by:
```bash
reaxkit xmolout trajget -h
```

which lists CLI examples and options (i.e. flags) such as `--atoms`, `plot`, etc.

---

## Step 4: Run a simple analysis

Once you know the tasks and flags related to a specific ReaxFF output file, you can use the full, appropriate CLI command
to extract data and generate a simple plot:

```bash
reaxkit xmolout trajget --atoms 1 --dims z --xaxis time --plot --export atom1_z.csv
```

This will show you the z-trajectory for atom 1 across time (not iteration), and then exports its data (i.e., z-coordination vs time) in a csv file.

Generally, depending on the workflow:

* data may be printed to the terminal,

* plots may appear interactively,

* or files may be saved to a `reaxkit_outputs/` directory (if you generate an input file such as tregime, it will be saved in `reaxkit_generated_inputs/` by default).

### What just happened?

Behind the scenes, ReaxKit:

1. parsed the raw ReaxFF file using the `xmolout_handler`,

2. processed the data using the `xmolout_analyzer`,

3. exposed the result through the `xmolout_workflow` via the CLI.

You do not need to interact with these layers directly to use ReaxKit.

---

## Where to go next

* If you want a detailed explanation of what happened in this quick start, see [01_understanding_quickstart.md](tutorials/01_understanding_quickstart.md)

* More examples, specifically how to use/develop analyzers instead of workflows (i.e., CLI commands): see [examples](examples/README.md)

* Developer templates: see [templates](file_templates/index.md)

* Installation details: see [Installation](installation.md)

* CLI discovery: `run reaxkit help`

This quick start intentionally keeps things minimal.
More advanced workflows (multi-file analysis, plotting options, video generation)
are documented in the examples and tutorials.





