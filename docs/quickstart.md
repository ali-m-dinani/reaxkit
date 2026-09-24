# Quick Start

This page walks through a minimal end-to-end ReaxKit CLI run.

---

## Step 1: Verify installation

run the following command to check if the CLI is available:

```bash
reaxkit help -h
```

If help is shown, the CLI is available.


`reaxkit -h` lists commands. Use `reaxkit COMMAND -h` for the usual input,
scientific choices, and outputs. Use `reaxkit COMMAND --help-all` to see every
accepted option, including execution, storage, diagnostics, and source-file
overrides. The same switches work for workflow tasks such as
`reaxkit fort7 get --help-all`.

Help exits before loading trajectory data or creating an analysis workspace.
You do not need to supply required scientific flags just to read help.

---
## Step 2: Discover commands and searchable topics

Reaxkit supports a variety of commands and topics. In order to discover them, you can use 
the `help` command with your question of interest. For example, you may write:

```bash
reaxkit help
reaxkit help pressure
reaxkit help "electric field"
```

The first one shows how you a short message on how you can use `help`. You have seen a more 
comprehensive help output when you ran `reaxkit help -h` in Step 1. The second and third examples show how you can ask for specific data such as pressure or electric field. 
The output will show you which commands and topics are relevant to your query, along with a 
short description of each.

But more often, you are looking for a specific task to turn your raw data into insights. 
In that case, you can ask for help on a specific task. For example:

```bash
reaxkit help msd
reaxkit help "sort geo"
reaxkit help "study design"
```

If the help output does not contain the information you need, 
you may pass some flags such as `--top 3` to `reaxkit help <YOUR_QUERY>` as in 
`reaxkit help msd --top 3` for more options.:

Another useful command is `reaxkit intspec` which lists all available files or commands
under a given folder or file. For example, you can run:


```bash
reaxkit intspec --folder workflow
reaxkit intspec --file trajectory_workflow
```

to see all available workflows and the details of the `trajectory_workflow` workflow, respectively.

---

## Step 3: Check task-specific help

Once you have identified a command that seems relevant to your task, 
you can see its specific help to find out what flags it supports and how to use it.
To do so, you can run the command with `-h` or `--help`. For example:

```bash
reaxkit timeseries -h
reaxkit get_msd -h
reaxkit get_msd --help-all
```

show the usual options for `timeseries` and `get_msd`, then every accepted
`get_msd` option. Required flags appear in short help; less common controls
remain in full help.

**Which help should I use?** `reaxkit help "QUERY"` searches for relevant
commands and topics, while `reaxkit -h` lists the command directory.
Once you know the command, `COMMAND -h` (or `--help`) shows its required and
commonly used options. `COMMAND --help-all` (also `--all-flags`) shows the full
categorized option reference, including advanced execution, storage,
diagnostic, and file-selection flags. It expands the same command's help;
it does not search topics or list every command's options. For nested tasks,
use the complete path, such as `reaxkit fort7 get --help-all`.
Use `reaxkit intspec` to inspect available workflows and their contents.

---

## Step 4: Run a simple analysis

As an example, you can extract the z-dimension trajectory of atom-1 and export it CSV using
the following command:

```bash
reaxkit timeseries --field trajectory[1].z --xaxis time --export atom1_z.csv
```

This command:
- resolves trajectory data input
- computes the requested time series
- exports a table to CSV

---

## Output behavior

Depending on flags and command:
- tables can be printed to terminal
- plots can be displayed (`--show`) or saved (`--save`)
- tables can be exported (`--export`)
- generated input files are written under generator output locations

---

## What happened internally

At a high level:
1. workflow parses CLI args
2. runtime resolves required data sources
3. analysis task executes
4. presentation/export layer handles outputs

---

## Next steps

- Detailed walkthrough: [01_understanding_quickstart.md](tutorials/01_understanding_quickstart.md)
- More runnable examples: [examples](examples/README.md)
- Full tutorial sequence: [tutorials](tutorials/index.md)
