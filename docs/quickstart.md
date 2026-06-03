# Quick Start

This page walks through a minimal end-to-end ReaxKit CLI run.

---

## Step 1: Verify installation

run the following command to check if the CLI is available:

```bash
reaxkit help -h
```

If help is shown, the CLI is available.


As you will see in the next steps, passing `-h` in the CLI gives you 3 types of information:

1. a description of the command and its purpose
2. a list of example usages
3. a list of flags and their descriptions

For example, the output of `reaxkit help -h` includes:

1. a description of the `help` command and its purpose
```
Interactive help and discovery for ReaxKit commands, capabilities, and file 
semantics. Use this command to search ReaxKit concepts (for example analyses
 or generators) and ReaxFF-related files by keyword. You can narrow results, 
 enforce exact matching, and request detailed mapping information.

For more information, you can see:
 ReaxKit code: https://github.com/ali-m-dinani/reaxkit
 ReaxFF documentation: https://ali-m-dinani.github.io/reaxkit/
```

2. a list of example usages of the `help` command:

```
Examples:
  1. Basic keyword search:
   reaxkit help "msd"

  2. Search with multi-word phrase:
   reaxkit help "bond order"

  3. Limit result count:
   reaxkit help "bond order" --top 3

  4. Search with explicit engine context:
   reaxkit help "restraint" --engine reaxff

  5. Show detailed mapping information:
   reaxkit help "fort.7" --all-info
   reaxkit help "xmolout" --all-info
```

3. a list of flags and their descriptions:

```
Options
Flag            | Required | Default | Help                                                                                                                                | Choices         
----------------+----------+---------+-------------------------------------------------------------------------------------------------------------------------------------+-----------------
-h, --help      | no       | -       | show this help message and exit                                                                                                     | -               
----------------+----------+---------+-------------------------------------------------------------------------------------------------------------------------------------+-----------------
--top TOP       | no       | 1       | Maximum results per category (generator/file/analyzer/workflow), sorted by score. Example: --top 3, which returns only the top 3    | -               
                |          |         | hits per category.                                                                                                                  |                 
----------------+----------+---------+-------------------------------------------------------------------------------------------------------------------------------------+-----------------
--engine ENGINE | no       | -       | Optional engine context for dataclass-to-file mappings. Example: --engine reaxff, which resolves relationships using ReaxFF         | -               
                |          |         | context.                                                                                                                            |                 
----------------+----------+---------+-------------------------------------------------------------------------------------------------------------------------------------+-----------------
--all-info      | no       | false   | Show detailed implementation and file/dataclass/analyzer mapping information. Example: --all-info, which expands output beyond      | -               
                |          |         | summary hits.                                                                                                                       |                 
----------------+----------+---------+-------------------------------------------------------------------------------------------------------------------------------------+-----------------
--exact-match   | no       | false   | Match query exactly against item title (and aliases) before returning results. Example: --exact-match, which avoids broad fuzzy     | -               
                |          |         | matches.                                                                                                                            |                 
----------------+----------+---------+-------------------------------------------------------------------------------------------------------------------------------------+-----------------
```

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
```

give you information on how to use the `timeseries` and `get_msd` commands, respectively.

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
