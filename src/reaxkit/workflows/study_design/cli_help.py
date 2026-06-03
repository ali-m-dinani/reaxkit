"""CLI help text for the study workflow.

This module stores long-form parser description text so the top-level
`study_workflow.py` entrypoint remains readable while preserving detailed
user guidance.

**Usage context**

- CLI help composition: Imported by `study_workflow.build_parser`.
- Documentation ownership: Keeps command-help prose centralized.
- Non-runtime role: Contains no execution logic.
"""

STUDY_PARSER_DESCRIPTION = (
    "Create and initialize parameter-sweep study layouts.\n\n"
    "Follow these steps to design and run a study:\n"
    " 1. Create a study YAML template using either of the followin commands and edit it to declare parameters, cases, stages, and analyses.\n"
    "  - Examples:\n"
    "    reaxkit study --make-yaml study.yaml\n"
    "    reaxkit study --gen-yaml\n"
    "  - NOTE: You need a template folder which is one example case with all stages and parameters declared to use as a reference for "
    "all other folders.\n"
    "  - NOTE: if you have multiple stages under run and they are dependent on each other, you need to have 'set -euo pipefail' in your job submission file "
    "to ensure successive stages run after each other instead of in parallel.\n\n"
    " 2. Initialize the study layout with --init, which generates folders and manifests for each case and stage.\n"
    "  - Examples:\n"
    "    reaxkit study --init study.yaml --root .\n"
    "    reaxkit study --init study.yaml --root studies --force\n\n"
    " 3. Run the study stages with --run, which executes the declared stages for each case and replicate, tracking status and artifacts.\n"
    "  - Cases can run in parallel by specifying --parallel-workers with the number of workers to use.\n"
    "  - Failed or waiting replicates can be rerun with --rerun-failed, which cleans stage artifacts before rerunning.\n"
    "  - Examples:\n"
    "    reaxkit study --run study_MgTemp/\n"
    "    reaxkit study --run study_MgTemp/ --parallel-workers 4\n"
    "    reaxkit study --run study_MgTemp/ --rerun-failed\n"
    "    reaxkit study --run study_MgTemp/ --stage MM\n"
    "    reaxkit study --run study_MgTemp/ --case mg_05__temp_300\n\n"
    " 4. Analyze the study with --analyze, which executes analysis pipelines declared in the top-level study YAML for each stage.\n"
    "  - Analyses can be filtered by title or variable name with --analysis.\n"
    "  - Failed or waiting analyses can be rerun with --rerun-failed, which cleans analysis artifacts before rerunning.\n"
    "  - Examples:\n"
    "    reaxkit study --analyze study_MgTemp/\n"
    "    reaxkit study --analyze study_MgTemp/ --rerun-failed\n"
    "    reaxkit study --analyze study_MgTemp/ --analysis msd\n\n"
    " 5. Aggregate results, which execute aggregation pipeline declared in the top-level study YAML. "
    "  - Aggregation means combining results across cases and replicates for a given analysis variable.\n"
    "  - Examples:\n"
    "    reaxkit study --aggregate study_MgTemp/\n"
    "    reaxkit study --aggregate study_MgTemp/ --aggregate msd_atom1_aggregation\n"
    "    reaxkit study --aggregate study_MgTemp/ --aggregate msd_atom1_aggregation --stage NVT\n\n"
    " 6. Present results, which generates plots and tables from completed analyses and aggregates.\n"
    "  - Examples:\n"
    "    reaxkit study --present study_MgTemp/\n"
    "    reaxkit study --present study_MgTemp/ --present msd_atom1_aggregation\n\n"
    " 7. Manage study metadata and artifacts with --manage, which supports path updates, case renaming, and removals of analysis, aggregate, and cache outputs.\n"
    "  - Path updates and case renaming are useful when study folder structures change after initialization, and removals are useful for cleaning up old or failed runs.\n"
    "  - Targets for removals can be filtered by case, replicate, stage, analysis title, and aggregate title.\n"
    "  - Examples:\n"
    "    reaxkit study --manage study_MgTemp/ --action update-paths --target paths\n"
    "    reaxkit study --manage study_MgTemp/ --action rename-cases --target case-names\n"
    "    reaxkit study --manage study_MgTemp/ --action remove --target cache --older-than 14 --dry-run\n\n"
)

