# core/study

## Purpose
Implements study orchestration engines and schema/io primitives for multi-run study workflows.

## What Belongs Here
- Study run/analyze/aggregate/present/manage engine modules.
- Study naming/schema/logging/io helpers.

## What Does Not Belong Here
- Generic command routing not specific to study lifecycle.

## Structure
- Engine modules: `run_engine.py`, `analyze_engine.py`, `aggregate_engine.py`, `present_engine.py`, `manage_engine.py`.
- Support modules: `schema.py`, `io.py`, `naming.py`, `logging.py`, `init.py`.

## Flow
Meta study workflows invoke these engines to create, execute, analyze, aggregate, and present study runs.

## Extension Points
- Add new study lifecycle phases by introducing new engine module plus schema/io updates.
