# domain

## Purpose
Defines typed contracts for requests, results, and common trajectory/connectivity data models.

## What Belongs Here
- Base request/result abstractions.
- Shared domain data classes consumed by analyzers and runtime.

## What Does Not Belong Here
- Parsing adapters.
- Command routing logic.
- Presentation/report formatting.

## Structure
- `base_request.py`
- `base_result.py`
- `data_models.py`

## Flow
Engine adapters materialize domain objects, analyzers consume them, and results are returned using domain result contracts.

## Extension Points
- Add new domain model fields/classes with backward compatibility for existing workflows/tasks.
