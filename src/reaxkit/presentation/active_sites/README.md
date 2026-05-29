# presentation/active_sites

## Purpose
Active-site-specific presentation helpers: report payload construction and figure exports.

## What Belongs Here
- Active-site report payload builders/registration helpers.
- Active-site plot/image export utilities.

## What Does Not Belong Here
- Active-site structural/event analyzer logic (belongs in `analysis/active_sites`).

## Structure
- `reporting.py`
- `plot_exports.py`

## Flow
Consumes active-site result tables/summaries and produces report sections and exported figure artifacts.

## Extension Points
- Add new active-site report sections or figure variants here.
