"""Compact molecular frame kernels without a pandas task invocation per frame."""

from contextlib import closing
import math
import re
import pandas as pd
from reaxkit.core.runtime.execution_contracts import resolve_execution_policy
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
from reaxkit.core.runtime.frame_tables import selected_frame_envelopes


def _numeric(value, default=math.nan):
    try:
        number = pd.to_numeric(value)
        return default if pd.isna(number) else number
    except (ValueError, TypeError):
        return default


def molecular_frame_table(task, frames, request, kind, pipeline=None, reporter=None):
    pipeline = pipeline or BoundedFramePipeline(resolve_execution_policy(task, request))
    columns = {
        "dominant": ["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"],
        "mass": ["frame_index", "iter", "molecular_formula", "freq", "molecular_mass"],
        "composition": ["frame_index", "iter", "element", "count"],
        "frequency": ["frame_index", "iter", "molecules", "freq", "molecular_mass"],
        "totals": ["frame_index", "iter", "quantity", "value"],
    }[kind]

    def kernel(data):
        iteration = int(data.iterations[0])
        if kind == "totals":
            if data.totals.empty:
                return []
            row = data.totals.iloc[0]
            return [{"iter": iteration, "quantity": quantity, "value": float(_numeric(row[quantity]))}
                    for quantity in sorted(request.quantities) if quantity in row]
        population = [(int(it), str(formula), _numeric(freq, math.nan if kind == "frequency" else 0.), _numeric(mass))
                      for it, formula, freq, mass in data.molecular_species[
                          ["iter", "molecular_formula", "freq", "molecular_mass"]].itertuples(index=False, name=None)]
        if kind == "frequency":
            by_formula = {}
            for it, formula, freq, mass in population:
                if formula in by_formula and formula in request.molecules:
                    raise ValueError("Cannot reindex duplicate molecular iteration/formula rows.")
                by_formula[formula] = freq, mass
            return [{"iter": iteration, "molecules": str(formula),
                     "freq": float(by_formula.get(str(formula), (0., math.nan))[0]),
                     "molecular_mass": float(by_formula.get(str(formula), (0., math.nan))[1])}
                    for formula in sorted(map(str, request.molecules))]
        if kind == "dominant":
            selected = sorted((row for row in population if row[2] >= float(request.min_freq)),
                              key=lambda row: (-row[2], math.isnan(row[3]), -row[3] if not math.isnan(row[3]) else 0., row[1]))[:max(1, int(request.top_n))]
            return [{"iter": row[0], "rank": rank, "molecular_formula": row[1],
                     "freq": float(row[2]), "molecular_mass": float(row[3])}
                    for rank, row in enumerate(selected, 1)]
        valid = [row for row in population if not math.isnan(row[3])]
        if not valid:
            return []
        it, formula, frequency, mass = max(valid, key=lambda row: row[3])
        if kind == "mass":
            return [{"iter": it, "molecular_formula": formula, "freq": frequency, "molecular_mass": mass}]
        return [{"iter": it, "element": element, "count": int(count)}
                for element, count in sorted(re.findall(r"([A-Z][a-z]*)(\d+)", formula), key=lambda pair: pair[0])]

    rows = []
    with closing(pipeline.map_ordered(selected_frame_envelopes(frames, request), kernel)) as completed:
        for count, item in enumerate(completed, 1):
            rows.extend({"frame_index": item.envelope.source_frame, **row} for row in item.value)
            if reporter:
                reporter("stream", count, 0, "Analyzing molecular populations")
    return pd.DataFrame(rows, columns=columns)
