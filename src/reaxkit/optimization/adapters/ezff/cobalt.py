"""Validate the explicit cobalt runner's labeled LAMMPS reports."""

import math

from ...models import finite, identity, prediction


OBJECTIVE = "sqrt_sum_squared_accuracy_divided_energy_residuals"
SCOPE = "Co_clusters/Co2_cluster energy targets only"


def close(first, second):
    first, second = finite(first), finite(second)
    return first is not None and second is not None and math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-8)


def vector_matches(first, second):
    return (isinstance(first, list) and isinstance(second, list) and len(first) == len(second)
            and all(finite(left) is not None and finite(left) == finite(right) for left, right in zip(first, second)))


def report_predictions(reader, path, report, candidate, score):
    if (not isinstance(report, dict) or report.get("engine") != "LAMMPS ReaxFF"
            or report.get("objective") != OBJECTIVE or report.get("dataset_scope") != SCOPE
            or report.get("geometry_relaxation") is not False or report.get("boundary") != "nonperiodic"
            or not close(report.get("loss"), score)):
        raise ValueError("profile or loss mismatch")
    items, energies = report.get("items"), report.get("energies")
    if not isinstance(items, list) or not items or not isinstance(energies, dict):
        raise ValueError("missing prediction items or geometry energies")
    records, errors, lines = [], [], set()
    source = reader.relative(path)
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("invalid prediction item")
        line = item.get("line")
        if type(line) is not int or line < 1 or line in lines or item.get("unit") != "kcal/mol":
            raise ValueError("invalid/duplicate training line or unknown units")
        lines.add(line)
        first, second = item.get("first"), item.get("second")
        if not isinstance(first, str) or not first or not isinstance(second, str) or not second:
            raise ValueError("missing geometry identities")
        values = [finite(item.get(key)) for key in ("first_divisor", "second_divisor", "tolerance", "reference", "prediction")]
        if any(value is None for value in values) or min(values[:3]) <= 0:
            raise ValueError("invalid prediction values or divisors")
        first_divisor, second_divisor, tolerance, reference, predicted = values
        first_energy, second_energy = finite(energies.get(first)), finite(energies.get(second))
        if first_energy is None or second_energy is None:
            raise ValueError("missing finite geometry energy")
        if not close(predicted, first_energy / first_divisor - second_energy / second_divisor):
            raise ValueError("prediction does not match geometry energy expression")
        error = ((predicted - reference) / tolerance) ** 2
        if not close(error, item.get("weighted_squared_error")):
            raise ValueError("weighted residual mismatch")
        errors.append(error)
        label = f"ENERGY:{first}/{first_divisor:.17g} - {second}/{second_divisor:.17g} (trainset line {line})"
        records.append(prediction(
            source, candidate, label, reference, predicted,
            training_entry_id=identity("ezff-cobalt", line, first, first_divisor, second, second_divisor),
            property="ENERGY", unit="kcal/mol", weight=tolerance,
            weight_semantics="accuracy_divisor", weighted_squared_error=error,
            source_record=index, training_source_record=line,
            geometry_relaxation=False, boundary="nonperiodic",
        ))
    if not close(math.sqrt(math.fsum(errors)), score):
        raise ValueError("recomputed objective does not match loss")
    return records


def collect_cobalt(reader, archive):
    manifest = reader.root / "cobalt-run.json"
    if not manifest.is_file():
        return
    profile = reader.json(manifest)
    if (not isinstance(profile, dict) or profile.get("engine") != "LAMMPS ReaxFF"
            or profile.get("objective") != OBJECTIVE or profile.get("dataset_scope") != SCOPE):
        reader.result.warnings.append("cobalt-run.json: unknown/incomplete profile; cobalt predictions withheld")
        return
    metadata = dict(engine_identity=profile["engine"], objective_definition=profile["objective"],
                    dataset_scope=profile["dataset_scope"], profile_source="cobalt-run.json")
    if isinstance(profile.get("ezff_version"), str):
        metadata["optimizer_version"] = profile["ezff_version"]
    for record in reader.result.progress:
        record.update(metadata)
    reader.result.warnings.append("Cobalt cluster/dimer subset only; cross-optimizer comparability remains unverified")
    if not isinstance(archive, dict) or not isinstance(archive.get("variables"), list):
        reader.result.warnings.append("Cobalt candidate association requires the evaluated variable archive")
        return
    bounds_path = reader.root / "variable_bounds"
    if not bounds_path.is_file():
        reader.result.warnings.append("No variable_bounds order; cobalt candidate predictions withheld")
        return
    names = []
    for _, line in reader.lines(bounds_path):
        fields = line.split()
        if len(fields) != 3 or fields[0] in names or finite(fields[1]) is None or finite(fields[2]) is None:
            reader.result.warnings.append("Invalid variable_bounds; cobalt candidate predictions withheld")
            return
        names.append(fields[0])
    if not names:
        return
    verified = []
    for record in reader.result.progress:
        if record.get("record_kind") != "evaluation":
            continue
        index = record["native_step"] - 1
        path = reader.root / "evaluations" / f"candidate-{index:04d}" / "report.json"
        parameters_path = path.parent / "parameters.json"
        if not path.is_file() or not parameters_path.is_file():
            reader.result.warnings.append(f"candidate-{index:04d}: missing report/parameters; predictions withheld")
            continue
        report, parameters = reader.json(path), reader.json(parameters_path)
        try:
            vector = archive["variables"][index]
            if (not isinstance(parameters, dict) or set(parameters) != set(names)
                    or not vector_matches(vector, [parameters[name] for name in names])):
                raise ValueError("candidate parameters do not match archive")
            if not isinstance(report, dict) or type(report.get("candidate")) is not int or report["candidate"] != index:
                raise ValueError("candidate identity mismatch")
            records = report_predictions(reader, path, report, record["candidate_id"], record["score"])
        except (ValueError, OverflowError, IndexError) as error:
            reader.result.warnings.append(f"{reader.relative(path)}: {error}; predictions withheld")
            continue
        for prediction_record in records:
            prediction_record.update(metadata)
        reader.result.predictions.extend(records)
        record["native_candidate_id"] = f"candidate-{index:04d}"
        record["candidate_report"] = reader.relative(path)
        verified.append((vector, record))
    recommendations = {}
    for record in reader.result.progress:
        if record.get("record_kind") != "recommendation":
            continue
        path = reader.root / record["source"]
        variables_path = path.with_name("variables")
        if variables_path not in recommendations:
            recommendations[variables_path] = dict(reader.lines(variables_path)) if variables_path.is_file() else {}
        line = recommendations[variables_path].get(record["source_record"], "")
        vector = [finite(value) for value in line.split()]
        matches = [candidate for values, candidate in verified
                   if vector_matches(vector, values) and close(record["score"], candidate["score"])]
        if len(matches) == 1:
            record["evaluated_candidate_id"] = matches[0]["candidate_id"]
        else:
            reader.result.warnings.append(f"{record['source']}:{record['source_record']}: recommendation candidate association unavailable/ambiguous")
