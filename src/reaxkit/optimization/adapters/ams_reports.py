"""Recognized AMS 2023.103 reports; unknown layouts retain explicit mapping."""

import math
import re

from .base import NUMBER
from ..models import finite, progress


def mcff_progress(reader):
    path = reader.root / "MCFFOptimizer.log"
    if reader.options or not path.is_file():
        reader.configured_progress(path.name, "iteration")
        return
    lines = reader.lines(path)
    header = "Iter Repl Current error Changed Bounded Accepted Rejected AcceptRate Beta MaxStep Distance Elapsed"
    if not lines or " ".join(lines[0][1].split()) != header:
        reader.configured_progress(path.name, "iteration")
        return
    best = {}
    initial = {}
    for row, line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 12 or not all(value.isdigit() for value in fields[:2]):
            reader.result.warnings.append(f"{path.name}:{row}: malformed MCFF record")
            continue
        step, replica = map(int, fields[:2])
        score = finite(fields[2].rstrip("*"))
        if step == 0:
            best[replica] = None
            initial[replica] = score
        if score is not None:
            previous_best = best.get(replica)
            best[replica] = score if previous_best is None else min(previous_best, score)
        reader.result.progress.append(progress(
            path.name, row, fields[2].rstrip("*"), native_step=step,
            step_kind="iteration", native_replica=replica,
            score_kind="initial" if step == 0 else "current",
            initial_score=initial.get(replica), best_score=best.get(replica),
            best_score_scope="observed_current_errors_including_initial",
            native_elapsed_seconds=finite(fields[-1]),
            native_elapsed_semantics="unverified_not_cumulative", raw_record=line,
        ))


def cmaes_progress(reader):
    path = reader.root / "cmaes_progress.txt"
    if reader.options or not path.is_file():
        reader.configured_progress(path.name, "generation")
        return
    lines = reader.lines(path)
    if not any(line.startswith("function evaluations ") for _, line in lines):
        reader.configured_progress(path.name, "generation")
        return
    blocks = []
    block = []
    for row, line in lines:
        if line.startswith("# ---------") and block:
            blocks.append(block)
            block = []
        block.append((row, line))
    if block:
        blocks.append(block)
    for block in blocks:
        text = "\n".join(line for _, line in block)
        evaluations = re.search(r"^function evaluations (\d+)\s*$", text, re.M)
        score = re.search(r"^function value f\(x\)=(" + NUMBER + r")\s*$", text, re.M | re.I)
        best = re.search(r"^xbestever found after (\d+) evaluations, function value (" + NUMBER + r")\s*$", text, re.M | re.I)
        cpu = re.search(r"^elapsed \(CPU\) time \[s\] (" + NUMBER + r")\s*$", text, re.M | re.I)
        if not all((evaluations, score, best, cpu)):
            if text.strip():
                reader.result.warnings.append(f"{path.name}:{block[0][0]}: incomplete CMA-ES scalar report deferred")
            continue
        reader.result.progress.append(progress(
            path.name, block[0][0], score[1], native_step=int(evaluations[1]),
            step_kind="evaluation", candidate_evaluations=int(evaluations[1]),
            score_kind="generation_best", best_score=finite(best[2]),
            best_score_scope="optimizer_candidates_excluding_initial_baseline",
            best_candidate_evaluation=int(best[1]), cpu_seconds=finite(cpu[1]),
            raw_record=text,
        ))
    baseline = reader.root / "fort.13"
    if baseline.is_file():
        records = reader.lines(baseline)
        values = [finite(line.strip()) for _, line in records]
        reports = reader.result.progress
        if reports and len(values) == reports[-1]["candidate_evaluations"] + 1 and all(value is not None for value in values):
            previous = 0
            consistent = True
            for report in reports:
                count = report["candidate_evaluations"]
                if count <= previous or report["score"] is None or report["best_score"] is None:
                    consistent = False
                    break
                if not math.isclose(min(values[previous + 1:count + 1]), report["score"], rel_tol=1e-5, abs_tol=1e-4) or not math.isclose(min(values[1:count + 1]), report["best_score"], rel_tol=1e-5, abs_tol=1e-4):
                    consistent = False
                    break
                previous = count
            if consistent:
                for report in reports:
                    report.update(initial_score=values[0], initial_score_source="fort.13",
                                  initial_score_source_record=records[0][0])
                if values[0] < reports[-1]["best_score"]:
                    reader.result.warnings.append("Initial fort.13 baseline is better than CMA-ES best candidate; best_score excludes that baseline")
            else:
                reader.result.warnings.append("fort.13 does not match candidate reports; initial baseline not associated")
        else:
            reader.result.warnings.append("fort.13 incomplete or ambiguous; initial baseline not associated")
