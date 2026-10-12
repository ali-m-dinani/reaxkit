"""Incremental, explicitly statistical summaries of faulthandler samples."""

import json
import os
from pathlib import Path
import re


class StackSummary:
    def __init__(self, interval, main_thread):
        self.interval = float(interval)
        self.main_thread = int(main_thread)
        self.samples = 0
        self.thread = None
        self.first_frame = True
        self.seen = set()
        self.counts = {}

    def feed(self, line):
        if line.startswith("Timeout ("):
            self.samples += 1
            self.thread = None
            self.seen.clear()
            return
        thread = re.match(r"(?:Current thread|Thread) (0x[0-9a-fA-F]+)", line)
        if thread:
            self.thread = int(thread.group(1), 16)
            self.first_frame = True
            return
        frame = re.match(r'  File "(.*)", line \d+ in (.*)', line)
        if frame is None or self.thread is None or not self.samples:
            return
        key = (self.thread, frame.group(1), frame.group(2))
        counts = self.counts.setdefault(key, [0, 0])
        if self.first_frame:
            counts[0] += 1
            self.first_frame = False
        if key not in self.seen:
            counts[1] += 1
            self.seen.add(key)

    def rows(self):
        result = []
        for (thread, filename, function), (self_samples, inclusive_samples) in self.counts.items():
            result.append(dict(thread="main" if thread == self.main_thread else hex(thread),
                file=filename, function=function, self_samples=self_samples,
                inclusive_samples=inclusive_samples,
                estimated_self_seconds=self_samples*self.interval,
                estimated_inclusive_seconds=inclusive_samples*self.interval))
        return sorted(result, key=lambda row: (row["thread"] != "main", -row["self_samples"],
                      -row["inclusive_samples"], row["thread"], row["file"], row["function"]))

    def write(self, directory):
        rows = self.rows()
        explanation = (
            "Samples are observations, NOT invocation counts. Estimated seconds = samples x interval.\n"
            "Self: top Python frame; inclusive: present anywhere in that thread's stack (recursion counted once).\n"
            "These are sampled thread wall-time estimates, NOT CPU timings. Waiting threads are included.\n"
            "Inclusive values overlap; different threads overlap. Do not sum them as command elapsed time.\n"
            "Short work between samples and the final unsampled interval are not attributed. Native work is\n"
            "attributed to its Python caller. This is a live partial summary, not a completion indicator.\n"
        )
        table = ["STACK SAMPLING SUMMARY", f"Stack dumps observed: {self.samples}; interval: {self.interval:g} seconds",
                 explanation, f"{'Thread':<18} {'Self samples':>12} {'Incl samples':>12} {'Est self s':>12} {'Est incl s':>12}  Function (source)",
                 "-" * 110]
        for row in rows:
            source = row["file"].replace("\\", "/")
            if "/reaxkit/" in source:
                source = "reaxkit/" + source.rsplit("/reaxkit/", 1)[1]
            else:
                source = Path(source).name
            table.append(f"{row['thread']:<18} {row['self_samples']:>12} {row['inclusive_samples']:>12} "
                         f"{row['estimated_self_seconds']:>12.2f} {row['estimated_inclusive_seconds']:>12.2f}  "
                         f"{row['function']} ({source})")
        if not rows:
            table.append("No Python stack frames sampled yet; no function timings can be inferred.")
        payloads = {
            "stack_summary.txt": "\n".join(table) + "\n",
            "stack_summary.json": json.dumps(dict(samples=self.samples, interval_seconds=self.interval,
                method="periodic stack sampling", limitations=explanation, functions=rows), indent=2) + "\n",
        }
        for name, payload in payloads.items():
            path = directory / name
            temporary = path.with_suffix(".tmp")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
