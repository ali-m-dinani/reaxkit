"""Opt-in deterministic profiling for the command process and its Python threads."""

import json
import sys
import threading


class CommandProfile:
    def __init__(self, clock="wall"):
        try:
            import yappi
        except ImportError as exc:
            raise ValueError('Profiling requires Yappi: install reaxkit[profiling] with the command interpreter.') from exc
        if clock not in {"wall", "cpu"}:
            raise ValueError("Profile clock must be wall or cpu.")
        if yappi.is_running() or sys.getprofile() or threading.getprofile() or yappi.get_func_stats():
            raise ValueError("Another profiler or retained Yappi session exists; use a fresh command process.")
        self.profiler = yappi
        self.clock = clock
        self.previous_clock = yappi.get_clock_type()
        self.active = False

    def start(self):
        self.profiler.set_clock_type(self.clock)
        self.profiler.start(builtins=True, profile_threads=True)
        self.active = True

    def finish(self, directory):
        if not self.active:
            return
        self.profiler.stop()
        self.active = False
        try:
            stats = self.profiler.get_func_stats()
            stats.save(str(directory / "profile.pstat"), type="pstat")
            functions = [dict(function=entry.name, file=entry.module, line=entry.lineno,
                              calls=entry.ncall, primitive_calls=entry.nactualcall,
                              self_seconds=entry.tsub, cumulative_seconds=entry.ttot)
                         for entry in stats]
            functions.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
            threads = [dict(id=entry.id, name=entry.name, seconds=entry.ttot)
                       for entry in self.profiler.get_thread_stats()]
            metadata = dict(profiler="yappi", clock=self.clock, functions=functions, threads=threads,
                            coverage="Command process Python threads and visible builtins; no subprocesses or native internal stacks.")
            (directory / "profile.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            with (directory / "profile_summary.txt").open("w", encoding="utf-8") as stream:
                stream.write(f"Deterministic Yappi profile | clock: {self.clock} | seconds\n")
                stream.write("Calls are recorded events, not stack samples. Timings include profiler overhead.\n")
                stream.write("Self excludes callees; cumulative includes callees. Recursive calls are included in Calls.\n")
                stream.write("Wall time includes waiting. Inclusive/thread times overlap; do not sum as job elapsed time.\n")
                stream.write(metadata["coverage"] + "\n\n")
                stream.write(f"{'Calls':>12} {'Primitive':>12} {'Self (s)':>14} {'Cumulative (s)':>16}  Function (source)\n")
                for row in functions:
                    stream.write(f"{row['calls']:12d} {row['primitive_calls']:12d} {row['self_seconds']:14.6f} "
                                 f"{row['cumulative_seconds']:16.6f}  {row['function']} ({row['file']}:{row['line']})\n")
                stream.write("\nThread timings\n")
                for row in threads:
                    stream.write(f"{row['id']:8d} {row['seconds']:14.6f}  {row['name']}\n")
        finally:
            self.profiler.clear_stats()
            self.profiler.set_clock_type(self.previous_clock)
