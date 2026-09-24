"""Owned disk-backed coordinates for optional post-analysis trajectory export."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import pickle
import numpy as np


class TrajectorySpool:
    def __init__(self):
        self.directory = TemporaryDirectory(prefix="reaxkit-result-trajectory-")
        self.path = Path(self.directory.name) / "positions.bin"
        self.file = self.path.open("w+b")
        self.template = None
        self.records = {}
        self.mapping = None
        self.labels = None
        self.label_file = (Path(self.directory.name) / "labels.pickle").open("w+b")
        self.label_width = 0

    def append(self, source, data):
        if self.template is None:
            self.template = replace(data, positions=data.positions[:0], iterations=None, simulation=None,
                                    source_frame_indices=None, atom_labels=None)
        elif tuple(data.atom_ids) != tuple(self.template.atom_ids):
            raise ValueError("Atom identities changed during trajectory export spooling.")
        xyz = np.asarray(data.positions[0], dtype=np.float64)
        self.file.seek(source * xyz.nbytes)
        self.file.write(xyz.tobytes())
        if data.atom_labels is not None:
            labels = np.asarray(data.atom_labels[0]).astype(str)
            self.label_width = max(self.label_width, max(map(len, labels), default=1))
        else:
            labels = np.asarray(data.elements).astype(str)
        pickle.dump((source, labels), self.label_file)
        simulation = data.simulation
        self.records[source] = (int(data.iterations[0]) if data.iterations is not None else source,
                               None if simulation is None or simulation.cell_lengths is None else simulation.cell_lengths[0].copy(),
                               None if simulation is None or simulation.cell_angles is None else simulation.cell_angles[0].copy(),
                               None if simulation is None or simulation.time is None else float(simulation.time[0]))

    def finish(self):
        from reaxkit.domain.data_models import SimulationData
        self.file.close()
        count = max(self.records) + 1
        self.mapping = np.memmap(self.path, mode="r", dtype=np.float64, shape=(count, len(self.template.atom_ids), 3))
        iterations = np.zeros(count, dtype=int)
        lengths = np.full((count, 3), np.nan)
        angles = np.full((count, 3), 90.)
        have_cells = False
        times = np.full(count, np.nan)
        have_time = False
        for source, (iteration, cell, angle, time) in self.records.items():
            iterations[source] = iteration
            if cell is not None:
                lengths[source], have_cells = cell, True
            if angle is not None:
                angles[source] = angle
            if time is not None:
                times[source], have_time = time, True
        if self.label_width:
            self.label_width = max(self.label_width, max(map(len, self.template.elements), default=1))
            self.labels = np.memmap(Path(self.directory.name) / "labels.bin", mode="w+",
                                   dtype=f"U{self.label_width}", shape=(count, len(self.template.atom_ids)))
            self.label_file.seek(0)
            for _ in self.records:
                source, labels = pickle.load(self.label_file)
                self.labels[source] = labels
            self.labels.flush()
        self.label_file.close()
        simulation = SimulationData(atom_ids=self.template.atom_ids, iterations=iterations,
                                     cell_lengths=lengths if have_cells else None, cell_angles=angles if have_cells else None,
                                     time=times if have_time else None)
        return replace(self.template, positions=self.mapping, iterations=iterations, simulation=simulation,
                       source_frame_indices=np.arange(count), atom_labels=self.labels)

    def close(self):
        self.file.close()
        self.label_file.close()
        if self.mapping is not None:
            self.mapping._mmap.close()
            self.mapping = None
        if self.labels is not None:
            self.labels._mmap.close()
            self.labels = None
        self.directory.cleanup()

    def __del__(self):
        try:
            self.close()
        except (OSError, AttributeError):
            pass
