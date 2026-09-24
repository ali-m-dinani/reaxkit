"""Bounded state for connectivity statistics and ordered bond events."""

from collections import deque

import numpy as np
import pandas as pd


PAIR_COLUMNS = ["source", "source_type", "destination", "destination_type"]
EVENT_COLUMNS = PAIR_COLUMNS + ["event", "frame_idx", "iter", "bo_at_event", "threshold", "hysteresis"]


class ConnectionStatistics:
    def __init__(self, how):
        self.how = how
        self.pairs = {}

    def add(self, table):
        for source, source_type, destination, destination_type, value in table[PAIR_COLUMNS + ["BO"]].itertuples(index=False, name=None):
            key = (source, source_type, destination, destination_type)
            count, valid, total, compensation, maximum = self.pairs.get(key, (0, 0, 0.0, 0.0, np.nan))
            count += 1
            if not np.isnan(value):
                valid += 1
                if np.isfinite(total) and np.isfinite(value):
                    corrected = value - compensation
                    updated = total + corrected
                    compensation = (updated - total) - corrected
                    total = updated
                else:
                    total += value
                    compensation = 0.0
                maximum = value if np.isnan(maximum) else max(maximum, value)
            self.pairs[key] = count, valid, total, compensation, maximum

    def merge(self, other):
        if self.how != other.how:
            raise ValueError("Cannot merge different connectivity statistics.")
        for key, (count, valid, total, compensation, maximum) in other.pairs.items():
            if key not in self.pairs:
                self.pairs[key] = (count, valid, total, compensation, maximum)
                continue
            n, v, s, c, m = self.pairs[key]
            correction = total - c
            updated = s + correction
            c = (updated - s) - correction
            maximum = maximum if np.isnan(m) else m if np.isnan(maximum) else max(m, maximum)
            self.pairs[key] = n + count, v + valid, updated, c, maximum

    def finalize(self):
        rows = []
        for key, (count, valid, total, _, maximum) in self.pairs.items():
            value = count if self.how == "count" else maximum if self.how == "max" else total / valid if valid else np.nan
            rows.append((*key, value))
        return pd.DataFrame(rows, columns=PAIR_COLUMNS + ["value"]).sort_values(
            ["source", "destination"], kind="stable").reset_index(drop=True)


class BondTraceState:
    """Centered smoothing and hysteresis with bounded delayed confirmation.

    Short runs are assigned to the preceding confirmed run, matching
    ``clean_flicker``. Leading short runs never produce an event.
    """

    def __init__(self, request):
        self.request = request
        self.window = max(1, int(request.window))
        self.samples = deque()
        self.offset = 0
        self.received = 0
        self.next_index = 0
        self.smoothed = np.nan
        self.old_weight = 1.0
        self.raw_state = None
        self.run_state = None
        self.run_length = 0
        self.run_start = None
        self.confirmed = None

    def _state(self, sample, value):
        threshold = float(self.request.threshold)
        half = max(0.0, float(self.request.hysteresis)) / 2
        if self.raw_state is None:
            self.raw_state = bool(value >= threshold + half)
        # Match the serial helper's update of the first sample too.
        if not self.raw_state and value >= threshold + half:
            self.raw_state = True
        elif self.raw_state and value <= threshold - half:
            self.raw_state = False
        if self.run_state != self.raw_state:
            self.run_state = self.raw_state
            self.run_length = 0
            self.run_start = (sample[0], sample[1], value)
        self.run_length += 1
        if self.run_length == max(1, int(self.request.min_run)):
            previous = self.confirmed
            self.confirmed = self.run_state
            if previous is not None and previous != self.confirmed:
                return (*self.run_start, "formation" if self.confirmed else "breakage")
        return None

    def add(self, frame, iteration, value):
        sample = (frame, iteration, value)
        if self.request.smooth is None:
            event = self._state(sample, value)
            return [] if event is None else [event]
        if self.request.smooth == "ema":
            alpha = self.request.ema_alpha
            alpha = 2.0 / (self.window + 1.0) if alpha is None else float(alpha)
            if not 0 < alpha <= 1:
                raise ValueError("alpha must satisfy 0 < alpha <= 1")
            if np.isnan(self.smoothed):
                self.smoothed = value
            else:
                self.old_weight *= 1 - alpha
                if not np.isnan(value):
                    self.smoothed = (self.old_weight * self.smoothed + alpha * value) / (self.old_weight + alpha)
                    self.old_weight = 1.0
            event = self._state(sample, self.smoothed)
            return [] if event is None else [event]
        self.samples.append(sample)
        self.received += 1
        return self._flush(final=False)

    def _flush(self, *, final):
        events = []
        future = (self.window - 1) // 2
        while self.next_index < self.received and (final or self.next_index + future < self.received):
            lower = max(0, self.next_index - self.window // 2)
            upper = min(self.received, self.next_index + future + 1)
            sample = self.samples[self.next_index - self.offset]
            values = np.array([self.samples[i - self.offset][2] for i in range(lower, upper)])
            finite = values[np.isfinite(values)]
            value = float(finite.mean()) if len(finite) else np.nan
            event = self._state(sample, value)
            if event is not None:
                events.append(event)
            self.next_index += 1
            keep_from = max(0, self.next_index - self.window // 2)
            while self.offset < keep_from:
                self.samples.popleft()
                self.offset += 1
        return events

    def finish(self):
        return self._flush(final=True) if self.request.smooth not in {None, "ema"} else []
