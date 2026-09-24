"""Read a dependency once without retaining the preceding trajectory in RAM."""

from contextlib import contextmanager
import pickle
from tempfile import TemporaryFile

from reaxkit.core.runtime.frame_tables import source_frame_index


@contextmanager
def reference_frames(frames, reference_frame, *, index_fn=source_frame_index):
    """Yield a reference and replayable selected source after finding it.

    Readers that can seek supply dependencies first. Forward-only readers spool
    the prefix to local temporary storage; source files are still read once.
    The spool is private to this process and is removed even on cancellation.
    """
    iterator = iter(frames)
    try:
        with TemporaryFile() as prefix:
            reference = None
            found_at = -1
            for index, data in enumerate(iterator):
                source = index_fn(data, index)
                if source == int(reference_frame):
                    reference, found_at = data, index
                    break
                pickle.dump((source, data), prefix, protocol=pickle.HIGHEST_PROTOCOL)
            if reference is None:
                raise ValueError(f"Reference frame {reference_frame} was not present in the input stream.")
            prefix.seek(0)

            def replay():
                for _ in range(found_at):
                    yield pickle.load(prefix)
                pending_reference = True
                for index, data in enumerate(iterator, found_at + 1):
                    source = index_fn(data, index)
                    if pending_reference and source > int(reference_frame):
                        yield int(reference_frame), reference
                        pending_reference = False
                    yield source, data
                if pending_reference:
                    yield int(reference_frame), reference

            yield reference, replay()
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()
