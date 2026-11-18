"""Thread pool that degrades to serial where threads can't spawn.

Pyodide (the Worker runtime) reports sys.platform == "emscripten" and has
no threads — ThreadPoolExecutor's constructor succeeds but submit() dies
with "can't start new thread". _Serial keeps the same submit/Future
contract so as_completed() and the with-block around it need no changes:
results are identical, just serial.
"""

import sys
from concurrent.futures import Future, ThreadPoolExecutor

_NO_THREADS = sys.platform == "emscripten"


def executor(max_workers: int):
    return _Serial() if _NO_THREADS else ThreadPoolExecutor(max_workers)


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *a, **kw):
        f = Future()
        try:
            f.set_result(fn(*a, **kw))
        except BaseException as e:  # noqa: BLE001 — futures capture anything, result() re-raises
            f.set_exception(e)
        return f
