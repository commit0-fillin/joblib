from __future__ import print_function, division, absolute_import
import asyncio
import concurrent.futures
import contextlib
import time
from uuid import uuid4
import weakref
from .parallel import parallel_config
from .parallel import AutoBatchingMixin, ParallelBackendBase
from ._utils import _TracebackCapturingWrapper, _retrieve_traceback_capturing_wrapped_call
try:
    import dask
    import distributed
except ImportError:
    dask = None
    distributed = None
if dask is not None and distributed is not None:
    from dask.utils import funcname
    from dask.sizeof import sizeof
    from dask.distributed import Client, as_completed, get_client, secede, rejoin
    from distributed.utils import thread_state
    try:
        from distributed.utils import TimeoutError as _TimeoutError
    except ImportError:
        from tornado.gen import TimeoutError as _TimeoutError

class _WeakKeyDictionary:
    """A variant of weakref.WeakKeyDictionary for unhashable objects.

    This datastructure is used to store futures for broadcasted data objects
    such as large numpy arrays or pandas dataframes that are not hashable and
    therefore cannot be used as keys of traditional python dicts.

    Furthermore using a dict with id(array) as key is not safe because the
    Python is likely to reuse id of recently collected arrays.
    """

    def __init__(self):
        self._data = {}

    def __getitem__(self, obj):
        ref, val = self._data[id(obj)]
        if ref() is not obj:
            raise KeyError(obj)
        return val

    def __setitem__(self, obj, value):
        key = id(obj)
        try:
            ref, _ = self._data[key]
            if ref() is not obj:
                raise KeyError(obj)
        except KeyError:

            def on_destroy(_):
                del self._data[key]
            ref = weakref.ref(obj, on_destroy)
        self._data[key] = (ref, value)

    def __len__(self):
        return len(self._data)

def _make_tasks_summary(tasks):
    """Summarize of list of (func, args, kwargs) function calls"""
    num_tasks = len(tasks)
    if num_tasks == 0:
        return 0, False, ''
    
    first_func = tasks[0][0]
    mixed = any(task[0] != first_func for task in tasks[1:])
    funcname = funcname(first_func)
    
    return num_tasks, mixed, funcname

class Batch:
    """dask-compatible wrapper that executes a batch of tasks"""

    def __init__(self, tasks):
        self._num_tasks, self._mixed, self._funcname = _make_tasks_summary(tasks)

    def __call__(self, tasks=None):
        results = []
        with parallel_config(backend='dask'):
            for func, args, kwargs in tasks:
                results.append(func(*args, **kwargs))
            return results

    def __repr__(self):
        descr = f'batch_of_{self._funcname}_{self._num_tasks}_calls'
        if self._mixed:
            descr = 'mixed_' + descr
        return descr

class DaskDistributedBackend(AutoBatchingMixin, ParallelBackendBase):
    MIN_IDEAL_BATCH_DURATION = 0.2
    MAX_IDEAL_BATCH_DURATION = 1.0
    supports_retrieve_callback = True
    default_n_jobs = -1

    def __init__(self, scheduler_host=None, scatter=None, client=None, loop=None, wait_for_workers_timeout=10, **submit_kwargs):
        super().__init__()
        if distributed is None:
            msg = "You are trying to use 'dask' as a joblib parallel backend but dask is not installed. Please install dask to fix this error."
            raise ValueError(msg)
        if client is None:
            if scheduler_host:
                client = Client(scheduler_host, loop=loop, set_as_default=False)
            else:
                try:
                    client = get_client()
                except ValueError as e:
                    msg = "To use Joblib with Dask first create a Dask Client\n\n    from dask.distributed import Client\n    client = Client()\nor\n    client = Client('scheduler-address:8786')"
                    raise ValueError(msg) from e
        self.client = client
        if scatter is not None and (not isinstance(scatter, (list, tuple))):
            raise TypeError('scatter must be a list/tuple, got `%s`' % type(scatter).__name__)
        if scatter is not None and len(scatter) > 0:
            self._scatter = list(scatter)
            scattered = self.client.scatter(scatter, broadcast=True)
            self.data_futures = {id(x): f for x, f in zip(scatter, scattered)}
        else:
            self._scatter = []
            self.data_futures = {}
        self.wait_for_workers_timeout = wait_for_workers_timeout
        self.submit_kwargs = submit_kwargs
        self.waiting_futures = as_completed([], loop=client.loop, with_results=True, raise_errors=False)
        self._results = {}
        self._callbacks = {}

    def __reduce__(self):
        return (DaskDistributedBackend, ())

    def abort_everything(self, ensure_ready=True):
        """ Tell the client to cancel any task submitted via this instance

        joblib.Parallel will never access those results
        """
        if ensure_ready:
            self.client.cancel(list(self._results.keys()))
        self._results.clear()
        self._callbacks.clear()
        # Reset the waiting_futures to an empty as_completed object
        self.waiting_futures = as_completed([], loop=self.client.loop, with_results=True, raise_errors=False)

    @contextlib.contextmanager
    def retrieval_context(self):
        """Override ParallelBackendBase.retrieval_context to avoid deadlocks.

        This removes thread from the worker's thread pool (using 'secede').
        Seceding avoids deadlock in nested parallelism settings.
        """
        if hasattr(thread_state, 'execution_state'):
            # We are inside a worker's thread, so secede to avoid deadlocks
            secede()
            try:
                yield
            finally:
                rejoin()
        else:
            # We are in the client's thread, no need to secede
            yield
