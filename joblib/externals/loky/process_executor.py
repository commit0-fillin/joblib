"""Implements ProcessPoolExecutor.

The follow diagram and text describe the data-flow through the system:

|======================= In-process =====================|== Out-of-process ==|

+----------+     +----------+       +--------+     +-----------+    +---------+
|          |  => | Work Ids |       |        |     | Call Q    |    | Process |
|          |     +----------+       |        |     +-----------+    |  Pool   |
|          |     | ...      |       |        |     | ...       |    +---------+
|          |     | 6        |    => |        |  => | 5, call() | => |         |
|          |     | 7        |       |        |     | ...       |    |         |
| Process  |     | ...      |       | Local  |     +-----------+    | Process |
|  Pool    |     +----------+       | Worker |                      |  #1..n  |
| Executor |                        | Thread |                      |         |
|          |     +----------- +     |        |     +-----------+    |         |
|          | <=> | Work Items | <=> |        | <=  | Result Q  | <= |         |
|          |     +------------+     |        |     +-----------+    |         |
|          |     | 6: call()  |     |        |     | ...       |    |         |
|          |     |    future  |     +--------+     | 4, result |    |         |
|          |     | ...        |                    | 3, except |    |         |
+----------+     +------------+                    +-----------+    +---------+

Executor.submit() called:
- creates a uniquely numbered _WorkItem and adds it to the "Work Items" dict
- adds the id of the _WorkItem to the "Work Ids" queue

Local worker thread:
- reads work ids from the "Work Ids" queue and looks up the corresponding
  WorkItem from the "Work Items" dict: if the work item has been cancelled then
  it is simply removed from the dict, otherwise it is repackaged as a
  _CallItem and put in the "Call Q". New _CallItems are put in the "Call Q"
  until "Call Q" is full. NOTE: the size of the "Call Q" is kept small because
  calls placed in the "Call Q" can no longer be cancelled with Future.cancel().
- reads _ResultItems from "Result Q", updates the future stored in the
  "Work Items" dict and deletes the dict entry

Process #1..n:
- reads _CallItems from "Call Q", executes the calls, and puts the resulting
  _ResultItems in "Result Q"
"""
__author__ = 'Thomas Moreau (thomas.moreau.2010@gmail.com)'
import os
import gc
import sys
import queue
import struct
import weakref
import warnings
import itertools
import traceback
import threading
from time import time, sleep
import multiprocessing as mp
from functools import partial
from pickle import PicklingError
from concurrent.futures import Executor
from concurrent.futures._base import LOGGER
from concurrent.futures.process import BrokenProcessPool as _BPPException
from multiprocessing.connection import wait
from ._base import Future
from .backend import get_context
from .backend.context import cpu_count, _MAX_WINDOWS_WORKERS
from .backend.queues import Queue, SimpleQueue
from .backend.reduction import set_loky_pickler, get_loky_pickler_name
from .backend.utils import kill_process_tree, get_exitcodes_terminated_worker
from .initializers import _prepare_initializer
MAX_DEPTH = int(os.environ.get('LOKY_MAX_DEPTH', 10))
_CURRENT_DEPTH = 0
_MEMORY_LEAK_CHECK_DELAY = 1.0
_MAX_MEMORY_LEAK_SIZE = int(300000000.0)
try:
    from psutil import Process
    _USE_PSUTIL = True
except ImportError:
    _USE_PSUTIL = False

class _ThreadWakeup:

    def __init__(self):
        self._closed = False
        self._reader, self._writer = mp.Pipe(duplex=False)

class _ExecutorFlags:
    """necessary references to maintain executor states without preventing gc

    It permits to keep the information needed by executor_manager_thread
    and crash_detection_thread to maintain the pool without preventing the
    garbage collection of unreferenced executors.
    """

    def __init__(self, shutdown_lock):
        self.shutdown = False
        self.broken = None
        self.kill_workers = False
        self.shutdown_lock = shutdown_lock
_global_shutdown = False
_global_shutdown_lock = threading.Lock()
_threads_wakeups = weakref.WeakKeyDictionary()
mp.util.register_after_fork(_threads_wakeups, lambda obj: obj.clear())
process_pool_executor_at_exit = None
EXTRA_QUEUED_CALLS = 1

class _RemoteTraceback(Exception):
    """Embed stringification of remote traceback in local traceback"""

    def __init__(self, tb=None):
        self.tb = f'\n"""\n{tb}"""'

    def __str__(self):
        return self.tb

class _ExceptionWithTraceback:

    def __init__(self, exc):
        tb = getattr(exc, '__traceback__', None)
        if tb is None:
            _, _, tb = sys.exc_info()
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        self.tb = tb

    def __reduce__(self):
        return (_rebuild_exc, (self.exc, self.tb))

class _WorkItem:
    __slots__ = ['future', 'fn', 'args', 'kwargs']

    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

class _ResultItem:

    def __init__(self, work_id, exception=None, result=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result

class _CallItem:

    def __init__(self, work_id, fn, args, kwargs):
        self.work_id = work_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.loky_pickler = get_loky_pickler_name()

    def __call__(self):
        set_loky_pickler(self.loky_pickler)
        return self.fn(*self.args, **self.kwargs)

    def __repr__(self):
        return f'CallItem({self.work_id}, {self.fn}, {self.args}, {self.kwargs})'

class _SafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""

    def __init__(self, max_size=0, ctx=None, pending_work_items=None, running_work_items=None, thread_wakeup=None, reducers=None):
        self.thread_wakeup = thread_wakeup
        self.pending_work_items = pending_work_items
        self.running_work_items = running_work_items
        super().__init__(max_size, reducers=reducers, ctx=ctx)

def _get_chunks(chunksize, *iterables):
    """Iterates over zip()ed iterables in chunks."""
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk

def _process_chunk(fn, chunk):
    """Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]

def _sendback_result(result_queue, work_id, result=None, exception=None):
    """Safely send back the given result or exception"""
    try:
        result_item = _ResultItem(work_id, exception, result)
        result_queue.put(result_item)
    except BaseException as e:
        exc = _ExceptionWithTraceback(e, e.__traceback__)
        result_item = _ResultItem(work_id, exc)
        result_queue.put(result_item)

def _process_worker(call_queue, result_queue, initializer, initargs, processes_management_lock, timeout, worker_exit_lock, current_depth):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A ctx.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A ctx.Queue of _ResultItems that will written
            to by the worker.
        initializer: A callable initializer, or None
        initargs: A tuple of args for the initializer
        processes_management_lock: A ctx.Lock avoiding worker timeout while
            some workers are being spawned.
        timeout: maximum time to wait for a new item in the call_queue. If that
            time is expired, the worker will shutdown.
        worker_exit_lock: Lock to avoid flagging the executor as broken on
            workers timeout.
        current_depth: Nested parallelism level, to avoid infinite spawning.
    """
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical('Exception in initializer:', exc_info=True)
            return

    while True:
        try:
            call_item = call_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            with worker_exit_lock:
                return
        if call_item is None:
            return
        try:
            r = call_item()
        except BaseException as e:
            exc = _ExceptionWithTraceback(e, e.__traceback__)
            _sendback_result(result_queue, call_item.work_id, exception=exc)
        else:
            _sendback_result(result_queue, call_item.work_id, result=r)
        del call_item

class _ExecutorManagerThread(threading.Thread):
    """Manages the communication between this process and the worker processes.

    The manager is run in a local thread.

    Args:
        executor: A reference to the ProcessPoolExecutor that owns
            this thread. A weakref will be own by the manager as well as
            references to internal objects used to introspect the state of
            the executor.
    """

    def __init__(self, executor):
        self.thread_wakeup = executor._executor_manager_thread_wakeup
        self.shutdown_lock = executor._shutdown_lock

        def weakref_cb(_, thread_wakeup=self.thread_wakeup, shutdown_lock=self.shutdown_lock):
            if mp is not None:
                mp.util.debug('Executor collected: triggering callback for QueueManager wakeup')
            with shutdown_lock:
                thread_wakeup.wakeup()
        self.executor_reference = weakref.ref(executor, weakref_cb)
        self.executor_flags = executor._flags
        self.processes = executor._processes
        self.call_queue = executor._call_queue
        self.result_queue = executor._result_queue
        self.work_ids_queue = executor._work_ids
        self.pending_work_items = executor._pending_work_items
        self.running_work_items = executor._running_work_items
        self.processes_management_lock = executor._processes_management_lock
        super().__init__(name='ExecutorManagerThread')
        if sys.version_info < (3, 9):
            self.daemon = True
_system_limits_checked = False
_system_limited = None

def _chain_from_iterable_of_lists(iterable):
    """
    Specialized implementation of itertools.chain.from_iterable.
    Each item in *iterable* should be a list.  This function is
    careful not to keep references to yielded objects.
    """
    for element in iterable:
        yield from element

class LokyRecursionError(RuntimeError):
    """A process tries to spawn too many levels of nested processes."""

class BrokenProcessPool(_BPPException):
    """
    Raised when the executor is broken while a future was in the running state.
    The cause can an error raised when unpickling the task in the worker
    process or when unpickling the result value in the parent process. It can
    also be caused by a worker process being terminated unexpectedly.
    """

class TerminatedWorkerError(BrokenProcessPool):
    """
    Raised when a process in a ProcessPoolExecutor terminated abruptly
    while a future was in the running state.
    """
BrokenExecutor = BrokenProcessPool

class ShutdownExecutorError(RuntimeError):
    """
    Raised when a ProcessPoolExecutor is shutdown while a future was in the
    running or pending state.
    """

class ProcessPoolExecutor(Executor):
    _at_exit = None

    def __init__(self, max_workers=None, job_reducers=None, result_reducers=None, timeout=None, context=None, initializer=None, initargs=(), env=None):
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: int, optional (default: cpu_count())
                The maximum number of processes that can be used to execute the
                given calls. If None or not given then as many worker processes
                will be created as the number of CPUs the current process
                can use.
            job_reducers, result_reducers: dict(type: reducer_func)
                Custom reducer for pickling the jobs and the results from the
                Executor. If only `job_reducers` is provided, `result_reducer`
                will use the same reducers
            timeout: int, optional (default: None)
                Idle workers exit after timeout seconds. If a new job is
                submitted after the timeout, the executor will start enough
                new Python processes to make sure the pool of workers is full.
            context: A multiprocessing context to launch the workers. This
                object should provide SimpleQueue, Queue and Process.
            initializer: An callable used to initialize worker processes.
            initargs: A tuple of arguments to pass to the initializer.
            env: A dict of environment variable to overwrite in the child
                process. The environment variables are set before any module is
                loaded. Note that this only works with the loky context.
        """
        _check_system_limits()
        if max_workers is None:
            self._max_workers = cpu_count()
        else:
            if max_workers <= 0:
                raise ValueError('max_workers must be greater than 0')
            self._max_workers = max_workers
        if sys.platform == 'win32' and self._max_workers > _MAX_WINDOWS_WORKERS:
            warnings.warn(f'On Windows, max_workers cannot exceed {_MAX_WINDOWS_WORKERS} due to limitations of the operating system.')
            self._max_workers = _MAX_WINDOWS_WORKERS
        if context is None:
            context = get_context()
        self._context = context
        self._env = env
        self._initializer, self._initargs = _prepare_initializer(initializer, initargs)
        _check_max_depth(self._context)
        if result_reducers is None:
            result_reducers = job_reducers
        self._timeout = timeout
        self._executor_manager_thread = None
        self._processes = {}
        self._processes = {}
        self._queue_count = 0
        self._pending_work_items = {}
        self._running_work_items = []
        self._work_ids = queue.Queue()
        self._processes_management_lock = self._context.Lock()
        self._executor_manager_thread = None
        self._shutdown_lock = threading.Lock()
        self._executor_manager_thread_wakeup = _ThreadWakeup()
        self._flags = _ExecutorFlags(self._shutdown_lock)
        self._setup_queues(job_reducers, result_reducers)
        mp.util.debug('ProcessPoolExecutor is setup')

    def _ensure_executor_running(self):
        """ensures all workers and management thread are running"""
        with self._processes_management_lock:
            if len(self._processes) == 0:
                self._start_processes()
            if self._executor_manager_thread is None:
                self._start_executor_manager_thread()
    submit.__doc__ = Executor.submit.__doc__

    def map(self, fn, *iterables, **kwargs):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a
                time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        timeout = kwargs.get('timeout', None)
        chunksize = kwargs.get('chunksize', 1)

        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        results = super().map(fn, *iterables, timeout=timeout)
        return _chain_from_iterable_of_lists(results)
    shutdown.__doc__ = Executor.shutdown.__doc__
