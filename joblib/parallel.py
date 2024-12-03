"""
Helpers for embarrassingly parallel code.
"""
from __future__ import division
import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext
from multiprocessing import TimeoutError
from ._multiprocessing_helpers import mp
from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import FallbackToBackend, MultiprocessingBackend, ThreadingBackend, SequentialBackend, LokyBackend
from ._utils import eval_expr, _Sentinel
from ._parallel_backends import AutoBatchingMixin
from ._parallel_backends import ParallelBackendBase
IS_PYPY = hasattr(sys, 'pypy_version_info')
BACKENDS = {'threading': ThreadingBackend, 'sequential': SequentialBackend}
DEFAULT_BACKEND = 'threading'
MAYBE_AVAILABLE_BACKENDS = {'multiprocessing', 'loky'}
if mp is not None:
    BACKENDS['multiprocessing'] = MultiprocessingBackend
    from .externals import loky
    BACKENDS['loky'] = LokyBackend
    DEFAULT_BACKEND = 'loky'
DEFAULT_THREAD_BACKEND = 'threading'
_backend = threading.local()

def _register_dask():
    """Register Dask Backend if called with parallel_config(backend="dask")"""
    from ._dask import DaskDistributedBackend
    register_parallel_backend('dask', DaskDistributedBackend)
EXTERNAL_BACKENDS = {'dask': _register_dask}
default_parallel_config = {'backend': _Sentinel(default_value=None), 'n_jobs': _Sentinel(default_value=None), 'verbose': _Sentinel(default_value=0), 'temp_folder': _Sentinel(default_value=None), 'max_nbytes': _Sentinel(default_value='1M'), 'mmap_mode': _Sentinel(default_value='r'), 'prefer': _Sentinel(default_value=None), 'require': _Sentinel(default_value=None)}
VALID_BACKEND_HINTS = ('processes', 'threads', None)
VALID_BACKEND_CONSTRAINTS = ('sharedmem', None)

def _get_config_param(param, context_config, key):
    """Return the value of a parallel config parameter

    Explicitly setting it in Parallel has priority over setting in a
    parallel_(config/backend) context manager.
    """
    if not isinstance(param, _Sentinel):
        return param
    return context_config.get(key, default_parallel_config[key])

def get_active_backend(prefer=default_parallel_config['prefer'], require=default_parallel_config['require'], verbose=default_parallel_config['verbose']):
    """Return the active default backend"""
    return _get_active_backend(prefer=prefer, require=require, verbose=verbose)[0]

def _get_active_backend(prefer=default_parallel_config['prefer'], require=default_parallel_config['require'], verbose=default_parallel_config['verbose']):
    """Return the active default backend"""
    context_config = getattr(_backend, 'config', default_parallel_config)
    backend = context_config['backend']
    if isinstance(backend, _Sentinel):
        backend = None
    if backend is None:
        backend = DEFAULT_BACKEND
        if require == 'sharedmem':
            backend = DEFAULT_THREAD_BACKEND
        elif prefer == 'threads':
            backend = DEFAULT_THREAD_BACKEND
        elif prefer == 'processes':
            backend = DEFAULT_BACKEND
    if isinstance(backend, str):
        backend_args = {}
        if backend == 'dask':
            _register_dask()
        if backend not in BACKENDS:
            raise ValueError(f"Invalid backend: {backend}")
        backend = BACKENDS[backend](**backend_args)
    return backend, context_config

class parallel_config:
    """Set the default backend or configuration for :class:`~joblib.Parallel`.

    This is an alternative to directly passing keyword arguments to the
    :class:`~joblib.Parallel` class constructor. It is particularly useful when
    calling into library code that uses joblib internally but does not expose
    the various parallel configuration arguments in its own API.

    Parameters
    ----------
    backend: str or ParallelBackendBase instance, default=None
        If ``backend`` is a string it must match a previously registered
        implementation using the :func:`~register_parallel_backend` function.

        By default the following backends are available:

        - 'loky': single-host, process-based parallelism (used by default),
        - 'threading': single-host, thread-based parallelism,
        - 'multiprocessing': legacy single-host, process-based parallelism.

        'loky' is recommended to run functions that manipulate Python objects.
        'threading' is a low-overhead alternative that is most efficient for
        functions that release the Global Interpreter Lock: e.g. I/O-bound
        code or CPU-bound code in a few calls to native code that explicitly
        releases the GIL. Note that on some rare systems (such as pyodide),
        multiprocessing and loky may not be available, in which case joblib
        defaults to threading.

        In addition, if the ``dask`` and ``distributed`` Python packages are
        installed, it is possible to use the 'dask' backend for better
        scheduling of nested parallel calls without over-subscription and
        potentially distribute parallel calls over a networked cluster of
        several hosts.

        It is also possible to use the distributed 'ray' backend for
        distributing the workload to a cluster of nodes. See more details
        in the Examples section below.

        Alternatively the backend can be passed directly as an instance.

    n_jobs: int, default=None
        The maximum number of concurrently running jobs, such as the number
        of Python worker processes when ``backend="loky"`` or the size of the
        thread-pool when ``backend="threading"``.
        This argument is converted to an integer, rounded below for float.
        If -1 is given, `joblib` tries to use all CPUs. The number of CPUs
        ``n_cpus`` is obtained with :func:`~cpu_count`.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. For instance,
        using ``n_jobs=-2`` will result in all CPUs but one being used.
        This argument can also go above ``n_cpus``, which will cause
        oversubscription. In some cases, slight oversubscription can be
        beneficial, e.g., for tasks with large I/O operations.
        If 1 is given, no parallel computing code is used at all, and the
        behavior amounts to a simple python `for` loop. This mode is not
        compatible with `timeout`.
        None is a marker for 'unset' that will be interpreted as n_jobs=1
        unless the call is performed under a :func:`~parallel_config`
        context manager that sets another value for ``n_jobs``.
        If n_jobs = 0 then a ValueError is raised.

    verbose: int, default=0
        The verbosity level: if non zero, progress messages are
        printed. Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    temp_folder: str or None, default=None
        Folder to be used by the pool for memmapping large arrays
        for sharing memory with worker processes. If None, this will try in
        order:

        - a folder pointed by the ``JOBLIB_TEMP_FOLDER`` environment
          variable,
        - ``/dev/shm`` if the folder exists and is writable: this is a
          RAM disk filesystem available by default on modern Linux
          distributions,
        - the default system temporary folder that can be
          overridden with ``TMP``, ``TMPDIR`` or ``TEMP`` environment
          variables, typically ``/tmp`` under Unix operating systems.

    max_nbytes int, str, or None, optional, default='1M'
        Threshold on the size of arrays passed to the workers that
        triggers automated memory mapping in temp_folder. Can be an int
        in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
        Use None to disable memmapping of large arrays.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, default='r'
        Memmapping mode for numpy arrays passed to workers. None will
        disable memmapping, other modes defined in the numpy.memmap doc:
        https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        Also, see 'max_nbytes' parameter documentation for more details.

    prefer: str in {'processes', 'threads'} or None, default=None
        Soft hint to choose the default backend.
        The default process-based backend is 'loky' and the default
        thread-based backend is 'threading'. Ignored if the ``backend``
        parameter is specified.

    require: 'sharedmem' or None, default=None
        Hard constraint to select the backend. If set to 'sharedmem',
        the selected backend will be single-host and thread-based.

    inner_max_num_threads: int, default=None
        If not None, overwrites the limit set on the number of threads
        usable in some third-party library threadpools like OpenBLAS,
        MKL or OpenMP. This is only used with the ``loky`` backend.

    backend_params: dict
        Additional parameters to pass to the backend constructor when
        backend is a string.

    Notes
    -----
    Joblib tries to limit the oversubscription by limiting the number of
    threads usable in some third-party library threadpools like OpenBLAS, MKL
    or OpenMP. The default limit in each worker is set to
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be
    overwritten with the ``inner_max_num_threads`` argument which will be used
    to set this limit in the child processes.

    .. versionadded:: 1.3

    Examples
    --------
    >>> from operator import neg
    >>> with parallel_config(backend='threading'):
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    ...
    [-1, -2, -3, -4, -5]

    To use the 'ray' joblib backend add the following lines:

    >>> from ray.util.joblib import register_ray  # doctest: +SKIP
    >>> register_ray()  # doctest: +SKIP
    >>> with parallel_config(backend="ray"):  # doctest: +SKIP
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    [-1, -2, -3, -4, -5]

    """

    def __init__(self, backend=default_parallel_config['backend'], *, n_jobs=default_parallel_config['n_jobs'], verbose=default_parallel_config['verbose'], temp_folder=default_parallel_config['temp_folder'], max_nbytes=default_parallel_config['max_nbytes'], mmap_mode=default_parallel_config['mmap_mode'], prefer=default_parallel_config['prefer'], require=default_parallel_config['require'], inner_max_num_threads=None, **backend_params):
        self.old_parallel_config = getattr(_backend, 'config', default_parallel_config)
        backend = self._check_backend(backend, inner_max_num_threads, **backend_params)
        new_config = {'n_jobs': n_jobs, 'verbose': verbose, 'temp_folder': temp_folder, 'max_nbytes': max_nbytes, 'mmap_mode': mmap_mode, 'prefer': prefer, 'require': require, 'backend': backend}
        self.parallel_config = self.old_parallel_config.copy()
        self.parallel_config.update({k: v for k, v in new_config.items() if not isinstance(v, _Sentinel)})
        setattr(_backend, 'config', self.parallel_config)

    def __enter__(self):
        return self.parallel_config

    def __exit__(self, type, value, traceback):
        self.unregister()

class parallel_backend(parallel_config):
    """Change the default backend used by Parallel inside a with block.

    .. warning::
        It is advised to use the :class:`~joblib.parallel_config` context
        manager instead, which allows more fine-grained control over the
        backend configuration.

    If ``backend`` is a string it must match a previously registered
    implementation using the :func:`~register_parallel_backend` function.

    By default the following backends are available:

    - 'loky': single-host, process-based parallelism (used by default),
    - 'threading': single-host, thread-based parallelism,
    - 'multiprocessing': legacy single-host, process-based parallelism.

    'loky' is recommended to run functions that manipulate Python objects.
    'threading' is a low-overhead alternative that is most efficient for
    functions that release the Global Interpreter Lock: e.g. I/O-bound code or
    CPU-bound code in a few calls to native code that explicitly releases the
    GIL. Note that on some rare systems (such as Pyodide),
    multiprocessing and loky may not be available, in which case joblib
    defaults to threading.

    You can also use the `Dask <https://docs.dask.org/en/stable/>`_ joblib
    backend to distribute work across machines. This works well with
    scikit-learn estimators with the ``n_jobs`` parameter, for example::

    >>> import joblib  # doctest: +SKIP
    >>> from sklearn.model_selection import GridSearchCV  # doctest: +SKIP
    >>> from dask.distributed import Client, LocalCluster # doctest: +SKIP

    >>> # create a local Dask cluster
    >>> cluster = LocalCluster()  # doctest: +SKIP
    >>> client = Client(cluster)  # doctest: +SKIP
    >>> grid_search = GridSearchCV(estimator, param_grid, n_jobs=-1)
    ... # doctest: +SKIP
    >>> with joblib.parallel_backend("dask", scatter=[X, y]):  # doctest: +SKIP
    ...     grid_search.fit(X, y)

    It is also possible to use the distributed 'ray' backend for distributing
    the workload to a cluster of nodes. To use the 'ray' joblib backend add
    the following lines::

     >>> from ray.util.joblib import register_ray  # doctest: +SKIP
     >>> register_ray()  # doctest: +SKIP
     >>> with parallel_backend("ray"):  # doctest: +SKIP
     ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
     [-1, -2, -3, -4, -5]

    Alternatively the backend can be passed directly as an instance.

    By default all available workers will be used (``n_jobs=-1``) unless the
    caller passes an explicit value for the ``n_jobs`` parameter.

    This is an alternative to passing a ``backend='backend_name'`` argument to
    the :class:`~Parallel` class constructor. It is particularly useful when
    calling into library code that uses joblib internally but does not expose
    the backend argument in its own API.

    >>> from operator import neg
    >>> with parallel_backend('threading'):
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    ...
    [-1, -2, -3, -4, -5]

    Joblib also tries to limit the oversubscription by limiting the number of
    threads usable in some third-party library threadpools like OpenBLAS, MKL
    or OpenMP. The default limit in each worker is set to
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be
    overwritten with the ``inner_max_num_threads`` argument which will be used
    to set this limit in the child processes.

    .. versionadded:: 0.10

    See Also
    --------
    joblib.parallel_config: context manager to change the backend
        configuration.
    """

    def __init__(self, backend, n_jobs=-1, inner_max_num_threads=None, **backend_params):
        super().__init__(backend=backend, n_jobs=n_jobs, inner_max_num_threads=inner_max_num_threads, **backend_params)
        if self.old_parallel_config is None:
            self.old_backend_and_jobs = None
        else:
            self.old_backend_and_jobs = (self.old_parallel_config['backend'], self.old_parallel_config['n_jobs'])
        self.new_backend_and_jobs = (self.parallel_config['backend'], self.parallel_config['n_jobs'])

    def __enter__(self):
        return self.new_backend_and_jobs
DEFAULT_MP_CONTEXT = None
if hasattr(mp, 'get_context'):
    method = os.environ.get('JOBLIB_START_METHOD', '').strip() or None
    if method is not None:
        DEFAULT_MP_CONTEXT = mp.get_context(method=method)

class BatchedCalls(object):
    """Wrap a sequence of (func, args, kwargs) tuples as a single callable"""

    def __init__(self, iterator_slice, backend_and_jobs, reducer_callback=None, pickle_cache=None):
        self.items = list(iterator_slice)
        self._size = len(self.items)
        self._reducer_callback = reducer_callback
        if isinstance(backend_and_jobs, tuple):
            self._backend, self._n_jobs = backend_and_jobs
        else:
            self._backend, self._n_jobs = (backend_and_jobs, None)
        self._pickle_cache = pickle_cache if pickle_cache is not None else {}

    def __call__(self):
        with parallel_config(backend=self._backend, n_jobs=self._n_jobs):
            return [func(*args, **kwargs) for func, args, kwargs in self.items]

    def __reduce__(self):
        if self._reducer_callback is not None:
            self._reducer_callback()
        return (BatchedCalls, (self.items, (self._backend, self._n_jobs), None, self._pickle_cache))

    def __len__(self):
        return self._size
TASK_DONE = 'Done'
TASK_ERROR = 'Error'
TASK_PENDING = 'Pending'

def cpu_count(only_physical_cores=False):
    """Return the number of CPUs.

    This delegates to loky.cpu_count that takes into account additional
    constraints such as Linux CFS scheduler quotas (typically set by container
    runtimes such as docker) and CPU affinity (for instance using the taskset
    command on Linux).

    If only_physical_cores is True, do not take hyperthreading / SMT logical
    cores into account.
    """
    return process_executor.cpu_count(only_physical_cores=only_physical_cores)

def _verbosity_filter(index, verbose):
    """ Returns False for indices increasingly apart, the distance
        depending on the value of verbose.

        We use a lag increasing as the square of index
    """
    if not verbose:
        return False
    elif verbose > 10:
        return True
    if index == 0:
        return True
    verbose = verbose ** 2
    return (index % verbose) == 0

def delayed(function):
    """Decorator used to capture the arguments of a function."""
    def delayed_function(*args, **kwargs):
        return function, args, kwargs
    return delayed_function

class BatchCompletionCallBack(object):
    """Callback to keep track of completed results and schedule the next tasks.

    This callable is executed by the parent process whenever a worker process
    has completed a batch of tasks.

    It is used for progress reporting, to update estimate of the batch
    processing duration and to schedule the next batch of tasks to be
    processed.

    It is assumed that this callback will always be triggered by the backend
    right after the end of a task, in case of success as well as in case of
    failure.
    """

    def __init__(self, dispatch_timestamp, batch_size, parallel):
        self.dispatch_timestamp = dispatch_timestamp
        self.batch_size = batch_size
        self.parallel = parallel
        self.parallel_call_id = parallel._call_id
        self.job = None
        if not parallel._backend.supports_retrieve_callback:
            self.status = None
        else:
            self.status = TASK_PENDING

    def register_job(self, job):
        """Register the object returned by `apply_async`."""
        pass

    def get_result(self, timeout):
        """Returns the raw result of the task that was submitted.

        If the task raised an exception rather than returning, this same
        exception will be raised instead.

        If the backend supports the retrieval callback, it is assumed that this
        method is only called after the result has been registered. It is
        ensured by checking that `self.status(timeout)` does not return
        TASK_PENDING. In this case, `get_result` directly returns the
        registered result (or raise the registered exception).

        For other backends, there are no such assumptions, but `get_result`
        still needs to synchronously retrieve the result before it can
        return it or raise. It will block at most `self.timeout` seconds
        waiting for retrieval to complete, after that it raises a TimeoutError.
        """
        pass

    def get_status(self, timeout):
        """Get the status of the task.

        This function also checks if the timeout has been reached and register
        the TimeoutError outcome when it is the case.
        """
        pass

    def __call__(self, out):
        """Function called by the callback thread after a job is completed."""
        if not self.parallel._backend.supports_retrieve_callback:
            self._dispatch_new()
            return
        with self.parallel._lock:
            if self.parallel._call_id != self.parallel_call_id:
                return
            if self.parallel._aborting:
                return
            job_succeeded = self._retrieve_result(out)
            if not self.parallel.return_ordered:
                self.parallel._jobs.append(self)
        if job_succeeded:
            self._dispatch_new()

    def _dispatch_new(self):
        """Schedule the next batch of tasks to be processed."""
        pass

    def _retrieve_result(self, out):
        """Fetch and register the outcome of a task.

        Return True if the task succeeded, False otherwise.
        This function is only called by backends that support retrieving
        the task result in the callback thread.
        """
        pass

    def _register_outcome(self, outcome):
        """Register the outcome of a task.

        This method can be called only once, future calls will be ignored.
        """
        pass

def register_parallel_backend(name, factory, make_default=False):
    """Register a new Parallel backend factory.

    The new backend can then be selected by passing its name as the backend
    argument to the :class:`~Parallel` class. Moreover, the default backend can
    be overwritten globally by setting make_default=True.

    The factory can be any callable that takes no argument and return an
    instance of ``ParallelBackendBase``.

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    .. versionadded:: 0.10
    """
    pass

def effective_n_jobs(n_jobs=-1):
    """Determine the number of jobs that can actually run in parallel

    n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
    means requesting all available workers for instance matching the number of
    CPU cores on the worker host(s).

    This method should return a guesstimate of the number of workers that can
    actually perform work concurrently with the currently enabled default
    backend. The primary use case is to make it possible for the caller to know
    in how many chunks to slice the work.

    In general working on larger data chunks is more efficient (less scheduling
    overhead and better use of CPU cache prefetching heuristics) as long as all
    the workers have enough work to do.

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    .. versionadded:: 0.10
    """
    pass

class Parallel(Logger):
    """ Helper class for readable parallel mapping.

        Read more in the :ref:`User Guide <parallel>`.

        Parameters
        ----------
        n_jobs: int, default=None
            The maximum number of concurrently running jobs, such as the number
            of Python worker processes when ``backend="loky"`` or the size of
            the thread-pool when ``backend="threading"``.
            This argument is converted to an integer, rounded below for float.
            If -1 is given, `joblib` tries to use all CPUs. The number of CPUs
            ``n_cpus`` is obtained with :func:`~cpu_count`.
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. For instance,
            using ``n_jobs=-2`` will result in all CPUs but one being used.
            This argument can also go above ``n_cpus``, which will cause
            oversubscription. In some cases, slight oversubscription can be
            beneficial, e.g., for tasks with large I/O operations.
            If 1 is given, no parallel computing code is used at all, and the
            behavior amounts to a simple python `for` loop. This mode is not
            compatible with ``timeout``.
            None is a marker for 'unset' that will be interpreted as n_jobs=1
            unless the call is performed under a :func:`~parallel_config`
            context manager that sets another value for ``n_jobs``.
            If n_jobs = 0 then a ValueError is raised.
        backend: str, ParallelBackendBase instance or None, default='loky'
            Specify the parallelization backend implementation.
            Supported backends are:

            - "loky" used by default, can induce some
              communication and memory overhead when exchanging input and
              output data with the worker Python processes. On some rare
              systems (such as Pyiodide), the loky backend may not be
              available.
            - "multiprocessing" previous process-based backend based on
              `multiprocessing.Pool`. Less robust than `loky`.
            - "threading" is a very low-overhead backend but it suffers
              from the Python Global Interpreter Lock if the called function
              relies a lot on Python objects. "threading" is mostly useful
              when the execution bottleneck is a compiled extension that
              explicitly releases the GIL (for instance a Cython loop wrapped
              in a "with nogil" block or an expensive call to a library such
              as NumPy).
            - finally, you can register backends by calling
              :func:`~register_parallel_backend`. This will allow you to
              implement a backend of your liking.

            It is not recommended to hard-code the backend name in a call to
            :class:`~Parallel` in a library. Instead it is recommended to set
            soft hints (prefer) or hard constraints (require) so as to make it
            possible for library users to change the backend from the outside
            using the :func:`~parallel_config` context manager.
        return_as: str in {'list', 'generator', 'generator_unordered'}, default='list'
            If 'list', calls to this instance will return a list, only when
            all results have been processed and retrieved.
            If 'generator', it will return a generator that yields the results
            as soon as they are available, in the order the tasks have been
            submitted with.
            If 'generator_unordered', the generator will immediately yield
            available results independently of the submission order. The output
            order is not deterministic in this case because it depends on the
            concurrency of the workers.
        prefer: str in {'processes', 'threads'} or None, default=None
            Soft hint to choose the default backend if no specific backend
            was selected with the :func:`~parallel_config` context manager.
            The default process-based backend is 'loky' and the default
            thread-based backend is 'threading'. Ignored if the ``backend``
            parameter is specified.
        require: 'sharedmem' or None, default=None
            Hard constraint to select the backend. If set to 'sharedmem',
            the selected backend will be single-host and thread-based even
            if the user asked for a non-thread based backend with
            :func:`~joblib.parallel_config`.
        verbose: int, default=0
            The verbosity level: if non zero, progress messages are
            printed. Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
        timeout: float or None, default=None
            Timeout limit for each task to complete.  If any task takes longer
            a TimeOutError will be raised. Only applied when n_jobs != 1
        pre_dispatch: {'all', integer, or expression, as in '3*n_jobs'}, default='2*n_jobs'
            The number of batches (of tasks) to be pre-dispatched.
            Default is '2*n_jobs'. When batch_size="auto" this is reasonable
            default and the workers should never starve. Note that only basic
            arithmetics are allowed here and no modules can be used in this
            expression.
        batch_size: int or 'auto', default='auto'
            The number of atomic tasks to dispatch at once to each
            worker. When individual evaluations are very fast, dispatching
            calls to workers can be slower than sequential computation because
            of the overhead. Batching fast computations together can mitigate
            this.
            The ``'auto'`` strategy keeps track of the time it takes for a
            batch to complete, and dynamically adjusts the batch size to keep
            the time on the order of half a second, using a heuristic. The
            initial batch size is 1.
            ``batch_size="auto"`` with ``backend="threading"`` will dispatch
            batches of a single task at a time as the threading backend has
            very little overhead and using larger batch size has not proved to
            bring any gain in that case.
        temp_folder: str or None, default=None
            Folder to be used by the pool for memmapping large arrays
            for sharing memory with worker processes. If None, this will try in
            order:

            - a folder pointed by the JOBLIB_TEMP_FOLDER environment
              variable,
            - /dev/shm if the folder exists and is writable: this is a
              RAM disk filesystem available by default on modern Linux
              distributions,
            - the default system temporary folder that can be
              overridden with TMP, TMPDIR or TEMP environment
              variables, typically /tmp under Unix operating systems.

            Only active when ``backend="loky"`` or ``"multiprocessing"``.
        max_nbytes int, str, or None, optional, default='1M'
            Threshold on the size of arrays passed to the workers that
            triggers automated memory mapping in temp_folder. Can be an int
            in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
            Use None to disable memmapping of large arrays.
            Only active when ``backend="loky"`` or ``"multiprocessing"``.
        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, default='r'
            Memmapping mode for numpy arrays passed to workers. None will
            disable memmapping, other modes defined in the numpy.memmap doc:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
            Also, see 'max_nbytes' parameter documentation for more details.

        Notes
        -----

        This object uses workers to compute in parallel the application of a
        function to many different arguments. The main functionality it brings
        in addition to using the raw multiprocessing or concurrent.futures API
        are (see examples for details):

        * More readable code, in particular since it avoids
          constructing list of arguments.

        * Easier debugging:
            - informative tracebacks even when the error happens on
              the client side
            - using 'n_jobs=1' enables to turn off parallel computing
              for debugging without changing the codepath
            - early capture of pickling errors

        * An optional progress meter.

        * Interruption of multiprocesses jobs with 'Ctrl-C'

        * Flexible pickling control for the communication to and from
          the worker processes.

        * Ability to use shared memory efficiently with worker
          processes for large numpy-based datastructures.

        Note that the intended usage is to run one call at a time. Multiple
        calls to the same Parallel object will result in a ``RuntimeError``

        Examples
        --------

        A simple example:

        >>> from math import sqrt
        >>> from joblib import Parallel, delayed
        >>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

        Reshaping the output when the function has several return
        values:

        >>> from math import modf
        >>> from joblib import Parallel, delayed
        >>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))
        >>> res, i = zip(*r)
        >>> res
        (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
        >>> i
        (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)

        The progress meter: the higher the value of `verbose`, the more
        messages:

        >>> from time import sleep
        >>> from joblib import Parallel, delayed
        >>> r = Parallel(n_jobs=2, verbose=10)(
        ...     delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP
        [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
        [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
        [Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished

        Traceback example, note how the line of the error is indicated
        as well as the values of the parameter passed to the function that
        triggered the exception, even though the traceback happens in the
        child process:

        >>> from heapq import nlargest
        >>> from joblib import Parallel, delayed
        >>> Parallel(n_jobs=2)(
        ... delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3))
        ... # doctest: +SKIP
        -----------------------------------------------------------------------
        Sub-process traceback:
        -----------------------------------------------------------------------
        TypeError                                      Mon Nov 12 11:37:46 2012
        PID: 12934                                Python 2.7.3: /usr/bin/python
        ........................................................................
        /usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)
            419         if n >= size:
            420             return sorted(iterable, key=key, reverse=True)[:n]
            421
            422     # When key is none, use simpler decoration
            423     if key is None:
        --> 424         it = izip(iterable, count(0,-1))           # decorate
            425         result = _nlargest(n, it)
            426         return map(itemgetter(0), result)          # undecorate
            427
            428     # General case, slowest method
         TypeError: izip argument #1 must support iteration
        _______________________________________________________________________


        Using pre_dispatch in a producer/consumer situation, where the
        data is generated on the fly. Note how the producer is first
        called 3 times before the parallel loop is initiated, and then
        called to generate new data on the fly:

        >>> from math import sqrt
        >>> from joblib import Parallel, delayed
        >>> def producer():
        ...     for i in range(6):
        ...         print('Produced %s' % i)
        ...         yield i
        >>> out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(
        ...     delayed(sqrt)(i) for i in producer()) #doctest: +SKIP
        Produced 0
        Produced 1
        Produced 2
        [Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
        Produced 3
        [Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
        Produced 4
        [Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
        Produced 5
        [Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
        [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s
        [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s finished

    """

    def __init__(self, n_jobs=default_parallel_config['n_jobs'], backend=default_parallel_config['backend'], return_as='list', verbose=default_parallel_config['verbose'], timeout=None, pre_dispatch='2 * n_jobs', batch_size='auto', temp_folder=default_parallel_config['temp_folder'], max_nbytes=default_parallel_config['max_nbytes'], mmap_mode=default_parallel_config['mmap_mode'], prefer=default_parallel_config['prefer'], require=default_parallel_config['require']):
        super().__init__()
        if n_jobs is None:
            n_jobs = default_parallel_config['n_jobs']
        active_backend, context_config = _get_active_backend(prefer=prefer, require=require, verbose=verbose)
        nesting_level = active_backend.nesting_level
        self.verbose = _get_config_param(verbose, context_config, 'verbose')
        self.timeout = timeout
        self.pre_dispatch = pre_dispatch
        if return_as not in {'list', 'generator', 'generator_unordered'}:
            raise ValueError(f'Expected `return_as` parameter to be a string equal to "list","generator" or "generator_unordered", but got {return_as} instead.')
        self.return_as = return_as
        self.return_generator = return_as != 'list'
        self.return_ordered = return_as != 'generator_unordered'
        self._backend_args = {k: _get_config_param(param, context_config, k) for param, k in [(max_nbytes, 'max_nbytes'), (temp_folder, 'temp_folder'), (mmap_mode, 'mmap_mode'), (prefer, 'prefer'), (require, 'require'), (verbose, 'verbose')]}
        if isinstance(self._backend_args['max_nbytes'], str):
            self._backend_args['max_nbytes'] = memstr_to_bytes(self._backend_args['max_nbytes'])
        self._backend_args['verbose'] = max(0, self._backend_args['verbose'] - 50)
        if DEFAULT_MP_CONTEXT is not None:
            self._backend_args['context'] = DEFAULT_MP_CONTEXT
        elif hasattr(mp, 'get_context'):
            self._backend_args['context'] = mp.get_context()
        if backend is default_parallel_config['backend'] or backend is None:
            backend = active_backend
        elif isinstance(backend, ParallelBackendBase):
            if backend.nesting_level is None:
                backend.nesting_level = nesting_level
        elif hasattr(backend, 'Pool') and hasattr(backend, 'Lock'):
            self._backend_args['context'] = backend
            backend = MultiprocessingBackend(nesting_level=nesting_level)
        elif backend not in BACKENDS and backend in MAYBE_AVAILABLE_BACKENDS:
            warnings.warn(f"joblib backend '{backend}' is not available on your system, falling back to {DEFAULT_BACKEND}.", UserWarning, stacklevel=2)
            BACKENDS[backend] = BACKENDS[DEFAULT_BACKEND]
            backend = BACKENDS[DEFAULT_BACKEND](nesting_level=nesting_level)
        else:
            try:
                backend_factory = BACKENDS[backend]
            except KeyError as e:
                raise ValueError('Invalid backend: %s, expected one of %r' % (backend, sorted(BACKENDS.keys()))) from e
            backend = backend_factory(nesting_level=nesting_level)
        n_jobs = _get_config_param(n_jobs, context_config, 'n_jobs')
        if n_jobs is None:
            n_jobs = backend.default_n_jobs
        try:
            n_jobs = int(n_jobs)
        except ValueError:
            raise ValueError('n_jobs could not be converted to int')
        self.n_jobs = n_jobs
        if require == 'sharedmem' and (not getattr(backend, 'supports_sharedmem', False)):
            raise ValueError('Backend %s does not support shared memory' % backend)
        if batch_size == 'auto' or (isinstance(batch_size, Integral) and batch_size > 0):
            self.batch_size = batch_size
        else:
            raise ValueError("batch_size must be 'auto' or a positive integer, got: %r" % batch_size)
        if not isinstance(backend, SequentialBackend):
            if self.return_generator and (not backend.supports_return_generator):
                raise ValueError('Backend {} does not support return_as={}'.format(backend, return_as))
            self._lock = threading.RLock()
            self._jobs = collections.deque()
            self._pending_outputs = list()
            self._ready_batches = queue.Queue()
            self._reducer_callback = None
        self._backend = backend
        self._running = False
        self._managed_backend = False
        self._id = uuid4().hex
        self._call_ref = None

    def __enter__(self):
        self._managed_backend = True
        self._calling = False
        self._initialize_backend()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._managed_backend = False
        if self.return_generator and self._calling:
            self._abort()
        self._terminate_and_reset()
        setattr(_backend, 'config', self.old_parallel_config)

    def _initialize_backend(self):
        """Build a process or thread pool and return the number of workers"""
        pass

    def _dispatch(self, batch):
        """Queue the batch for computing, with or without multiprocessing

        WARNING: this method is not thread-safe: it should be only called
        indirectly via dispatch_one_batch.

        """
        pass

    def dispatch_next(self):
        """Dispatch more data for parallel processing

        This method is meant to be called concurrently by the multiprocessing
        callback. We rely on the thread-safety of dispatch_one_batch to protect
        against concurrent consumption of the unprotected iterator.

        """
        pass

    def dispatch_one_batch(self, iterator):
        """Prefetch the tasks for the next batch and dispatch them.

        The effective size of the batch is computed here.
        If there are no more jobs to dispatch, return False, else return True.

        The iterator consumption and dispatching is protected by the same
        lock so calling this function should be thread safe.

        """
        pass

    def _get_batch_size(self):
        """Returns the effective batch size for dispatch"""
        pass

    def _print(self, msg):
        """Display the message on stout or stderr depending on verbosity"""
        pass

    def _is_completed(self):
        """Check if all tasks have been completed"""
        pass

    def print_progress(self):
        """Display the process of the parallel execution only a fraction
           of time, controlled by self.verbose.
        """
        pass

    def _get_outputs(self, iterator, pre_dispatch):
        """Iterator returning the tasks' output as soon as they are ready."""
        pass

    def _wait_retrieval(self):
        """Return True if we need to continue retrieving some tasks."""
        pass

    def _raise_error_fast(self):
        """If we are aborting, raise if a job caused an error."""
        pass

    def _warn_exit_early(self):
        """Warn the user if the generator is gc'ed before being consumned."""
        pass

    def _get_sequential_output(self, iterable):
        """Separate loop for sequential output.

        This simplifies the traceback in case of errors and reduces the
        overhead of calling sequential tasks with `joblib`.
        """
        pass

    def _reset_run_tracking(self):
        """Reset the counters and flags used to track the execution."""
        pass

    def __call__(self, iterable):
        """Main function to dispatch parallel tasks."""
        self._reset_run_tracking()
        self._start_time = time.time()
        if not self._managed_backend:
            n_jobs = self._initialize_backend()
        else:
            n_jobs = self._effective_n_jobs()
        if n_jobs == 1:
            output = self._get_sequential_output(iterable)
            next(output)
            return output if self.return_generator else list(output)
        with self._lock:
            self._call_id = uuid4().hex
        self._cached_effective_n_jobs = n_jobs
        if isinstance(self._backend, LokyBackend):

            def _batched_calls_reducer_callback():
                self._backend._workers._temp_folder_manager.set_current_context(self._id)
            self._reducer_callback = _batched_calls_reducer_callback
        self._cached_effective_n_jobs = n_jobs
        backend_name = self._backend.__class__.__name__
        if n_jobs == 0:
            raise RuntimeError('%s has no active worker.' % backend_name)
        self._print(f'Using backend {backend_name} with {n_jobs} concurrent workers.')
        if hasattr(self._backend, 'start_call'):
            self._backend.start_call()
        self._calling = True
        iterator = iter(iterable)
        pre_dispatch = self.pre_dispatch
        if pre_dispatch == 'all':
            self._original_iterator = None
            self._pre_dispatch_amount = 0
        else:
            self._original_iterator = iterator
            if hasattr(pre_dispatch, 'endswith'):
                pre_dispatch = eval_expr(pre_dispatch.replace('n_jobs', str(n_jobs)))
            self._pre_dispatch_amount = pre_dispatch = int(pre_dispatch)
            iterator = itertools.islice(iterator, self._pre_dispatch_amount)
        self._pickle_cache = dict()
        output = self._get_outputs(iterator, pre_dispatch)
        self._call_ref = weakref.ref(output)
        next(output)
        return output if self.return_generator else list(output)

    def __repr__(self):
        return '%s(n_jobs=%s)' % (self.__class__.__name__, self.n_jobs)
