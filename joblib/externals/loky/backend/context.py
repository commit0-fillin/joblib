import os
import sys
import math
import subprocess
import traceback
import warnings
import multiprocessing as mp
from multiprocessing import get_context as mp_get_context
from multiprocessing.context import BaseContext
from .process import LokyProcess, LokyInitMainProcess
if sys.version_info >= (3, 8):
    from concurrent.futures.process import _MAX_WINDOWS_WORKERS
    if sys.version_info < (3, 10):
        _MAX_WINDOWS_WORKERS = _MAX_WINDOWS_WORKERS - 1
else:
    _MAX_WINDOWS_WORKERS = 60
START_METHODS = ['loky', 'loky_init_main', 'spawn']
if sys.platform != 'win32':
    START_METHODS += ['fork', 'forkserver']
_DEFAULT_START_METHOD = None
physical_cores_cache = None

def cpu_count(only_physical_cores=False):
    """Return the number of CPUs the current process can use.

    The returned number of CPUs accounts for:
     * the number of CPUs in the system, as given by
       ``multiprocessing.cpu_count``;
     * the CPU affinity settings of the current process
       (available on some Unix systems);
     * Cgroup CPU bandwidth limit (available on Linux only, typically
       set by docker and similar container orchestration systems);
     * the value of the LOKY_MAX_CPU_COUNT environment variable if defined.
    and is given as the minimum of these constraints.

    If ``only_physical_cores`` is True, return the number of physical cores
    instead of the number of logical cores (hyperthreading / SMT). Note that
    this option is not enforced if the number of usable cores is controlled in
    any other way such as: process affinity, Cgroup restricted CPU bandwidth
    or the LOKY_MAX_CPU_COUNT environment variable. If the number of physical
    cores is not found, return the number of logical cores.

    Note that on Windows, the returned number of CPUs cannot exceed 61 (or 60 for
    Python < 3.10), see:
    https://bugs.python.org/issue26903.

    It is also always larger or equal to 1.
    """
    import multiprocessing
    import os

    # Get the number of CPUs from the system
    try:
        system_cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        system_cpu_count = 1

    # Check for CPU affinity (Unix systems)
    try:
        import psutil
        process = psutil.Process()
        affinity_cpu_count = len(process.cpu_affinity())
    except (ImportError, AttributeError):
        affinity_cpu_count = system_cpu_count

    # Check for Cgroup CPU bandwidth limit (Linux only)
    cgroup_cpu_count = float('inf')
    if sys.platform.startswith('linux'):
        try:
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
                quota = int(f.read())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
                period = int(f.read())
            if quota > 0 and period > 0:
                cgroup_cpu_count = max(1, int(quota / period))
        except:
            pass

    # Check for LOKY_MAX_CPU_COUNT environment variable
    env_cpu_count = int(os.environ.get('LOKY_MAX_CPU_COUNT', float('inf')))

    # Get the minimum of all constraints
    cpu_count = min(system_cpu_count, affinity_cpu_count, cgroup_cpu_count, env_cpu_count)

    # Handle physical cores if requested
    if only_physical_cores:
        physical_cores, _ = _count_physical_cores()
        if physical_cores != "not found":
            cpu_count = min(cpu_count, physical_cores)

    # Ensure the count is at least 1 and doesn't exceed the Windows limit
    if sys.platform == 'win32':
        cpu_count = min(cpu_count, _MAX_WINDOWS_WORKERS)

    return max(1, cpu_count)

def _cpu_count_user(os_cpu_count):
    """Number of user defined available CPUs"""
    import os
    cpu_count_user = os.environ.get('LOKY_MAX_CPU_COUNT')
    if cpu_count_user is not None:
        try:
            cpu_count_user = int(cpu_count_user)
        except ValueError:
            print(f"LOKY_MAX_CPU_COUNT should be an int. Got '{cpu_count_user}'. "
                  "Defaulting to os.cpu_count().")
            return os_cpu_count
        if cpu_count_user < 1:
            print(f"LOKY_MAX_CPU_COUNT should be >= 1. Got {cpu_count_user}. "
                  "Defaulting to os.cpu_count().")
            return os_cpu_count
        return min(cpu_count_user, os_cpu_count)
    return os_cpu_count

def _count_physical_cores():
    """Return a tuple (number of physical cores, exception)

    If the number of physical cores is found, exception is set to None.
    If it has not been found, return ("not found", exception).

    The number of physical cores is cached to avoid repeating subprocess calls.
    """
    global physical_cores_cache

    if physical_cores_cache is not None:
        return physical_cores_cache

    import subprocess

    try:
        if sys.platform == 'linux':
            # Try to use lscpu
            output = subprocess.check_output(['lscpu', '-p=Core,Socket']).decode()
            core_socket_pairs = {tuple(line.split(',')) for line in output.splitlines()
                                 if not line.startswith('#')}
            num_physical_cores = len(core_socket_pairs)
        elif sys.platform == 'darwin':
            # Try to use sysctl on macOS
            output = subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).decode()
            num_physical_cores = int(output.strip())
        elif sys.platform == 'win32':
            # Try to use wmic on Windows
            output = subprocess.check_output(['wmic', 'cpu', 'get', 'NumberOfCores']).decode()
            num_physical_cores = int(output.split('\n')[1].strip())
        else:
            raise NotImplementedError(f"Unsupported platform: {sys.platform}")

        physical_cores_cache = (num_physical_cores, None)
    except Exception as e:
        physical_cores_cache = ("not found", e)

    return physical_cores_cache

class LokyContext(BaseContext):
    """Context relying on the LokyProcess."""
    _name = 'loky'
    Process = LokyProcess
    cpu_count = staticmethod(cpu_count)

    def Queue(self, maxsize=0, reducers=None):
        """Returns a queue object"""
        from .queues import Queue
        return Queue(maxsize, reducers=reducers, ctx=self.get_context())

    def SimpleQueue(self, reducers=None):
        """Returns a queue object"""
        from .queues import SimpleQueue
        return SimpleQueue(reducers=reducers, ctx=self.get_context())
    if sys.platform != 'win32':
        'For Unix platform, use our custom implementation of synchronize\n        ensuring that we use the loky.backend.resource_tracker to clean-up\n        the semaphores in case of a worker crash.\n        '

        def Semaphore(self, value=1):
            """Returns a semaphore object"""
            from .synchronize import Semaphore
            return Semaphore(value)

        def BoundedSemaphore(self, value):
            """Returns a bounded semaphore object"""
            from .synchronize import BoundedSemaphore
            return BoundedSemaphore(value)

        def Lock(self):
            """Returns a lock object"""
            from .synchronize import Lock
            return Lock()

        def RLock(self):
            """Returns a recurrent lock object"""
            from .synchronize import RLock
            return RLock()

        def Condition(self, lock=None):
            """Returns a condition object"""
            from .synchronize import Condition
            return Condition(lock)

        def Event(self):
            """Returns an event object"""
            from .synchronize import Event
            return Event()

class LokyInitMainContext(LokyContext):
    """Extra context with LokyProcess, which does load the main module

    This context is used for compatibility in the case ``cloudpickle`` is not
    present on the running system. This permits to load functions defined in
    the ``main`` module, using proper safeguards. The declaration of the
    ``executor`` should be protected by ``if __name__ == "__main__":`` and the
    functions and variable used from main should be out of this block.

    This mimics the default behavior of multiprocessing under Windows and the
    behavior of the ``spawn`` start method on a posix system.
    For more details, see the end of the following section of python doc
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    """
    _name = 'loky_init_main'
    Process = LokyInitMainProcess
ctx_loky = LokyContext()
mp.context._concrete_contexts['loky'] = ctx_loky
mp.context._concrete_contexts['loky_init_main'] = LokyInitMainContext()
