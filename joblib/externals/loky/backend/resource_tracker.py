import os
import shutil
import sys
import signal
import warnings
import threading
from _multiprocessing import sem_unlink
from multiprocessing import util
from . import spawn
if sys.platform == 'win32':
    import _winapi
    import msvcrt
    from multiprocessing.reduction import duplicate
__all__ = ['ensure_running', 'register', 'unregister']
_HAVE_SIGMASK = hasattr(signal, 'pthread_sigmask')
_IGNORED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_CLEANUP_FUNCS = {'folder': shutil.rmtree, 'file': os.unlink}
if os.name == 'posix':
    _CLEANUP_FUNCS['semlock'] = sem_unlink
VERBOSE = False

class ResourceTracker:

    def __init__(self):
        self._lock = threading.Lock()
        self._fd = None
        self._pid = None

    def ensure_running(self):
        """Make sure that resource tracker process is running.

        This can be run from any process.  Usually a child process will use
        the resource created by its parent."""
        with self._lock:
            if self._fd is not None:
                # Resource tracker already running
                return
            
            if self._pid is not None:
                # Check if the existing process is still alive
                if self._check_alive():
                    return
                else:
                    # Clean up the dead process
                    os.close(self._fd)
                    self._fd = None
                    self._pid = None

            # Start a new resource tracker process
            fds_to_pass = []
            cmd = [sys.executable, '-c', 'from joblib.externals.loky.backend.resource_tracker import main; main()']
            r, w = os.pipe()
            try:
                fds_to_pass.append(r)
                # Start the resource tracker process
                pid = spawn.spawnv_passfds(sys.executable, cmd, fds_to_pass)
            except:
                os.close(r)
                os.close(w)
                raise
            else:
                self._fd = w
                self._pid = pid
                if VERBOSE:
                    print('Started resource tracker process with PID', pid)

    def _check_alive(self):
        """Check for the existence of the resource tracker process."""
        if self._pid is None:
            return False
        try:
            os.kill(self._pid, 0)
        except OSError:
            # The process is no longer alive
            return False
        return True

    def register(self, name, rtype):
        """Register a named resource, and increment its refcount."""
        self.ensure_running()
        msg = f'REGISTER {name} {rtype}\n'.encode('ascii')
        if len(name) > 512:
            # Prevent overflow on the C side
            raise ValueError("name too long")
        with self._lock:
            if self._fd is None:
                raise ValueError("Resource tracker not running")
            try:
                os.write(self._fd, msg)
            except OSError:
                # The resource tracker process might have died
                self._fd = None
                self._pid = None
                raise

    def unregister(self, name, rtype):
        """Unregister a named resource with resource tracker."""
        self.ensure_running()
        msg = f'UNREGISTER {name} {rtype}\n'.encode('ascii')
        if len(name) > 512:
            # Prevent overflow on the C side
            raise ValueError("name too long")
        with self._lock:
            if self._fd is None:
                return  # Resource tracker not running
            try:
                os.write(self._fd, msg)
            except OSError:
                # The resource tracker process might have died
                self._fd = None
                self._pid = None

    def maybe_unlink(self, name, rtype):
        """Decrement the refcount of a resource, and delete it if it hits 0"""
        self.ensure_running()
        msg = f'MAYBE_UNLINK {name} {rtype}\n'.encode('ascii')
        if len(name) > 512:
            # Prevent overflow on the C side
            raise ValueError("name too long")
        with self._lock:
            if self._fd is None:
                return  # Resource tracker not running
            try:
                os.write(self._fd, msg)
            except OSError:
                # The resource tracker process might have died
                self._fd = None
                self._pid = None
_resource_tracker = ResourceTracker()
ensure_running = _resource_tracker.ensure_running
register = _resource_tracker.register
maybe_unlink = _resource_tracker.maybe_unlink
unregister = _resource_tracker.unregister
getfd = _resource_tracker.getfd

def main(fd=None, verbose=0):
    """Run resource tracker."""
    global VERBOSE
    VERBOSE = verbose

    if fd is None:
        fd = sys.stdin.fileno()

    if _HAVE_SIGMASK:
        signal.pthread_sigmask(signal.SIG_BLOCK, _IGNORED_SIGNALS)

    cache = {}
    try:
        # Ignore SIGINT and SIGTERM
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        for line in iter(lambda: os.read(fd, 1024).decode('ascii'), ''):
            cmd, name, rtype = line.strip().split()
            if cmd == 'REGISTER':
                cache.setdefault(rtype, {}).setdefault(name, 0)
                cache[rtype][name] += 1
            elif cmd == 'UNREGISTER':
                if name in cache.get(rtype, {}):
                    cache[rtype][name] -= 1
                    if cache[rtype][name] == 0:
                        del cache[rtype][name]
                        _CLEANUP_FUNCS[rtype](name)
            elif cmd == 'MAYBE_UNLINK':
                if name in cache.get(rtype, {}):
                    cache[rtype][name] -= 1
                    if cache[rtype][name] == 0:
                        del cache[rtype][name]
                        try:
                            _CLEANUP_FUNCS[rtype](name)
                        except OSError as e:
                            warnings.warn(f'Error cleaning up {name}: {e}')
            else:
                warnings.warn(f'Unrecognized command {cmd}')
    finally:
        if _HAVE_SIGMASK:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, _IGNORED_SIGNALS)
