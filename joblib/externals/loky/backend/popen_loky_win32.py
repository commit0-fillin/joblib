import os
import sys
import msvcrt
import _winapi
from pickle import load
from multiprocessing import process, util
from multiprocessing.context import set_spawning_popen
from multiprocessing.popen_spawn_win32 import Popen as _Popen
from . import reduction, spawn
__all__ = ['Popen']
WINENV = hasattr(sys, '_base_executable') and (not _path_eq(sys.executable, sys._base_executable))

class Popen(_Popen):
    """
    Start a subprocess to run the code of a process object.

    We differ from cpython implementation with the way we handle environment
    variables, in order to be able to modify then in the child processes before
    importing any library, in order to control the number of threads in C-level
    threadpools.

    We also use the loky preparation data, in particular to handle main_module
    inits and the loky resource tracker.
    """
    method = 'loky'

    def __init__(self, process_obj):
        prep_data = spawn.get_preparation_data(process_obj._name, getattr(process_obj, 'init_main_module', True))
        rhandle, whandle = _winapi.CreatePipe(None, 0)
        wfd = msvcrt.open_osfhandle(whandle, 0)
        cmd = get_command_line(parent_pid=os.getpid(), pipe_handle=rhandle)
        python_exe = spawn.get_executable()
        child_env = {**os.environ, **process_obj.env}
        if WINENV and _path_eq(python_exe, sys.executable):
            cmd[0] = python_exe = sys._base_executable
            child_env['__PYVENV_LAUNCHER__'] = sys.executable
        cmd = ' '.join((f'"{x}"' for x in cmd))
        with open(wfd, 'wb') as to_child:
            try:
                hp, ht, pid, _ = _winapi.CreateProcess(python_exe, cmd, None, None, False, 0, child_env, None, None)
                _winapi.CloseHandle(ht)
            except BaseException:
                _winapi.CloseHandle(rhandle)
                raise
            self.pid = pid
            self.returncode = None
            self._handle = hp
            self.sentinel = int(hp)
            self.finalizer = util.Finalize(self, _close_handles, (self.sentinel, int(rhandle)))
            set_spawning_popen(self)
            try:
                reduction.dump(prep_data, to_child)
                reduction.dump(process_obj, to_child)
            finally:
                set_spawning_popen(None)

def get_command_line(pipe_handle, parent_pid, **kwds):
    """Returns prefix of command line used for spawning a child process."""
    from . import spawn
    from .context import get_context
    
    prog = f'from {__name__} import main; main()'
    opts = {'pipe_handle': pipe_handle, 'parent_pid': parent_pid, **kwds}
    
    return [
        sys.executable,
        '-c',
        prog,
        '--pipe=' + str(pipe_handle),
        '--parent=' + str(parent_pid)
    ] + [f'--{k}={v}' for k, v in opts.items() if k not in ('pipe_handle', 'parent_pid')]

def is_forking(argv):
    """Return whether commandline indicates we are forking."""
    return len(argv) >= 2 and argv[1] == '--multiprocessing-fork'

def main(pipe_handle, parent_pid=None):
    """Run code specified by data received over pipe."""
    from . import spawn
    from .reduction import loads
    
    # Retrieve the preparation data
    with os.fdopen(msvcrt.open_osfhandle(pipe_handle, os.O_RDONLY), 'rb') as from_parent:
        preparation_data = loads(from_parent.read())
    
    # Prepare the process
    spawn.prepare(preparation_data, parent_sentinel=pipe_handle)
    
    # Get the process object
    from_parent = os.fdopen(msvcrt.open_osfhandle(pipe_handle, os.O_RDONLY), 'rb')
    process_obj = loads(from_parent.read())
    from_parent.close()

    # Start the process
    self = process.current_process()
    self._inheriting = True
    try:
        process_obj._bootstrap()
    finally:
        self._inheriting = False
