import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback
try:
    import psutil
except ImportError:
    psutil = None

def kill_process_tree(process, use_psutil=True):
    """Terminate process and its descendants with SIGKILL"""
    if use_psutil and psutil is not None:
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            # Process already terminated
            pass
    else:
        _kill_process_tree_without_psutil(process)

def _kill_process_tree_without_psutil(process):
    """Terminate a process and its descendants."""
    if sys.platform != 'win32':
        _posix_recursive_kill(process.pid)
    else:
        # On Windows, we need to use the Win32 API
        import ctypes
        PROCESS_TERMINATE = 1
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, process.pid)
        ctypes.windll.kernel32.TerminateProcess(handle, -1)
        ctypes.windll.kernel32.CloseHandle(handle)

def _posix_recursive_kill(pid):
    """Recursively kill the descendants of a process before killing it."""
    try:
        # Get the list of children processes
        children = subprocess.check_output(['pgrep', '-P', str(pid)]).decode().split()
        for child_pid in children:
            _posix_recursive_kill(int(child_pid))
        os.kill(pid, signal.SIGKILL)
    except subprocess.CalledProcessError:
        # No children found
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as e:
            if e.errno != errno.ESRCH:  # ESRCH: No such process
                raise

def get_exitcodes_terminated_worker(processes):
    """Return a formatted string with the exitcodes of terminated workers.

    If necessary, wait (up to .25s) for the system to correctly set the
    exitcode of one terminated worker.
    """
    exitcodes = []
    for process in processes:
        if process.exitcode is None:
            # Wait for up to 0.25s for the system to set the exitcode
            for _ in range(25):
                time.sleep(0.01)
                if process.exitcode is not None:
                    break
        if process.exitcode is not None:
            exitcodes.append(process.exitcode)
    
    if exitcodes:
        return _format_exitcodes(exitcodes)
    return ""

def _format_exitcodes(exitcodes):
    """Format a list of exit code with names of the signals if possible"""
    def _format_exitcode(exitcode):
        if exitcode < 0:
            try:
                return f"{exitcode} ({signal.Signals(-exitcode).name})"
            except ValueError:
                return str(exitcode)
        return str(exitcode)
    
    formatted_exitcodes = [_format_exitcode(code) for code in exitcodes]
    if len(formatted_exitcodes) == 1:
        return formatted_exitcodes[0]
    return f"[{', '.join(formatted_exitcodes)}]"
