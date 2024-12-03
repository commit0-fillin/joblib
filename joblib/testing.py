"""
Helper for testing.
"""
import sys
import warnings
import os.path
import re
import subprocess
import threading
import pytest
import _pytest
raises = pytest.raises
warns = pytest.warns
SkipTest = _pytest.runner.Skipped
skipif = pytest.mark.skipif
fixture = pytest.fixture
parametrize = pytest.mark.parametrize
timeout = pytest.mark.timeout
xfail = pytest.mark.xfail
param = pytest.param

def warnings_to_stdout():
    """ Redirect all warnings to stdout.
    """
    warnings.filterwarnings("always")
    warnings.simplefilter("always")
    for warning in warnings.filters:
        warnings.filterwarnings("always", category=warning[2])
    
    old_showwarning = warnings.showwarning
    def showwarning(message, category, filename, lineno, file=None, line=None):
        sys.stdout.write(warnings.formatwarning(message, category, filename, lineno, line))
    warnings.showwarning = showwarning

def check_subprocess_call(cmd, timeout=5, stdout_regex=None, stderr_regex=None):
    """Runs a command in a subprocess with timeout in seconds.

    A SIGTERM is sent after `timeout` and if it does not terminate, a
    SIGKILL is sent after `2 * timeout`.

    Also checks returncode is zero, stdout if stdout_regex is set, and
    stderr if stderr_regex is set.
    """
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()

    returncode = process.returncode

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd, stdout, stderr)

    if stdout_regex and not re.search(stdout_regex, stdout):
        raise AssertionError(f"Stdout did not match regex: {stdout_regex}\nStdout: {stdout}")

    if stderr_regex and not re.search(stderr_regex, stderr):
        raise AssertionError(f"Stderr did not match regex: {stderr_regex}\nStderr: {stderr}")

    return stdout, stderr
