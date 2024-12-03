"""
Disk management utilities.
"""
import os
import sys
import time
import errno
import shutil
from multiprocessing import util
try:
    WindowsError
except NameError:
    WindowsError = OSError

def disk_used(path):
    """ Return the disk usage in a directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def memstr_to_bytes(text):
    """ Convert a memory text to its value in bytes.
    """
    units = {
        'K': 1024,
        'M': 1024 ** 2,
        'G': 1024 ** 3,
        'T': 1024 ** 4,
    }
    text = text.upper().strip()
    if text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    else:
        return int(float(text))

def mkdirp(d):
    """Ensure directory d exists (like mkdir -p on Unix)
    No guarantee that the directory is writable.
    """
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
RM_SUBDIRS_RETRY_TIME = 0.1
RM_SUBDIRS_N_RETRY = 10

def rm_subdirs(path, onerror=None):
    """Remove all subdirectories in this path.

    The directory indicated by `path` is left in place, and its subdirectories
    are erased.

    If onerror is set, it is called to handle the error with arguments (func,
    path, exc_info) where func is os.listdir, os.remove, or os.rmdir;
    path is the argument to that function that caused it to fail; and
    exc_info is a tuple returned by sys.exc_info().  If onerror is None,
    an exception is raised.
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                os.remove(file_path)
            except Exception as e:
                if onerror is not None:
                    onerror(os.remove, file_path, sys.exc_info())
                else:
                    raise
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                os.rmdir(dir_path)
            except Exception as e:
                if onerror is not None:
                    onerror(os.rmdir, dir_path, sys.exc_info())
                else:
                    raise

def delete_folder(folder_path, onerror=None, allow_non_empty=True):
    """Utility function to cleanup a temporary folder if it still exists."""
    if os.path.exists(folder_path):
        if allow_non_empty:
            shutil.rmtree(folder_path, onerror=onerror)
        else:
            try:
                os.rmdir(folder_path)
            except OSError as e:
                if e.errno != errno.ENOTEMPTY:
                    if onerror is not None:
                        onerror(os.rmdir, folder_path, sys.exc_info())
                    else:
                        raise
