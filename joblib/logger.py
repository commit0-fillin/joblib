"""
Helpers for logging.

This module needs much love to become useful.
"""
from __future__ import print_function
import time
import sys
import os
import shutil
import logging
import pprint
from .disk import mkdirp

def _squeeze_time(t):
    """Remove .1s to the time under Windows: this is the time it take to
    stat files. This is needed to make results similar to timings under
    Unix, for tests
    """
    if sys.platform.startswith('win'):
        return max(0, t - 0.1)
    return t

class Logger(object):
    """ Base class for logging messages.
    """

    def __init__(self, depth=3, name=None):
        """
            Parameters
            ----------
            depth: int, optional
                The depth of objects printed.
            name: str, optional
                The namespace to log to. If None, defaults to joblib.
        """
        self.depth = depth
        self._name = name if name else 'joblib'

    def format(self, obj, indent=0):
        """Return the formatted representation of the object."""
        if isinstance(obj, str):
            return obj
        
        formatted = pprint.pformat(obj, indent=indent, depth=self.depth)
        if '\n' in formatted:
            return '\n' + formatted
        return formatted

class PrintTime(object):
    """ Print and log messages while keeping track of time.
    """

    def __init__(self, logfile=None, logdir=None):
        if logfile is not None and logdir is not None:
            raise ValueError('Cannot specify both logfile and logdir')
        self.last_time = time.time()
        self.start_time = self.last_time
        if logdir is not None:
            logfile = os.path.join(logdir, 'joblib.log')
        self.logfile = logfile
        if logfile is not None:
            mkdirp(os.path.dirname(logfile))
            if os.path.exists(logfile):
                for i in range(1, 9):
                    try:
                        shutil.move(logfile + '.%i' % i, logfile + '.%i' % (i + 1))
                    except:
                        'No reason failing here'
                try:
                    shutil.copy(logfile, logfile + '.1')
                except:
                    'No reason failing here'
            try:
                with open(logfile, 'w') as logfile:
                    logfile.write('\nLogging joblib python script\n')
                    logfile.write('\n---%s---\n' % time.ctime(self.last_time))
            except:
                ' Multiprocessing writing to files can create race\n                    conditions. Rather fail silently than crash the\n                    computation.\n                '

    def __call__(self, msg='', total=False):
        """ Print the time elapsed between the last call and the current
            call, with an optional message.
        """
        if not total:
            time_lapse = time.time() - self.last_time
            full_msg = '%s: %s' % (msg, format_time(time_lapse))
        else:
            time_lapse = time.time() - self.start_time
            full_msg = '%s: %.2fs, %.1f min' % (msg, time_lapse, time_lapse / 60)
        print(full_msg, file=sys.stderr)
        if self.logfile is not None:
            try:
                with open(self.logfile, 'a') as f:
                    print(full_msg, file=f)
            except:
                ' Multiprocessing writing to files can create race\n                    conditions. Rather fail silently than crash the\n                    calculation.\n                '
        self.last_time = time.time()
