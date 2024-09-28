"""Helper module to factorize the conditional multiprocessing import logic

We use a distinct module to simplify import statements and avoid introducing
circular dependencies (for instance for the assert_spawning name).
"""
import os
import warnings
mp = int(os.environ.get('JOBLIB_MULTIPROCESSING', 1)) or None
if mp:
    try:
        import multiprocessing as mp
        import _multiprocessing
    except ImportError:
        mp = None
if mp is not None:
    try:
        import tempfile
        from _multiprocessing import SemLock
        _rand = tempfile._RandomNameSequence()
        for i in range(100):
            try:
                name = '/joblib-{}-{}'.format(os.getpid(), next(_rand))
                _sem = SemLock(0, 0, 1, name=name, unlink=True)
                del _sem
                break
            except FileExistsError as e:
                if i >= 99:
                    raise FileExistsError('cannot find name for semaphore') from e
    except (FileExistsError, AttributeError, ImportError, OSError) as e:
        mp = None
        warnings.warn('%s.  joblib will operate in serial mode' % (e,))
if mp is not None:
    from multiprocessing.context import assert_spawning
else:
    assert_spawning = None