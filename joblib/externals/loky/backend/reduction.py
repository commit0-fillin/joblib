import copyreg
import io
import functools
import types
import sys
import os
from multiprocessing import util
from pickle import loads, HIGHEST_PROTOCOL
_dispatch_table = {}

class _C:
    pass
register(type(_C().f), _reduce_method)
register(type(_C.h), _reduce_method)
if not hasattr(sys, 'pypy_version_info'):
    register(type(list.append), _reduce_method_descriptor)
    register(type(int.__add__), _reduce_method_descriptor)
register(functools.partial, _reduce_partial)
if sys.platform != 'win32':
    from ._posix_reduction import _mk_inheritable
else:
    from . import _win_reduction
try:
    from joblib.externals import cloudpickle
    DEFAULT_ENV = 'cloudpickle'
except ImportError:
    DEFAULT_ENV = 'pickle'
ENV_LOKY_PICKLER = os.environ.get('LOKY_PICKLER', DEFAULT_ENV)
_LokyPickler = None
_loky_pickler_name = None
set_loky_pickler()

def dump(obj, file, reducers=None, protocol=None):
    """Replacement for pickle.dump() using _LokyPickler."""
    if protocol is None:
        protocol = HIGHEST_PROTOCOL
    
    if reducers is None:
        reducers = _dispatch_table

    pickler = _LokyPickler(file, protocol=protocol)
    
    # Register custom reducers
    for type_, reducer in reducers.items():
        pickler.register(type_, reducer)
    
    pickler.dump(obj)
__all__ = ['dump', 'dumps', 'loads', 'register', 'set_loky_pickler']
if sys.platform == 'win32':
    from multiprocessing.reduction import duplicate
    __all__ += ['duplicate']
