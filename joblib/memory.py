"""
A context object for caching a function's return value each time it
is called with the same input arguments.

"""
import asyncio
import datetime
import functools
import inspect
import logging
import os
import pathlib
import pydoc
import re
import textwrap
import time
import tokenize
import traceback
import warnings
import weakref
from . import hashing
from ._store_backends import CacheWarning
from ._store_backends import FileSystemStoreBackend, StoreBackendBase
from .func_inspect import filter_args, format_call, format_signature, get_func_code, get_func_name
from .logger import Logger, format_time, pformat
FIRST_LINE_TEXT = '# first line:'

def extract_first_line(func_code):
    """ Extract the first line information from the function code
        text if available.
    """
    if func_code is None:
        return None
    
    lines = func_code.split('\n')
    for line in lines:
        if line.strip().startswith(FIRST_LINE_TEXT):
            return line.strip()[len(FIRST_LINE_TEXT):].strip()
    return None

class JobLibCollisionWarning(UserWarning):
    """ Warn that there might be a collision between names of functions.
    """
_STORE_BACKENDS = {'local': FileSystemStoreBackend}

def register_store_backend(backend_name, backend):
    """Extend available store backends.

    The Memory, MemorizeResult and MemorizeFunc objects are designed to be
    agnostic to the type of store used behind. By default, the local file
    system is used but this function gives the possibility to extend joblib's
    memory pattern with other types of storage such as cloud storage (S3, GCS,
    OpenStack, HadoopFS, etc) or blob DBs.

    Parameters
    ----------
    backend_name: str
        The name identifying the store backend being registered. For example,
        'local' is used with FileSystemStoreBackend.
    backend: StoreBackendBase subclass
        The name of a class that implements the StoreBackendBase interface.

    """
    if not issubclass(backend, StoreBackendBase):
        raise ValueError("The backend class should be a subclass of StoreBackendBase.")
    
    if backend_name in _STORE_BACKENDS:
        warnings.warn(f"Overriding existing backend '{backend_name}'.")
    
    _STORE_BACKENDS[backend_name] = backend

def _store_backend_factory(backend, location, verbose=0, backend_options=None):
    """Return the correct store object for the given location."""
    if backend_options is None:
        backend_options = {}

    if isinstance(backend, str):
        if backend not in _STORE_BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. "
                             f"Available backends are: {list(_STORE_BACKENDS.keys())}")
        backend_class = _STORE_BACKENDS[backend]
    elif isinstance(backend, type) and issubclass(backend, StoreBackendBase):
        backend_class = backend
    else:
        raise ValueError("Backend should be either a string or a StoreBackendBase subclass.")

    backend_instance = backend_class()
    backend_instance.configure(location, verbose=verbose, backend_options=backend_options)
    return backend_instance

def _build_func_identifier(func):
    """Build a roughly unique identifier for the cached function."""
    func_id = []
    try:
        func_code = get_func_code(func)
        if func_code is not None:
            func_id.append(func_code)
    except Exception:
        pass

    if hasattr(func, '__name__'):
        func_id.append(func.__name__)

    if hasattr(func, '__module__'):
        func_id.append(func.__module__)

    # Include the function's class if it's a method
    if hasattr(func, '__self__'):
        try:
            func_id.append(func.__self__.__class__.__name__)
        except AttributeError:
            pass

    return ':'.join(func_id)
_FUNCTION_HASHES = weakref.WeakKeyDictionary()

class MemorizedResult(Logger):
    """Object representing a cached value.

    Attributes
    ----------
    location: str
        The location of joblib cache. Depends on the store backend used.

    func: function or str
        function whose output is cached. The string case is intended only for
        instantiation based on the output of repr() on another instance.
        (namely eval(repr(memorized_instance)) works).

    argument_hash: str
        hash of the function arguments.

    backend: str
        Type of store backend for reading/writing cache files.
        Default is 'local'.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
        The memmapping mode used when loading from cache numpy arrays. See
        numpy.load for the meaning of the different values.

    verbose: int
        verbosity level (0 means no message).

    timestamp, metadata: string
        for internal use only.
    """

    def __init__(self, location, call_id, backend='local', mmap_mode=None, verbose=0, timestamp=None, metadata=None):
        Logger.__init__(self)
        self._call_id = call_id
        self.store_backend = _store_backend_factory(backend, location, verbose=verbose)
        self.mmap_mode = mmap_mode
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = self.store_backend.get_metadata(self._call_id)
        self.duration = self.metadata.get('duration', None)
        self.verbose = verbose
        self.timestamp = timestamp

    def get(self):
        """Read value from cache and return it."""
        pass

    def clear(self):
        """Clear value from cache"""
        pass

    def __repr__(self):
        return '{}(location="{}", func="{}", args_id="{}")'.format(self.__class__.__name__, self.store_backend.location, *self._call_id)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['timestamp'] = None
        return state

class NotMemorizedResult(object):
    """Class representing an arbitrary value.

    This class is a replacement for MemorizedResult when there is no cache.
    """
    __slots__ = ('value', 'valid')

    def __init__(self, value):
        self.value = value
        self.valid = True

    def __repr__(self):
        if self.valid:
            return '{class_name}({value})'.format(class_name=self.__class__.__name__, value=pformat(self.value))
        else:
            return self.__class__.__name__ + ' with no value'

    def __getstate__(self):
        return {'valid': self.valid, 'value': self.value}

    def __setstate__(self, state):
        self.valid = state['valid']
        self.value = state['value']

class NotMemorizedFunc(object):
    """No-op object decorating a function.

    This class replaces MemorizedFunc when there is no cache. It provides an
    identical API but does not write anything on disk.

    Attributes
    ----------
    func: callable
        Original undecorated function.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return '{0}(func={1})'.format(self.__class__.__name__, self.func)

class AsyncNotMemorizedFunc(NotMemorizedFunc):
    pass

class MemorizedFunc(Logger):
    """Callable object decorating a function for caching its return value
    each time it is called.

    Methods are provided to inspect the cache or clean it.

    Attributes
    ----------
    func: callable
        The original, undecorated, function.

    location: string
        The location of joblib cache. Depends on the store backend used.

    backend: str
        Type of store backend for reading/writing cache files.
        Default is 'local', in which case the location is the path to a
        disk storage.

    ignore: list or None
        List of variable names to ignore when choosing whether to
        recompute.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
        The memmapping mode used when loading from cache
        numpy arrays. See numpy.load for the meaning of the different
        values.

    compress: boolean, or integer
        Whether to zip the stored data on disk. If an integer is
        given, it should be between 1 and 9, and sets the amount
        of compression. Note that compressed arrays cannot be
        read by memmapping.

    verbose: int, optional
        The verbosity flag, controls messages that are issued as
        the function is evaluated.

    cache_validation_callback: callable, optional
        Callable to check if a result in cache is valid or is to be recomputed.
        When the function is called with arguments for which a cache exists,
        the callback is called with the cache entry's metadata as its sole
        argument. If it returns True, the cached result is returned, else the
        cache for these arguments is cleared and the result is recomputed.
    """

    def __init__(self, func, location, backend='local', ignore=None, mmap_mode=None, compress=False, verbose=1, timestamp=None, cache_validation_callback=None):
        Logger.__init__(self)
        self.mmap_mode = mmap_mode
        self.compress = compress
        self.func = func
        self.cache_validation_callback = cache_validation_callback
        self.func_id = _build_func_identifier(func)
        self.ignore = ignore if ignore is not None else []
        self._verbose = verbose
        self.store_backend = _store_backend_factory(backend, location, verbose=verbose, backend_options=dict(compress=compress, mmap_mode=mmap_mode))
        if self.store_backend is not None:
            self.store_backend.store_cached_func_code([self.func_id])
        self.timestamp = timestamp if timestamp is not None else time.time()
        try:
            functools.update_wrapper(self, func)
        except Exception:
            pass
        if inspect.isfunction(func):
            doc = pydoc.TextDoc().document(func)
            doc = doc.replace('\n', '\n\n', 1)
            doc = re.sub('\x08.', '', doc)
        else:
            doc = func.__doc__
        self.__doc__ = 'Memoized version of %s' % doc
        self._func_code_info = None
        self._func_code_id = None

    def _is_in_cache_and_valid(self, call_id):
        """Check if the function call is cached and valid for given arguments.

        - Compare the function code with the one from the cached function,
        asserting if it has changed.
        - Check if the function call is present in the cache.
        - Call `cache_validation_callback` for user define cache validation.

        Returns True if the function call is in cache and can be used, and
        returns False otherwise.
        """
        pass

    def _cached_call(self, args, kwargs, shelving):
        """Call wrapped function and cache result, or read cache if available.

        This function returns the wrapped function output or a reference to
        the cached result.

        Arguments:
        ----------

        args, kwargs: list and dict
            input arguments for wrapped function

        shelving: bool
            True when called via the call_and_shelve function.


        Returns
        -------
        output: Output of the wrapped function if shelving is false, or a
            MemorizedResult reference to the value if shelving is true.
        metadata: dict containing the metadata associated with the call.
        """
        pass

    def call_and_shelve(self, *args, **kwargs):
        """Call wrapped function, cache result and return a reference.

        This method returns a reference to the cached result instead of the
        result itself. The reference object is small and pickeable, allowing
        to send or store it easily. Call .get() on reference object to get
        result.

        Returns
        -------
        cached_result: MemorizedResult or NotMemorizedResult
            reference to the value returned by the wrapped function. The
            class "NotMemorizedResult" is used when there is no cache
            activated (e.g. location=None in Memory).
        """
        pass

    def __call__(self, *args, **kwargs):
        return self._cached_call(args, kwargs, shelving=False)[0]

    def __getstate__(self):
        _ = self.func_code_info
        state = self.__dict__.copy()
        state['timestamp'] = None
        state['_func_code_id'] = None
        return state

    def check_call_in_cache(self, *args, **kwargs):
        """Check if function call is in the memory cache.

        Does not call the function or do any work besides func inspection
        and arg hashing.

        Returns
        -------
        is_call_in_cache: bool
            Whether or not the result of the function has been cached
            for the input arguments that have been passed.
        """
        pass

    def _get_args_id(self, *args, **kwargs):
        """Return the input parameter hash of a result."""
        pass

    def _hash_func(self):
        """Hash a function to key the online cache"""
        pass

    def _write_func_code(self, func_code, first_line):
        """ Write the function code and the filename to a file.
        """
        pass

    def _check_previous_func_code(self, stacklevel=2):
        """
            stacklevel is the depth a which this function is called, to
            issue useful warnings to the user.
        """
        pass

    def clear(self, warn=True):
        """Empty the function's cache."""
        pass

    def call(self, *args, **kwargs):
        """Force the execution of the function with the given arguments.

        The output values will be persisted, i.e., the cache will be updated
        with any new values.

        Parameters
        ----------
        *args: arguments
            The arguments.
        **kwargs: keyword arguments
            Keyword arguments.

        Returns
        -------
        output : object
            The output of the function call.
        metadata : dict
            The metadata associated with the call.
        """
        pass

    def _persist_input(self, duration, call_id, args, kwargs, this_duration_limit=0.5):
        """ Save a small summary of the call using json format in the
            output directory.

            output_dir: string
                directory where to write metadata.

            duration: float
                time taken by hashing input arguments, calling the wrapped
                function and persisting its output.

            args, kwargs: list and dict
                input arguments for wrapped function

            this_duration_limit: float
                Max execution time for this function before issuing a warning.
        """
        pass

    def __repr__(self):
        return '{class_name}(func={func}, location={location})'.format(class_name=self.__class__.__name__, func=self.func, location=self.store_backend.location)

class AsyncMemorizedFunc(MemorizedFunc):

    async def __call__(self, *args, **kwargs):
        out = self._cached_call(args, kwargs, shelving=False)
        out = await out if asyncio.iscoroutine(out) else out
        return out[0]

class Memory(Logger):
    """ A context object for caching a function's return value each time it
        is called with the same input arguments.

        All values are cached on the filesystem, in a deep directory
        structure.

        Read more in the :ref:`User Guide <memory>`.

        Parameters
        ----------
        location: str, pathlib.Path or None
            The path of the base directory to use as a data store
            or None. If None is given, no caching is done and
            the Memory object is completely transparent. This option
            replaces cachedir since version 0.12.

        backend: str, optional
            Type of store backend for reading/writing cache files.
            Default: 'local'.
            The 'local' backend is using regular filesystem operations to
            manipulate data (open, mv, etc) in the backend.

        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
            The memmapping mode used when loading from cache
            numpy arrays. See numpy.load for the meaning of the
            arguments.

        compress: boolean, or integer, optional
            Whether to zip the stored data on disk. If an integer is
            given, it should be between 1 and 9, and sets the amount
            of compression. Note that compressed arrays cannot be
            read by memmapping.

        verbose: int, optional
            Verbosity flag, controls the debug messages that are issued
            as functions are evaluated.

        bytes_limit: int | str, optional
            Limit in bytes of the size of the cache. By default, the size of
            the cache is unlimited. When reducing the size of the cache,
            ``joblib`` keeps the most recently accessed items first. If a
            str is passed, it is converted to a number of bytes using units
            { K | M | G} for kilo, mega, giga.

            **Note:** You need to call :meth:`joblib.Memory.reduce_size` to
            actually reduce the cache size to be less than ``bytes_limit``.

            **Note:** This argument has been deprecated. One should give the
            value of ``bytes_limit`` directly in
            :meth:`joblib.Memory.reduce_size`.

        backend_options: dict, optional
            Contains a dictionary of named parameters used to configure
            the store backend.
    """

    def __init__(self, location=None, backend='local', mmap_mode=None, compress=False, verbose=1, bytes_limit=None, backend_options=None):
        Logger.__init__(self)
        self._verbose = verbose
        self.mmap_mode = mmap_mode
        self.timestamp = time.time()
        if bytes_limit is not None:
            warnings.warn('bytes_limit argument has been deprecated. It will be removed in version 1.5. Please pass its value directly to Memory.reduce_size.', category=DeprecationWarning)
        self.bytes_limit = bytes_limit
        self.backend = backend
        self.compress = compress
        if backend_options is None:
            backend_options = {}
        self.backend_options = backend_options
        if compress and mmap_mode is not None:
            warnings.warn('Compressed results cannot be memmapped', stacklevel=2)
        self.location = location
        if isinstance(location, str):
            location = os.path.join(location, 'joblib')
        self.store_backend = _store_backend_factory(backend, location, verbose=self._verbose, backend_options=dict(compress=compress, mmap_mode=mmap_mode, **backend_options))

    def cache(self, func=None, ignore=None, verbose=None, mmap_mode=False, cache_validation_callback=None):
        """ Decorates the given function func to only compute its return
            value for input arguments not cached on disk.

            Parameters
            ----------
            func: callable, optional
                The function to be decorated
            ignore: list of strings
                A list of arguments name to ignore in the hashing
            verbose: integer, optional
                The verbosity mode of the function. By default that
                of the memory object is used.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. By default that of the memory object is used.
            cache_validation_callback: callable, optional
                Callable to validate whether or not the cache is valid. When
                the cached function is called with arguments for which a cache
                exists, this callable is called with the metadata of the cached
                result as its sole argument. If it returns True, then the
                cached result is returned, else the cache for these arguments
                is cleared and recomputed.

            Returns
            -------
            decorated_func: MemorizedFunc object
                The returned object is a MemorizedFunc object, that is
                callable (behaves like a function), but offers extra
                methods for cache lookup and management. See the
                documentation for :class:`joblib.memory.MemorizedFunc`.
        """
        pass

    def clear(self, warn=True):
        """ Erase the complete cache directory.
        """
        pass

    def reduce_size(self, bytes_limit=None, items_limit=None, age_limit=None):
        """Remove cache elements to make the cache fit its limits.

        The limitation can impose that the cache size fits in ``bytes_limit``,
        that the number of cache items is no more than ``items_limit``, and
        that all files in cache are not older than ``age_limit``.

        Parameters
        ----------
        bytes_limit: int | str, optional
            Limit in bytes of the size of the cache. By default, the size of
            the cache is unlimited. When reducing the size of the cache,
            ``joblib`` keeps the most recently accessed items first. If a
            str is passed, it is converted to a number of bytes using units
            { K | M | G} for kilo, mega, giga.

        items_limit: int, optional
            Number of items to limit the cache to.  By default, the number of
            items in the cache is unlimited.  When reducing the size of the
            cache, ``joblib`` keeps the most recently accessed items first.

        age_limit: datetime.timedelta, optional
            Maximum age of items to limit the cache to.  When reducing the size
            of the cache, any items last accessed more than the given length of
            time ago are deleted.
        """
        pass

    def eval(self, func, *args, **kwargs):
        """ Eval function func with arguments `*args` and `**kwargs`,
            in the context of the memory.

            This method works similarly to the builtin `apply`, except
            that the function is called only if the cache is not
            up to date.

        """
        pass

    def __repr__(self):
        return '{class_name}(location={location})'.format(class_name=self.__class__.__name__, location=None if self.store_backend is None else self.store_backend.location)

    def __getstate__(self):
        """ We don't store the timestamp when pickling, to avoid the hash
            depending from it.
        """
        state = self.__dict__.copy()
        state['timestamp'] = None
        return state

def expires_after(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0):
    """Helper cache_validation_callback to force recompute after a duration.

    Parameters
    ----------
    days, seconds, microseconds, milliseconds, minutes, hours, weeks: numbers
        argument passed to a timedelta.
    """
    pass
