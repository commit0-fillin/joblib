"""
Reducer using memory mapping for numpy arrays
"""
from mmap import mmap
import errno
import os
import stat
import threading
import atexit
import tempfile
import time
import warnings
import weakref
from uuid import uuid4
from multiprocessing import util
from pickle import whichmodule, loads, dumps, HIGHEST_PROTOCOL, PicklingError
try:
    WindowsError
except NameError:
    WindowsError = type(None)
try:
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
except ImportError:
    np = None
from .numpy_pickle import dump, load, load_temporary_memmap
from .backports import make_memmap
from .disk import delete_folder
from .externals.loky.backend import resource_tracker
SYSTEM_SHARED_MEM_FS = '/dev/shm'
SYSTEM_SHARED_MEM_FS_MIN_SIZE = int(2000000000.0)
FOLDER_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
FILE_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR
JOBLIB_MMAPS = set()

def unlink_file(filename):
    """Wrapper around os.unlink with a retry mechanism.

    The retry mechanism has been implemented primarily to overcome a race
    condition happening during the finalizer of a np.memmap: when a process
    holding the last reference to a mmap-backed np.memmap/np.array is about to
    delete this array (and close the reference), it sends a maybe_unlink
    request to the resource_tracker. This request can be processed faster than
    it takes for the last reference of the memmap to be closed, yielding (on
    Windows) a PermissionError in the resource_tracker loop.
    """
    retry = 5
    while retry > 0:
        try:
            os.unlink(filename)
            return
        except (OSError, WindowsError) as e:
            if e.errno != errno.ENOENT:
                time.sleep(0.1)
                retry -= 1
            else:
                return
    warnings.warn(f"Failed to unlink {filename}")
resource_tracker._CLEANUP_FUNCS['file'] = unlink_file

class _WeakArrayKeyMap:
    """A variant of weakref.WeakKeyDictionary for unhashable numpy arrays.

    This datastructure will be used with numpy arrays as obj keys, therefore we
    do not use the __get__ / __set__ methods to avoid any conflict with the
    numpy fancy indexing syntax.
    """

    def __init__(self):
        self._data = {}

    def __getstate__(self):
        raise PicklingError('_WeakArrayKeyMap is not pickleable')

def _get_backing_memmap(a):
    """Recursively look up the original np.memmap instance base if any."""
    if isinstance(a, np.memmap):
        return a
    elif hasattr(a, '__array_interface__'):
        base = a.__array_interface__.get('data')
        if base is not None and len(base) == 2:
            base = base[0]
        if isinstance(base, int):
            if hasattr(a, 'base'):
                return _get_backing_memmap(a.base)
    return None

def _get_temp_dir(pool_folder_name, temp_folder=None):
    """Get the full path to a subfolder inside the temporary folder.

    Parameters
    ----------
    pool_folder_name : str
        Sub-folder name used for the serialization of a pool instance.

    temp_folder: str, optional
        Folder to be used by the pool for memmapping large arrays
        for sharing memory with worker processes. If None, this will try in
        order:

        - a folder pointed by the JOBLIB_TEMP_FOLDER environment
          variable,
        - /dev/shm if the folder exists and is writable: this is a
          RAMdisk filesystem available by default on modern Linux
          distributions,
        - the default system temporary folder that can be
          overridden with TMP, TMPDIR or TEMP environment
          variables, typically /tmp under Unix operating systems.

    Returns
    -------
    pool_folder : str
       full path to the temporary folder
    use_shared_mem : bool
       whether the temporary folder is written to the system shared memory
       folder or some other temporary folder.
    """
    use_shared_mem = False
    if temp_folder is None:
        temp_folder = os.environ.get('JOBLIB_TEMP_FOLDER', None)
    if temp_folder is None:
        if os.path.exists(SYSTEM_SHARED_MEM_FS):
            try:
                shm_stats = os.statvfs(SYSTEM_SHARED_MEM_FS)
                available_nbytes = shm_stats.f_bsize * shm_stats.f_bavail
                if available_nbytes > SYSTEM_SHARED_MEM_FS_MIN_SIZE:
                    temp_folder = SYSTEM_SHARED_MEM_FS
                    use_shared_mem = True
            except OSError:
                pass
    if temp_folder is None:
        temp_folder = tempfile.gettempdir()
    pool_folder = os.path.join(temp_folder, 'joblib', pool_folder_name)
    return pool_folder, use_shared_mem

def has_shareable_memory(a):
    """Return True if a is backed by some mmap buffer directly or not."""
    if isinstance(a, np.memmap):
        return True
    if hasattr(a, '__array_interface__'):
        d = a.__array_interface__
        if isinstance(d['data'][0], int):
            return True
    if hasattr(a, 'base'):
        return has_shareable_memory(a.base)
    return False

def _strided_from_memmap(filename, dtype, mode, offset, order, shape, strides, total_buffer_len, unlink_on_gc_collect):
    """Reconstruct an array view on a memory mapped file."""
    if unlink_on_gc_collect:
        resource_tracker.register(filename, 'file')
    m = make_memmap(filename, dtype=dtype, shape=(total_buffer_len,),
                    mode=mode, offset=offset, order=order)
    if m.shape == shape and m.strides == strides:
        return m
    else:
        return as_strided(m, shape=shape, strides=strides)

def _reduce_memmap_backed(a, m):
    """Pickling reduction for memmap backed arrays.

    a is expected to be an instance of np.ndarray (or np.memmap)
    m is expected to be an instance of np.memmap on the top of the ``base``
    attribute ancestry of a. ``m.base`` should be the real python mmap object.
    """
    if not isinstance(m, np.memmap):
        raise ValueError("m is not a numpy.memmap instance.")
    
    offset = m.offset
    mode = m.mode
    if mode == 'c':
        mode = 'r'
    
    order = 'C' if m.flags['C_CONTIGUOUS'] else 'F'
    
    if a is m:
        return (_strided_from_memmap,
                (m.filename, m.dtype, mode, offset, order, m.shape,
                 m.strides, m.size * m.dtype.itemsize, False))
    else:
        return (_strided_from_memmap,
                (m.filename, a.dtype, mode, offset, order, a.shape,
                 a.strides, m.size * m.dtype.itemsize, False))

def reduce_array_memmap_backward(a):
    """reduce a np.array or a np.memmap from a child process"""
    m = _get_backing_memmap(a)
    if m is not None and isinstance(m, np.memmap):
        return _reduce_memmap_backed(a, m)
    else:
        return (loads, (dumps(a, protocol=HIGHEST_PROTOCOL),))

class ArrayMemmapForwardReducer(object):
    """Reducer callable to dump large arrays to memmap files.

    Parameters
    ----------
    max_nbytes: int
        Threshold to trigger memmapping of large arrays to files created
        a folder.
    temp_folder_resolver: callable
        An callable in charge of resolving a temporary folder name where files
        for backing memmapped arrays are created.
    mmap_mode: 'r', 'r+' or 'c'
        Mode for the created memmap datastructure. See the documentation of
        numpy.memmap for more details. Note: 'w+' is coerced to 'r+'
        automatically to avoid zeroing the data on unpickling.
    verbose: int, optional, 0 by default
        If verbose > 0, memmap creations are logged.
        If verbose > 1, both memmap creations, reuse and array pickling are
        logged.
    prewarm: bool, optional, False by default.
        Force a read on newly memmapped array to make sure that OS pre-cache it
        memory. This can be useful to avoid concurrent disk access when the
        same data array is passed to different worker processes.
    """

    def __init__(self, max_nbytes, temp_folder_resolver, mmap_mode, unlink_on_gc_collect, verbose=0, prewarm=True):
        self._max_nbytes = max_nbytes
        self._temp_folder_resolver = temp_folder_resolver
        self._mmap_mode = mmap_mode
        self.verbose = int(verbose)
        if prewarm == 'auto':
            self._prewarm = not self._temp_folder.startswith(SYSTEM_SHARED_MEM_FS)
        else:
            self._prewarm = prewarm
        self._prewarm = prewarm
        self._memmaped_arrays = _WeakArrayKeyMap()
        self._temporary_memmaped_filenames = set()
        self._unlink_on_gc_collect = unlink_on_gc_collect

    def __reduce__(self):
        args = (self._max_nbytes, None, self._mmap_mode, self._unlink_on_gc_collect)
        kwargs = {'verbose': self.verbose, 'prewarm': self._prewarm}
        return (ArrayMemmapForwardReducer, args, kwargs)

    def __call__(self, a):
        m = _get_backing_memmap(a)
        if m is not None and isinstance(m, np.memmap):
            return _reduce_memmap_backed(a, m)
        if not a.dtype.hasobject and self._max_nbytes is not None and (a.nbytes > self._max_nbytes):
            try:
                os.makedirs(self._temp_folder)
                os.chmod(self._temp_folder, FOLDER_PERMISSIONS)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise e
            try:
                basename = self._memmaped_arrays.get(a)
            except KeyError:
                basename = '{}-{}-{}.pkl'.format(os.getpid(), id(threading.current_thread()), uuid4().hex)
                self._memmaped_arrays.set(a, basename)
            filename = os.path.join(self._temp_folder, basename)
            is_new_memmap = filename not in self._temporary_memmaped_filenames
            self._temporary_memmaped_filenames.add(filename)
            if self._unlink_on_gc_collect:
                resource_tracker.register(filename, 'file')
            if is_new_memmap:
                resource_tracker.register(filename, 'file')
            if not os.path.exists(filename):
                util.debug('[ARRAY DUMP] Pickling new array (shape={}, dtype={}) creating a new memmap at {}'.format(a.shape, a.dtype, filename))
                for dumped_filename in dump(a, filename):
                    os.chmod(dumped_filename, FILE_PERMISSIONS)
                if self._prewarm:
                    load(filename, mmap_mode=self._mmap_mode).max()
            else:
                util.debug('[ARRAY DUMP] Pickling known array (shape={}, dtype={}) reusing memmap file: {}'.format(a.shape, a.dtype, os.path.basename(filename)))
            return (load_temporary_memmap, (filename, self._mmap_mode, self._unlink_on_gc_collect))
        else:
            util.debug('[ARRAY DUMP] Pickling array (NO MEMMAPPING) (shape={},  dtype={}).'.format(a.shape, a.dtype))
            return (loads, (dumps(a, protocol=HIGHEST_PROTOCOL),))

def get_memmapping_reducers(forward_reducers=None, backward_reducers=None, temp_folder_resolver=None, max_nbytes=1000000.0, mmap_mode='r', verbose=0, prewarm=False, unlink_on_gc_collect=True, **kwargs):
    """Construct a pair of memmapping reducer linked to a tmpdir.

    This function manage the creation and the clean up of the temporary folders
    underlying the memory maps and should be use to get the reducers necessary
    to construct joblib pool or executor.
    """
    if forward_reducers is None:
        forward_reducers = {}
    if backward_reducers is None:
        backward_reducers = {}

    if temp_folder_resolver is None:
        temp_folder_resolver = TemporaryResourcesManager(**kwargs).resolve_temp_folder_name

    forward_reduce_ndarray = ArrayMemmapForwardReducer(
        max_nbytes=max_nbytes,
        temp_folder_resolver=temp_folder_resolver,
        mmap_mode=mmap_mode,
        verbose=verbose,
        prewarm=prewarm,
        unlink_on_gc_collect=unlink_on_gc_collect
    )

    forward_reducers.update({
        np.ndarray: forward_reduce_ndarray,
        np.memmap: forward_reduce_ndarray
    })

    backward_reducers.update({
        np.ndarray: reduce_array_memmap_backward,
        np.memmap: reduce_array_memmap_backward
    })

    return forward_reducers, backward_reducers

class TemporaryResourcesManager(object):
    """Stateful object able to manage temporary folder and pickles

    It exposes:
    - a per-context folder name resolving API that memmap-based reducers will
      rely on to know where to pickle the temporary memmaps
    - a temporary file/folder management API that internally uses the
      resource_tracker.
    """

    def __init__(self, temp_folder_root=None, context_id=None):
        self._current_temp_folder = None
        self._temp_folder_root = temp_folder_root
        self._use_shared_mem = None
        self._cached_temp_folders = dict()
        self._id = uuid4().hex
        self._finalizers = {}
        if context_id is None:
            context_id = uuid4().hex
        self.set_current_context(context_id)

    def resolve_temp_folder_name(self):
        """Return a folder name specific to the currently activated context"""
        if self._current_temp_folder is None:
            if self._temp_folder_root is None:
                self._temp_folder_root, self._use_shared_mem = _get_temp_dir(self._id)
            self._current_temp_folder = os.path.join(self._temp_folder_root, self._current_context)
        return self._current_temp_folder

    def _clean_temporary_resources(self, context_id=None, force=False, allow_non_empty=False):
        """Clean temporary resources created by a process-based pool"""
        if context_id is None:
            context_id = self._current_context

        if context_id in self._cached_temp_folders:
            folder_path = self._cached_temp_folders[context_id]
            if os.path.exists(folder_path):
                try:
                    if force or allow_non_empty:
                        delete_folder(folder_path)
                    else:
                        os.rmdir(folder_path)
                except OSError:
                    warnings.warn(f"Failed to delete temporary folder: {folder_path}")
            del self._cached_temp_folders[context_id]

        if context_id in self._finalizers:
            finalizer = self._finalizers.pop(context_id)
            if force:
                finalizer()
